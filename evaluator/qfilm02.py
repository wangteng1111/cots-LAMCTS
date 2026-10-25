"""Alpha Lense Qfilm0.2 optical evaluator (provisional FFT-pupil backend).

Physics trace: sequential spherical/conic/asphere surfaces, Snell refraction,
fixed-film eikonal, scalar exit-wave approximation and FFT PSF/OTF.

IMPORTANT: this is an engineering candidate, NOT certified training ground truth:
rigorous exit-pupil mapping, off-axis Huygens verification, spectral coating
transmission and true photographic polychromatic EE are pending validation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hashlib
import json
import math
import numpy as np
import torch
from evaluator.optical_primitives import Surf, cardinal, front_matrix, cauchy_from_ne_ve

VERSION = 'Qfilm0.2-provisional-fft-v1'
FREQS = (5., 10., 15., 20., 30., 40., 50., 60., 70., 80.)
WAVES = (430., 470., 510., 550., 590., 630., 670.)
WEIGHTS = (0.075, 0.125, 0.16, 0.28, 0.16, 0.125, 0.075)

@dataclass(frozen=True)
class Config:
    fields: int = 100
    pupil_n: int = 64
    fft_n: int = 256
    wavelengths_nm: tuple[float,...] = WAVES
    spectral_weights: tuple[float,...] = WEIGHTS
    frame_width_mm: float = 36.
    frame_height_mm: float = 24.
    target_efl_mm: float = 50.
    frequencies_lpmm: tuple[float,...] = FREQS
    max_fields_per_chunk: int = 20
    def validate(self):
        if self.fields < 2 or self.pupil_n < 16 or self.fft_n < self.pupil_n or self.fft_n % self.pupil_n:
            raise ValueError('invalid field/pupil/fft dimensions')
        if len(self.wavelengths_nm) != len(self.spectral_weights) or any(w <= 0 for w in self.spectral_weights):
            raise ValueError('invalid wavelength weights')
        if min(self.frame_width_mm,self.frame_height_mm,self.target_efl_mm) <= 0:
            raise ValueError('invalid geometry')
    @property
    def signature(self):
        return hashlib.sha256(json.dumps(self.__dict__,sort_keys=True).encode()).hexdigest()


def pupil_grid(n:int,device):
    grid=(torch.arange(n,device=device,dtype=torch.float64)+.5)*2./n-1.
    yy,xx=torch.meshgrid(grid,grid,indexing='ij')
    uv=torch.stack([xx.flatten(),yy.flatten()],dim=-1)
    return uv,torch.sum(uv*uv,dim=-1)<=1.


def _sag_and_grad(r:torch.Tensor,R:float,conic:float,asphere:tuple):
    if math.isinf(R):
        z=torch.zeros_like(r);grad=torch.zeros_like(r);ok=torch.ones_like(r,dtype=torch.bool)
    else:
        c=1./R
        q=1.-(1.+conic)*(c*c)*r*r
        ok=q>1e-14
        sq=torch.sqrt(torch.clamp(q,min=1e-14))
        z=c*r*r/(1.+sq)
        grad=c*r/sq
    for i,coeff in enumerate(asphere):
        power=4+2*i
        z=z+float(coeff)*r.pow(power)
        grad=grad+power*float(coeff)*r.pow(power-1)
    return z,grad,ok


def _refract_step(z,x,y,dz,dx,dy,s:Surf,valid,opl=None):
    # Newton intersection with true conic/asphere sag; fixed iteration count
    t=(s.z-z)/dz.clamp(min=1e-10)
    for _ in range(12):
        xx=x+t*dx; yy=y+t*dy
        r=torch.hypot(xx,yy)
        sag,ds,ok=_sag_and_grad(r,s.R,s.conic,s.asphere)
        drdt=torch.where(r>1e-14,(xx*dx+yy*dy)/r.clamp(min=1e-14),0.)
        f=z+t*dz-s.z-sag
        df=dz-ds*drdt
        t=t-torch.where(torch.abs(df)>1e-12,f/torch.where(torch.abs(df)>1e-12,df,1.),0.)
        valid=valid & ok & (torch.abs(df)>1e-12)
    xx=x+t*dx; yy=y+t*dy; zz=z+t*dz
    rr=torch.hypot(xx,yy)
    _,ds,ok=_sag_and_grad(rr,s.R,s.conic,s.asphere)
    valid=valid & ok & (t>0) & torch.isfinite(t)
    # Surface aperture is stored as a full CLEAR DIAMETER by P2P parser.
    aperture=getattr(s,'clear_aperture',None)
    if aperture is not None and aperture>0:
        valid=valid & (rr<=aperture/2.+1e-9)
    nx=-ds*xx/rr.clamp(min=1e-14);ny=-ds*yy/rr.clamp(min=1e-14)
    nz=torch.ones_like(nx)
    norm=torch.sqrt(nx*nx+ny*ny+nz*nz)
    nx,ny,nz=nx/norm,ny/norm,nz/norm
    dot=nx*dx+ny*dy+nz*dz
    flip=dot>0
    nx=torch.where(flip,-nx,nx);ny=torch.where(flip,-ny,ny);nz=torch.where(flip,-nz,nz)
    eta=s.n1/s.n2
    ci=-(nx*dx+ny*dy+nz*dz)
    root2=1.-eta*eta*(1.-ci*ci)
    valid=valid & (root2>=0)
    k=eta*ci-torch.sqrt(torch.clamp(root2,min=0))
    dx2=eta*dx+k*nx;dy2=eta*dy+k*ny;dz2=eta*dz+k*nz
    nn=torch.sqrt(dx2*dx2+dy2*dy2+dz2*dz2).clamp(min=1e-14)
    dx2,dy2,dz2=dx2/nn,dy2/nn,dz2/nn
    if opl is not None:
        opl=opl+s.n1*torch.sqrt((xx-x)**2+(yy-y)**2+(zz-z)**2)
    eps=1e-9
    return zz+eps*dz2,xx+eps*dx2,yy+eps*dy2,dz2,dx2,dy2,valid,opl


def _front_to_stop(surfs,angles,zstart,stop_z,xstart,ystart):
    dz=torch.cos(angles).expand_as(xstart)
    dx=torch.sin(angles).expand_as(xstart)
    dy=torch.zeros_like(xstart)
    z=torch.full_like(xstart,zstart)
    x,y=xstart,ystart
    valid=torch.ones_like(xstart,dtype=torch.bool)
    for s in surfs:
        if s.z>=stop_z-1e-9:break
        z,x,y,dz,dx,dy,valid,_=_refract_step(z,x,y,dz,dx,dy,s,valid)
    t=(stop_z-z)/dz.clamp(min=1e-12)
    return x+t*dx,y+t*dy,valid & (t>=0)


def _start_for_stop(surfs,angles,uv,stop_z,stop_r,zstart):
    targetx=stop_r*uv[:,0][None,:].expand(angles.shape[0],-1)
    targety=stop_r*uv[:,1][None,:].expand(angles.shape[0],-1)
    x=targetx-(stop_z-zstart)*torch.tan(angles)
    y=targety.clone()
    if not any(s.z<stop_z-1e-9 for s in surfs):
        return x,y,torch.ones_like(x,dtype=torch.bool)
    valid=torch.ones_like(x,dtype=torch.bool)
    # Solve the actual stop-coordinate map, not a paraxial x approximation.
    for _ in range(6):
        fx,fy,v0=_front_to_stop(surfs,angles,zstart,stop_z,x,y)
        h=1e-3
        fxx,fyx,vx=_front_to_stop(surfs,angles,zstart,stop_z,x+h,y)
        fxy,fyy,vy=_front_to_stop(surfs,angles,zstart,stop_z,x,y+h)
        j00=(fxx-fx)/h;j10=(fyx-fy)/h
        j01=(fxy-fx)/h;j11=(fyy-fy)/h
        det=j00*j11-j01*j10
        ok=v0 & vx & vy & (det.abs()>1e-12)
        ex=fx-targetx;ey=fy-targety
        x=x-torch.where(ok,(j11*ex-j01*ey)/torch.where(ok,det,1.),0.)
        y=y-torch.where(ok,(-j10*ex+j00*ey)/torch.where(ok,det,1.),0.)
        valid=valid & ok
    fx,fy,ok=_front_to_stop(surfs,angles,zstart,stop_z,x,y)
    valid=valid & ok & ((fx-targetx)**2+(fy-targety)**2<1e-8)
    return x,y,valid


def trace_pupil(surfs,angles,uv,stop_z,stop_r,film_z):
    device=uv.device
    zstart=min(stop_z,surfs[0].z)-20.
    x,y,valid=_start_for_stop(surfs,angles,uv,stop_z,stop_r,zstart)
    dz=torch.cos(angles).expand_as(x);dx=torch.sin(angles).expand_as(x)
    dy=torch.zeros_like(x)
    z=torch.full_like(x,zstart)
    opl=dx*x+dz*z
    crossed=False
    for s in surfs:
        if not crossed and s.z>=stop_z-1e-9:
            tt=(stop_z-z)/dz.clamp(min=1e-12)
            sx=x+tt*dx;sy=y+tt*dy
            valid=valid & (sx*sx+sy*sy<=(stop_r+1e-8)**2)
            crossed=True
        z,x,y,dz,dx,dy,valid,opl=_refract_step(z,x,y,dz,dx,dy,s,valid,opl)
    tf=(film_z-z)/dz.clamp(min=1e-12)
    valid=valid & (tf>0) & torch.isfinite(tf)
    xfilm=x+tf*dx;yfilm=y+tf*dy
    return (z,x,y,opl,xfilm,yfilm,valid)


def fft_pupil_metrics(wave_mm,mask,lam_nm,fno,nfft=256,frequencies=FREQS):
    """Return monochromatic PSF, complex OTF T/S samples, and pixel pitch in mm.

    Scalar paraxial pupil propagation: approximate until external Huygens validated.
    """
    if wave_mm.shape[-1]!=wave_mm.shape[-2]:raise ValueError('square pupil required')
    n=wave_mm.shape[-1]
    lam_mm=lam_nm*1e-6
    denom=mask.sum((-1,-2),keepdim=True).clamp(min=1)
    safe_wave=torch.where(mask,torch.nan_to_num(wave_mm,nan=0.,posinf=0.,neginf=0.),0.)
    piston=(safe_wave.sum((-1,-2),keepdim=True)/denom)
    phase=(2.*math.pi/lam_mm)*(safe_wave-piston)
    pupil=torch.polar(mask.double(),phase)
    pad=(nfft-n)//2
    pupil_pad=torch.nn.functional.pad(pupil,(pad,nfft-n-pad,pad,nfft-n-pad))
    field=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(pupil_pad,dim=(-2,-1))),dim=(-2,-1))
    psf=field.abs().square();psf=psf/psf.sum((-1,-2),keepdim=True).clamp(min=1e-30)
    otf=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(psf,dim=(-2,-1))),dim=(-2,-1))
    center=nfft//2
    otf=otf/otf[...,center,center].real.unsqueeze(-1).unsqueeze(-1)
    pitch_mm=lam_mm*abs(fno)*n/nfft
    df=1./(nfft*pitch_mm)
    # Complex bilinear along exact T/S principal axes; chromatic shift done later.
    vals=[]
    for freq in frequencies:
        ix=freq/df
        lo=int(ix);t=ix-lo
        if lo+center+1>=nfft: raise ValueError('frequency outside sampled OTF')
        vals.append(torch.stack([(1-t)*otf[...,center,center+lo]+t*otf[...,center,center+lo+1],
                                 (1-t)*otf[...,center+lo,center]+t*otf[...,center+lo+1,center]],dim=-1))
    otfs=torch.stack(vals,dim=-2)
    ideal_pupil=mask.double().to(torch.complex128)
    ideal_pad=torch.nn.functional.pad(ideal_pupil,(pad,nfft-n-pad,pad,nfft-n-pad))
    ideal=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(ideal_pad,dim=(-2,-1))),dim=(-2,-1)).abs().square()
    ideal=ideal/ideal.sum((-1,-2),keepdim=True).clamp(min=1e-30)
    strehl=(psf.amax(dim=(-1,-2))/ideal.amax(dim=(-1,-2)).clamp(min=1e-30)).clamp(min=0,max=1.000001)
    return psf,otfs,pitch_mm,strehl


def area_weights(fields,width,height,samples=512):
    x=(np.arange(samples)+.5)*width/samples-width/2
    y=(np.arange(samples)+.5)*height/samples-height/2
    r=np.hypot(x[None,:],y[:,None])/(math.hypot(width,height)/2)
    hist=np.histogram(np.minimum(r,1.-1e-12),bins=np.linspace(0,1,fields+1))[0].astype(float)
    return hist/hist.sum()


def score_image(q,weights,task_weights=(1.,.1,.05,.05)):
    """Deterministic task scalar, separate from optical physics. Lower is better."""
    qa=np.asarray(q,dtype=float)
    mask=(qa[:,30]>.5)
    if not np.all(mask):return {'valid':False,'J_task':None,'invalid_fields':np.where(~mask)[0].tolist()}
    detail=np.mean((1.-np.clip(qa[:,10:30],0,1))**2,axis=1)
    chroma=np.minimum(qa[:,6].clip(min=0)/10.,5.)**2
    illum=np.maximum(-np.log(np.clip(qa[:,1],1e-4,1)),0)
    distortion=(np.abs(qa[:,3])/1.)**2
    comp=np.stack((detail,chroma,illum,distortion),axis=-1)
    loss=comp@np.asarray(task_weights)
    w=np.asarray(weights,float);w=w/w.sum()
    order=np.argsort(-loss)
    rem=.10;tail=0.
    for j in order:
        take=min(rem,w[j]);tail+=take*loss[j];rem-=take
        if rem<=1e-12:break
    cvar=tail/.10
    return {'valid':True,'J_task':float(.75*(loss@w)+.25*cvar),'J_area':float(loss@w),'J_tail':float(cvar),
            'J_components':(comp*w[:,None]).sum(0).tolist()}


def _extract_spec(p):
    if isinstance(p,dict):return p['surfaces'],p.get('stop_z_mm'),p.get('image_z_mm'),p.get('design_spec',{})
    return p.surfaces,p.stop_z_mm,p.image_z_mm,p.design_spec


def _surfaces_at(surfs,lam):
    z=0.;n1=1.;out=[]
    for s in surfs:
        def val(key,default=None):return s.get(key,default) if isinstance(s,dict) else getattr(s,key,default)
        nbase=float(val('n_after'))
        n2=1. if nbase<=1.000001 else cauchy_from_ne_ve(nbase,float(val('v_after')),lam)
        out.append(Surf(z,float(val('radius')),n1,n2,float(val('conic',0.)),tuple(val('asphere',()) or ())))
        # Surf is a frozen dataclass? currently mutable, attach clear diameter.
        out[-1].clear_aperture=val('clear_aperture')
        z+=float(val('thickness'));n1=n2
    if abs(n1-1)>1e-7:raise ValueError('last medium must be air')
    return out


def evaluate(p,config:Config=Config(),device='cpu'):
    config.validate();device=torch.device(device)
    src,stop_z,film_z,spec=_extract_spec(p)
    if stop_z is None or film_z is None:raise ValueError('stop and film must be explicitly fixed')
    sd=spec.get('stop_diameter_mm')
    if sd is None or sd<=0:raise ValueError('explicit stop_diameter_mm required')
    stop_r=float(sd)/2.
    uv,inside=pupil_grid(config.pupil_n,device)
    field_h=np.linspace(0,1,config.fields)
    rmax=math.hypot(config.frame_width_mm,config.frame_height_mm)/2
    field_angle=np.arctan(field_h*rmax/config.target_efl_mm)
    waves=list(config.wavelengths_nm);weights=np.asarray(config.spectral_weights,dtype=float);weights/=weights.sum()
    wavelengths_data=[]
    for lam in waves:
        surfs=_surfaces_at(src,lam)
        efl,bfl=cardinal(surfs)
        fm=front_matrix(surfs,float(stop_z))
        am=1. if fm is None else float(fm[0,0])
        fno=abs(efl*am)/(2.*stop_r)
        if not np.isfinite(fno) or fno<=0:raise ValueError('invalid paraxial focal length/F-number')
        wavelengths_data.append((lam,surfs,fno,efl))
    # Each λ uses a physically shared fixed film, and its own chief reference
    # solely for numerical conditioning. Chromatic chief-ray shift restored in OTF.
    chunks=[]
    for start in range(0,config.fields,config.max_fields_per_chunk):
        end=min(config.fields,start+config.max_fields_per_chunk)
        angle=torch.tensor(field_angle[start:end],device=device,dtype=torch.float64)[:,None]
        refs=[];sampled=[]
        for lam,surfs,fno,efl in wavelengths_data:
            chief=trace_pupil(surfs,angle,torch.zeros((1,2),device=device,dtype=torch.float64),float(stop_z),stop_r,float(film_z))
            refs.append((chief[4].squeeze(-1),chief[5].squeeze(-1)))
        for li,(lam,surfs,fno,efl) in enumerate(wavelengths_data):
            z,x,y,opl,xfilm,yfilm,valid=trace_pupil(surfs,angle,uv,float(stop_z),stop_r,float(film_z))
            rx,ry=refs[li]
            dist=torch.sqrt((float(film_z)-z)**2+(rx[:,None]-x)**2+(ry[:,None]-y)**2)
            wave=opl+dist
            n=config.pupil_n
            pupil_mask=(valid & inside[None,:]).reshape(-1,n,n)
            wave=wave.reshape(-1,n,n)
            psf,otf,pitch,strehl=fft_pupil_metrics(wave,pupil_mask,lam,fno,config.fft_n,config.frequencies_lpmm)
            # NOTE: the OTF is still referenced to its monochromatic chief ray.
            sampled.append((lam,otf,psf,pitch,strehl,pupil_mask,wave,valid, fno))
        # Chromatic relative displacement: x is sagittal image-axis (tangential field axis).
        bx,by=refs[len(refs)//2]
        polys=[]
        for li,(lam,otf,psf,pitch,strehl,pm,wave,valid,fno) in enumerate(sampled):
            sx=refs[li][0]-bx;sy=refs[li][1]-by
            f=torch.tensor(config.frequencies_lpmm,dtype=torch.float64,device=device)[None,:]
            shift_t=torch.exp(-2j*math.pi*f*sx[:,None])
            shift_s=torch.exp(-2j*math.pi*f*sy[:,None])
            phase_shift=torch.stack([shift_t,shift_s],dim=-1)
            polys.append(otf*phase_shift*float(weights[li]))
        mtf=torch.stack(polys).sum(0).abs().clamp(0.,1.) # [fields, frequencies, TS]
        obs=torch.zeros((end-start,32),device=device,dtype=torch.float64)
        obs[:,0]=torch.tensor(field_h[start:end],dtype=torch.float64,device=device)
        inside_count=float(inside.sum().item())
        fill=[(item[5].flatten(1).sum(-1)/inside_count) for item in sampled]
        fill=torch.stack(fill,dim=0)
        obs[:,2]=(fill*torch.tensor(weights,device=device)[:,None]).sum(0)
        obs[:,1]=obs[:,2]*torch.cos(angle.squeeze(-1)).pow(4)
        obs[:,3]=100*(bx-(config.target_efl_mm*torch.tan(angle.squeeze(-1))))/(config.target_efl_mm*torch.tan(angle.squeeze(-1))).clamp(min=1e-6)
        obs[obs[:,0]==0,3]=0.
        obs[:,4]=(refs[-1][0]-bx)*1e3;obs[:,5]=(refs[0][0]-bx)*1e3
        obs[:,6]=torch.max(torch.stack([x[0] for x in refs]),dim=0).values.sub(torch.min(torch.stack([x[0] for x in refs]),dim=0).values)*1e3
        # EE on the GREEN reference wave only; full spectral EE is not yet certified.
        _,_,psfg,pitchg,strehlg,pupg,waveg,validg,_=sampled[len(waves)//2]
        obs[:,7:9]=encircled_energy(psfg,pitchg,(.5,.8)) * 1e3
        obs[:,9]=strehlg
        obs[:,10:20]=mtf[:,:,0];obs[:,20:30]=mtf[:,:,1]
        obs[:,30]=(fill.min(dim=0).values>.20).double()
        obs[:,31]=bx
        wavem=torch.where(pupg,waveg,0.)
        count=pupg.sum((-1,-2),keepdim=True).clamp(min=1)
        wmean=wavem.sum((-1,-2),keepdim=True)/count
        centered=torch.where(pupg,torch.nan_to_num(waveg-wmean,nan=0.,posinf=0.,neginf=0.),0.)
        basis=wavefront_basis(config.pupil_n,device)
        qw=torch.zeros((end-start,16),device=device,dtype=torch.float64)
        for k in range(12):
            phi=basis[k][None,:,:]*pupg
            qw[:,k]=(centered*phi).sum((-1,-2))/(phi*phi).sum((-1,-2)).clamp(min=1e-10)*1e6
        qw[:,12]=torch.sqrt((centered.square().sum((-1,-2))/count.squeeze(-1).squeeze(-1)))*1e6
        qw[:,13]=qw[:,3]
        qw[:,14]=torch.rad2deg(angle.squeeze(-1))
        qw[:,15]=obs[:,2]
        chunks.append((obs.cpu().numpy(),qw.cpu().numpy()))
    q=np.concatenate([x[0] for x in chunks]);qw=np.concatenate([x[1] for x in chunks])
    aw=area_weights(config.fields,config.frame_width_mm,config.frame_height_mm)
    j=score_image(q,aw)
    return {'schema':'QfilmResult-0.2','version':VERSION,'certified':False,'physics_status':'provisional_scalar_fft_exit_pupil_approx',
            'config_hash':config.signature,'Q_image':q,'Q_wavefront':qw,'area_weights':aw,'quality':j,
            'efl_mm':float(wavelengths_data[len(waves)//2][3]),'fno':float(wavelengths_data[len(waves)//2][2]),
            'assumptions':['cauchy_nV_dispersion','unit_pupil_amplitude','cos4_illumination_proxy',
                           'green_only_encircled_energy','scalar_fft_no_huygens_certification'],
            'backend':str(device)}


def wavefront_basis(n,device):
    uv,mask=pupil_grid(n,device)
    x=uv[:,0].reshape(n,n);y=uv[:,1].reshape(n,n)
    r2=x*x+y*y
    raw=[x,y,2*r2-1,x*x-y*y,2*x*y,(3*r2-2)*x,(3*r2-2)*y,
         6*r2*r2-6*r2+1,(4*r2-3)*(x*x-y*y),(4*r2-3)*2*x*y,
         (10*r2*r2-12*r2+3)*x,(10*r2*r2-12*r2+3)*y]
    # Reproducible masked orthogonal basis, every component has rms=1 inside disk.
    basis=[]
    for v in raw:
        v=torch.where(mask.reshape(n,n),v,0.).clone()
        for b in basis:v=v-(v*b).sum()/b.square().sum().clamp(min=1e-12)*b
        norm=torch.sqrt((v.square().sum()/mask.sum().clamp(min=1)).clamp(min=1e-18))
        basis.append(v/norm)
    return basis


def encircled_energy(psf,pitch_mm,fractions=(.5,.8)):
    n=psf.shape[-1]
    coord=(torch.arange(n,device=psf.device,dtype=torch.float64)-n//2)*pitch_mm
    rr=torch.hypot(coord[None,:],coord[:,None]).flatten()
    idx=rr.argsort()
    rad=rr[idx]
    intensity=psf.flatten(1)[:,idx]
    cdf=intensity.cumsum(-1)
    vals=[]
    for fraction in fractions:
        k=torch.searchsorted(cdf.contiguous(),torch.full((psf.shape[0],1),fraction,dtype=cdf.dtype,device=cdf.device))
        vals.append(rad[k.clamp(0,len(rad)-1).squeeze(-1)])
    return torch.stack(vals,dim=1)


def compact_json(r):
    def conv(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,(np.floating,np.integer)):return v.item()
        return v
    return json.dumps({k:conv(v) for k,v in r.items()},ensure_ascii=False)