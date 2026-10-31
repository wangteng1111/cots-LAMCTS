"""CUDA-only ray visibility and field-coordinate audit.

The audit keeps legacy stop aiming, aperture-free geometric stop aiming, and
an independent first-surface raster as separate observations. The last two
are diagnostics until their geometric residuals and physical apertures pass.
"""
from __future__ import annotations
import copy,math
import torch
from evaluator.qfilm02 import _surfaces_at,pupil_grid,trace_pupil,_sag_and_grad


def cuda_device(device='cuda:0'):
    d=torch.device(device)
    if d.type!='cuda':raise ValueError('QfilmRW optical computation is GPU-only')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    return d


def step(z,x,y,dz,dx,dy,s,valid,opl=None):
    """Final-root validation; temporary Newton iterates do not clip the ray."""
    den=torch.where(dz.abs()>1e-12,dz,torch.ones_like(dz))
    t=(s.z-z)/den
    for _ in range(16):
        xx=x+t*dx;yy=y+t*dy;r=torch.hypot(xx,yy)
        sag,ds,_=_sag_and_grad(r,s.R,s.conic,s.asphere)
        dr=torch.where(r>1e-14,(xx*dx+yy*dy)/r.clamp_min(1e-14),0.)
        f=z+t*dz-s.z-sag;df=dz-ds*dr
        t=t-torch.where(df.abs()>1e-12,f/torch.where(df.abs()>1e-12,df,1.),0.)
    xx=x+t*dx;yy=y+t*dy;zz=z+t*dz;r=torch.hypot(xx,yy)
    sag,ds,domain=_sag_and_grad(r,s.R,s.conic,s.asphere)
    residual=zz-s.z-sag
    intersection=domain & torch.isfinite(t)&(t>=-1e-8)&torch.isfinite(residual)&(residual.abs()<=1e-7)
    nx=-ds*xx/r.clamp_min(1e-14);ny=-ds*yy/r.clamp_min(1e-14);nz=torch.ones_like(nx)
    norm=torch.sqrt(nx*nx+ny*ny+nz*nz);nx,ny,nz=nx/norm,ny/norm,nz/norm
    flip=nx*dx+ny*dy+nz*dz>0
    nx=torch.where(flip,-nx,nx);ny=torch.where(flip,-ny,ny);nz=torch.where(flip,-nz,nz)
    ci=-(nx*dx+ny*dy+nz*dz);eta=s.n1/s.n2;root=1-eta*eta*(1-ci*ci)
    geometry=valid & intersection & torch.isfinite(root)&(root>=0)
    k=eta*ci-torch.sqrt(root.clamp_min(0));dx2=eta*dx+k*nx;dy2=eta*dy+k*ny;dz2=eta*dz+k*nz
    norm=torch.sqrt(dx2*dx2+dy2*dy2+dz2*dz2).clamp_min(1e-14);dx2,dy2,dz2=dx2/norm,dy2/norm,dz2/norm
    if opl is not None:opl=opl+s.n1*torch.sqrt((xx-x)**2+(yy-y)**2+(zz-z)**2)
    eps=1e-9
    return (zz+eps*dz2,xx+eps*dx2,yy+eps*dy2,dz2,dx2,dy2,geometry,opl),intersection,root,residual


def front_to_stop(surfs,angle,zstart,stop_z,x,y):
    dz=torch.cos(angle).expand_as(x);dx=torch.sin(angle).expand_as(x);dy=torch.zeros_like(x)
    z=torch.full_like(x,zstart);valid=torch.ones_like(x,dtype=torch.bool)
    for s in surfs:
        if s.z>=stop_z-1e-9:break
        (z,x,y,dz,dx,dy,valid,_),_,_,_=step(z,x,y,dz,dx,dy,s,valid)
    t=(stop_z-z)/torch.where(dz.abs()>1e-12,dz,torch.ones_like(dz))
    return x+t*dx,y+t*dy,valid & (t>=-1e-8)&torch.isfinite(t)


def front_ab(surfs,stop_z,device):
    M=torch.eye(2,dtype=torch.float64,device=device);zp=surfs[0].z;seen=False
    for s in surfs:
        if s.z>=stop_z-1e-9:break
        if seen:M=torch.tensor([[1.,s.z-zp],[0.,1.]],dtype=torch.float64,device=device)@M
        power=0. if math.isinf(s.R) else -(s.n2-s.n1)/(s.n2*s.R)
        M=torch.tensor([[1.,0.],[power,s.n1/s.n2]],dtype=torch.float64,device=device)@M
        zp=s.z;seen=True
    if seen:M=torch.tensor([[1.,stop_z-zp],[0.,1.]],dtype=torch.float64,device=device)@M
    return M


def aim(surfs,angle,uv,stop_z,stop_r,zstart):
    tx=stop_r*uv[:,0][None,:].expand(angle.shape[0],-1);ty=stop_r*uv[:,1][None,:].expand_as(tx)
    M=front_ab(surfs,stop_z,uv.device);A=M[0,0];B=M[0,1]
    if abs(float(A))>1e-8:
        x=(tx-B*torch.tan(angle))/A-(surfs[0].z-zstart)*torch.tan(angle);y=ty/A
    else:x=tx-(stop_z-zstart)*torch.tan(angle);y=ty.clone()
    h=max(1e-4,stop_r*1e-4);limit=max(1.,stop_r)*2
    for _ in range(18):
        fx,fy,v=front_to_stop(surfs,angle,zstart,stop_z,x,y)
        fxx,fyx,vx=front_to_stop(surfs,angle,zstart,stop_z,x+h,y)
        fxy,fyy,vy=front_to_stop(surfs,angle,zstart,stop_z,x,y+h)
        a=(fxx-fx)/h;b=(fxy-fx)/h;c=(fyx-fy)/h;d=(fyy-fy)/h;det=a*d-b*c
        good=v&vx&vy&torch.isfinite(det)&(det.abs()>1e-12)
        safe=torch.where(good,det,1.);ex=fx-tx;ey=fy-ty
        ux=((d*ex-b*ey)/safe).clamp(-limit,limit);uy=((-c*ex+a*ey)/safe).clamp(-limit,limit)
        x=x-torch.where(good,ux,0.);y=y-torch.where(good,uy,0.)
    fx,fy,ok=front_to_stop(surfs,angle,zstart,stop_z,x,y)
    residual=torch.hypot(fx-tx,fy-ty)
    return x,y,ok&torch.isfinite(residual)&(residual<=1e-4),residual


def propagate(surfs,angle,x,y,zstart,stop_z,stop_r,film_z,valid,clip=True):
    dz=torch.cos(angle).expand_as(x);dx=torch.sin(angle).expand_as(x);dy=torch.zeros_like(x)
    z=torch.full_like(x,zstart);opl=dx*x+dz*z;crossed=False;losses=[]
    for i,s in enumerate(surfs):
        if not crossed and s.z>=stop_z-1e-9:
            t=(stop_z-z)/torch.where(dz.abs()>1e-12,dz,1.);sx=x+t*dx;sy=y+t*dy
            stop_ok=(sx*sx+sy*sy<=(stop_r+1e-8)**2)&(t>=-1e-8)
            if clip:
                losses.append({'stage':'stop','lost':(valid&~stop_ok).sum(-1).cpu().tolist()});valid &=stop_ok
            crossed=True
        before=valid.clone()
        (z,x,y,dz,dx,dy,valid,opl),inter,root,residual=step(z,x,y,dz,dx,dy,s,valid,opl)
        aperture=getattr(s,'clear_aperture',None)
        ap=torch.ones_like(valid) if aperture is None else x*x+y*y<=(float(aperture)/2+1e-8)**2
        geomloss=before&~valid;aploss=valid&~ap
        if clip:valid &=ap
        losses.append({'stage':f'surface_{i}','source_surface_index':i,
            'geometry_lost':geomloss.sum(-1).cpu().tolist(),'aperture_lost':aploss.sum(-1).cpu().tolist() if clip else [0]*angle.shape[0]})
    tf=(film_z-z)/torch.where(dz.abs()>1e-12,dz,1.)
    valid &=torch.isfinite(tf)&(tf>0)
    return (z,x,y,opl,x+tf*dx,y+tf*dy,valid),losses

@torch.inference_mode()
def bundle(p,wavelength_nm,angles_rad,pupil_n=64,device='cuda:0',entry_raster=True,legacy=True):
    dev=cuda_device(device);surfs=_surfaces_at(p['surfaces'],wavelength_nm)
    angle=torch.as_tensor(angles_rad,dtype=torch.float64,device=dev).reshape(-1,1)
    uv,inside=pupil_grid(pupil_n,dev);stop_z=float(p['stop_z_mm']);rs=float(p['design_spec']['stop_diameter_mm'])/2
    zstart=min(stop_z,surfs[0].z)-20.;film_z=float(p['image_z_mm'])
    x,y,aimed,residual=aim(surfs,angle,uv,stop_z,rs,zstart)
    traced,losses=propagate(surfs,angle,x,y,zstart,stop_z,rs,film_z,aimed&inside[None,:])
    chief_uv=torch.zeros((1,2),dtype=torch.float64,device=dev)
    cx,cy,cv,cr=aim(surfs,angle,chief_uv,stop_z,rs,zstart)
    virtual,_=propagate(surfs,angle,cx,cy,zstart,stop_z,rs,film_z,cv,clip=False)
    physical,_=propagate(surfs,angle,cx,cy,zstart,stop_z,rs,film_z,cv,clip=True)
    old=trace_pupil(surfs,angle,uv,stop_z,rs,film_z) if legacy else None
    raster=None
    if entry_raster:
        radius=float(surfs[0].clear_aperture)/2
        xs=uv[:,0]*radius;ys=uv[:,1]*radius;sag,_,dom=_sag_and_grad(torch.hypot(xs,ys),surfs[0].R,surfs[0].conic,surfs[0].asphere)
        x0=xs[None,:]-(surfs[0].z+sag[None,:]-zstart)*torch.tan(angle);y0=ys[None,:].expand_as(x0)
        raster,_=propagate(surfs,angle,x0,y0,zstart,stop_z,rs,film_z,(inside&dom)[None,:].expand_as(x0).clone())
    rows=[]
    for k,ang in enumerate(angles_rad):
        valid=traced[6][k];count=int(valid.sum());oldcount=int((old[6][k]&inside).sum()) if old else None
        entrycount=int(raster[6][k].sum()) if raster else None
        center=[float(v[k][valid].mean()) for v in (traced[4],traced[5])] if count else None
        chief=[float(virtual[4][k,0]),float(virtual[5][k,0])] if bool(virtual[6][k,0]) else None
        diagnosis='TRANSMITTED' if count else 'NO_TRANSMISSION_IN_SAMPLED_RAYS'
        if oldcount==0 and count:diagnosis='LEGACY_AIMING_FALSE_DARK'
        elif not count and entrycount:diagnosis='STOP_AIMING_UNRESOLVED'
        rows.append({'object_half_angle_deg':math.degrees(ang),'pupil_n':pupil_n,'wavelength_nm':wavelength_nm,'actual_device':str(dev),
            'legacy_transmitted':oldcount,'geometric_aim_converged':int((aimed[k]&inside).sum()),'physical_transmitted':count,
            'entry_raster_transmitted':entrycount,'stop_grid_fill':count/int(inside.sum()),'centroid_mm':center,
            'geometric_chief_mm':chief,'physical_chief_valid':bool(physical[6][k,0]),'diagnosis':diagnosis,
            'losses':[{key:(value[k] if isinstance(value,list) else value) for key,value in item.items()} for item in losses]})
    return rows

@torch.inference_mode()
def efl_gpu(p,wavelength_nm=550.,device='cuda:0'):
    dev=cuda_device(device);surfs=_surfaces_at(p['surfaces'],wavelength_nm)
    M=torch.eye(2,dtype=torch.float64,device=dev);z=surfs[0].z
    for i,s in enumerate(surfs):
        if i:M=torch.tensor([[1.,s.z-z],[0.,1.]],dtype=torch.float64,device=dev)@M
        power=0. if math.isinf(s.R) else -(s.n2-s.n1)/(s.n2*s.R)
        M=torch.tensor([[1.,0.],[power,s.n1/s.n2]],dtype=torch.float64,device=dev)@M;z=s.z
    return float(-1/M[1,0])
