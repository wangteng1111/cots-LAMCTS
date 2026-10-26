"""Experimental physical exit-plane pupil gridding + FFT PSF for Qfilm.

Unlike the legacy stop-plane FFT this constructs the field from the actual
ray-traced exit-plane positions, with a flux-conserving ray-tube Jacobian.
Scalar paraxial focal-plane Fourier propagation is used after gridding;
this is NOT yet validated for high-NA or strong-caustic COTS systems.
"""
from __future__ import annotations
import math
import torch
from evaluator.qfilm02_huygens import exit_area_weights,exit_plane_from_rays


def exit_plane_fft_psf(xe,ye,ople,valid,lam_nm,stop_pitch_mm,
                       film_z_mm,exit_z_mm,center_xy_mm=(0.,0.),nfft=None,
                       margin=0.1):
    """Conservative complex-field CIC resampling of the physical exit plane.

    Parameters are square [p,p] float64 torch arrays; this routine is designed
    for individual field/color diagnostic comparisons. Later it can be batched.
    """
    n=xe.shape[-1]
    if nfft is None:nfft=n*4
    if nfft<n or nfft%2:raise ValueError('invalid FFT size')
    if not (0<margin<0.5):raise ValueError('margin out of range')
    area,bad,jac=exit_area_weights(xe,ye,valid,stop_pitch_mm)
    good=valid & ~bad
    if int(good.sum().item())<32:raise ValueError('insufficient physical pupil points')
    xmin=float(xe[good].min());xmax=float(xe[good].max())
    ymin=float(ye[good].min());ymax=float(ye[good].max())
    # stable padding independent of assignment order
    xa=float(xe[good].min());xb=float(xe[good].max())
    ya=float(ye[good].min());yb=float(ye[good].max())
    xmid=(xa+xb)*0.5;ymid=(ya+yb)*0.5
    hwidth=(xb-xa)*(0.5+margin);hheight=(yb-ya)*(0.5+margin)
    if min(hwidth,hheight)<1e-10:raise ValueError('collapsed exit pupil')
    dx=2*hwidth/n;dy=2*hheight/n
    # Sample locations are cell centers. Overlapping ray contributions add in
    # complex amplitude (important near pupil folds).
    gx=(xe-(xmid-hwidth))/dx-.5
    gy=(ye-(ymid-hheight))/dy-.5
    ix=torch.floor(gx).long();iy=torch.floor(gy).long()
    fx=gx-ix;fy=gy-iy
    L=film_z_mm-exit_z_mm
    if L<=0:raise ValueError('exit plane after film')
    dist=torch.sqrt((float(center_xy_mm[0])-xe).square()+
                    (float(center_xy_mm[1])-ye).square()+L*L)
    optical_path=ople+dist
    # Subtract a common piston to keep phase accurate at nanometre OPD.
    avg=optical_path[good].mean()
    phase=(2*math.pi/(lam_nm*1e-6))*(optical_path-avg)
    wave=torch.polar(torch.ones_like(phase),phase)
    # Exit area quadrature:  sqrt(J)*dA_input / dA_exit, two sided grid CIC.
    amp=wave*area/(dx*dy)
    flat=torch.zeros(n*n,dtype=torch.complex128,device=xe.device)
    for ox in (0,1):
      for oy in (0,1):
        xx=ix+ox; yy=iy+oy
        w=(fx if ox else (1-fx))*(fy if oy else (1-fy))
        mask=good&(xx>=0)&(xx<n)&(yy>=0)&(yy<n)
        loc=(yy[mask]*n+xx[mask]).flatten()
        flat.index_add_(0,loc,(amp[mask]*w[mask]).flatten())
    pupil=flat.reshape(n,n)
    pad=(nfft-n)//2
    field=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(
          torch.nn.functional.pad(pupil,(pad,nfft-n-pad,pad,nfft-n-pad)))))
    psf=field.abs().square()
    psf=psf/psf.sum().clamp_min(1.e-300)
    return {'psf':psf,'pupil':pupil,'pupil_dx_mm':dx,'pupil_dy_mm':dy,
            'detector_pitch_x_mm':lam_nm*1e-6*L/(nfft*dx),
            'detector_pitch_y_mm':lam_nm*1e-6*L/(nfft*dy),
            'pupil_physical_bbox':[xa,xb,ya,yb],
            'bad_mapping_fraction':float(bad.sum().item())/max(float(valid.sum().item()),1),
            'physics_certified':False}


def mtf_exit_fft(result,frequencies_lpmm):
    psf=result['psf'];n=psf.shape[-1];c=n//2
    otf=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(psf)))
    otf=otf/otf[c,c]
    vals=[]
    for f in frequencies_lpmm:
      d=[]
      for p,axis in [(result['detector_pitch_x_mm'],0),(result['detector_pitch_y_mm'],1)]:
        freq_step=1/(n*p);u=f/freq_step; i=int(math.floor(u));a=u-i
        if i+1>=c:d.append(torch.zeros((),device=psf.device,dtype=torch.float64));continue
        if axis==0:val=(1-a)*otf[c,c+i]+a*otf[c,c+i+1]
        else:val=(1-a)*otf[c+i,c]+a*otf[c+i+1,c]
        d.append(val.abs())
      vals.append(torch.stack(d))
    return torch.stack(vals)


def real_lens_exit_fft(p, field_index=0, field_total=3,pupil_n=64,lam_nm=550.,
                       nfft=None,device='cuda:0'):
    """Single-field physical exit-plane FFT diagnostic, preserving actual film."""
    from evaluator.qfilm02 import _extract_spec,_surfaces_at,trace_pupil,pupil_grid
    src,stop_z,film_z,spec=_extract_spec(p)
    Rstop=float(spec['stop_diameter_mm'])/2.
    f_target=float(spec.get('efl_target_mm',50.))
    angle=math.atan((field_index/(field_total-1))*math.hypot(36.,24.)/(2*f_target))
    d=torch.device(device)
    ang=torch.tensor([[angle]],device=d,dtype=torch.float64)
    uv,inside=pupil_grid(pupil_n,d)
    surfs=_surfaces_at(src,lam_nm)
    z,x,y,opl,xf,yf,valid=trace_pupil(surfs,ang,uv,float(stop_z),Rstop,float(film_z))
    chief=trace_pupil(surfs,ang,torch.zeros((1,2),device=d,dtype=torch.float64),float(stop_z),Rstop,float(film_z))
    valid=valid & inside[None,:]
    exit_z=float(z[valid].max())+0.1
    n=pupil_n
    xe,ye,ople,v=exit_plane_from_rays(*[a.reshape(n,n) for a in (z,x,y,opl,xf,yf,valid)],
                                       float(film_z),exit_z)
    out=exit_plane_fft_psf(xe,ye,ople,v,lam_nm,2*Rstop/n,float(film_z),exit_z,
                           (float(chief[4].item()),float(chief[5].item())),
                           nfft=nfft or 4*n)
    out.update({'field_h':field_index/(field_total-1),'wavelength_nm':lam_nm})
    return out