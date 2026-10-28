"""Deterministic coherent ray-grid Huygens diagnostic on fixed physical film.

This is a distinct, optional non-production solver and does not modify Qfilm J.
Allows comparing the exact same Qfilm ray geometry and OPL using two amplitude
models (uniform stop-source and energy/Jacobian weighted outgoing wave).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from evaluator.qfilm02 import _surfaces_at,pupil_grid,trace_pupil
from evaluator.qfilm02_huygens import exit_plane_from_rays,exit_area_weights

@dataclass(frozen=True)
class Propagation:
    pupil_n:int=64
    pixel_n:int=128
    pixel_pitch_um:float=5.
    pixel_block:int=32
    weight_mode:str="uniform_stop"
    obliquity:str="deep_ray"
    def validate(self):
        if self.pupil_n<16 or self.pixel_n<16 or self.pixel_block<1:
            raise ValueError("bad physical-grid dimensions")
        if self.weight_mode not in ("uniform_stop","sqrt_exit_jac"):
            raise ValueError("invalid wave amplitude mode")

def eval_fixed_film(p,wave_nm,field_h,config=Propagation(),device="cuda:0"):
    config.validate()
    stop=float(p["stop_z_mm"]);film=float(p["image_z_mm"])
    rs=float(p["design_spec"]["stop_diameter_mm"])/2.
    alpha=math.atan(field_h*math.hypot(36.,24.)/2/p["design_spec"]["efl_target_mm"])
    angles=torch.tensor([[alpha]],dtype=torch.float64,device=device)
    pupil_n=config.pupil_n
    uv,inside=pupil_grid(pupil_n,device)
    surfs=_surfaces_at(p["surfaces"],wave_nm)
    with torch.inference_mode():
        trace=trace_pupil(surfs,angles,uv,stop,rs,film)
        chief=trace_pupil(surfs,angles,torch.zeros((1,2),device=device,dtype=torch.float64),stop,rs,film)
        z,x,y,opl,xf,yf,valid=trace
        mask=(valid & inside.unsqueeze(0)).reshape(pupil_n,pupil_n)
        zz=z.reshape(pupil_n,pupil_n)
        max_z=float(zz[mask].max())
        exit_z=max_z+0.1
        if exit_z>=film:raise ValueError("exit plane beyond film")
        xe,ye,oe,ok=exit_plane_from_rays(
          *[u.reshape(pupil_n,pupil_n) for u in (z,x,y,opl,xf,yf,(valid & inside.unsqueeze(0)))],
          film,exit_z)
        jac_area,bad,jac=exit_area_weights(xe,ye,ok,2*rs/pupil_n)
        if config.weight_mode=="uniform_stop":
            amp=torch.where(ok,torch.ones_like(jac_area)*(2*rs/pupil_n)**2,0.)
        else:
            amp=jac_area
        slope_x=(xf.reshape(pupil_n,pupil_n)-x.reshape(pupil_n,pupil_n))/(film-zz)
        slope_y=(yf.reshape(pupil_n,pupil_n)-y.reshape(pupil_n,pupil_n))/(film-zz)
        dz=(1+slope_x*slope_x+slope_y*slope_y).rsqrt()
        active=(amp>0)&torch.isfinite(oe)
        X=xe[active];Y=ye[active];OPL=oe[active];A=amp[active];dirz=dz[active]
        center=(float(chief[4].item()),float(chief[5].item()))
        n=config.pixel_n;pitch=config.pixel_pitch_um*1e-3
        coords=(torch.arange(n,device=device,dtype=torch.float64)-(n-1)*.5)*pitch
        yy,xx=torch.meshgrid(coords,coords,indexing="ij")
        px=(xx+center[0]).flatten();py=(yy+center[1]).flatten()
        ref_opl=OPL[0]+(film-exit_z)
        z_distance=film-exit_z
        k=2*math.pi/(wave_nm*1e-6)
        accum=[]
        for start in range(0,n*n,config.pixel_block):
            sx=px[start:start+config.pixel_block,None]
            sy=py[start:start+config.pixel_block,None]
            r=torch.sqrt(z_distance*z_distance+(sx-X[None,:])**2+(sy-Y[None,:])**2)
            phase=torch.remainder((OPL[None,:]+r-ref_opl)/(wave_nm*1e-6),1.)*(2*math.pi)
            ob=(1+dirz[None,:])/2 if config.obliquity=="deep_ray" else (1+z_distance/r)/2
            E=(torch.polar(torch.ones_like(phase),phase)*(A[None,:]*ob/r)).sum(-1)
            accum.append(E)
        field=torch.cat(accum).reshape(n,n)
        psf=field.abs().square();psf=psf/psf.sum()
        return {"psf":psf,"valid_rays":int(active.sum()),"bad_jacobian_fraction":float(bad.sum()/mask.sum()),
                "jac_min":float(jac[active].min()),"jac_p95":float(torch.quantile(jac[active],.95)),
                "center_film_mm":center,"exit_z_mm":exit_z,
                "weight_mode":config.weight_mode,"pupil_n":pupil_n}

def mtf_from_psf_exact(psf,pitch_mm,freqs):
    import numpy as np
    p=psf.detach().cpu().numpy()
    p=p/p.sum()
    n=p.shape[0]
    otf=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(p)))
    otf=abs(otf/otf[n//2,n//2])
    f=np.fft.fftshift(np.fft.fftfreq(n,pitch_mm))
    ix=n//2
    return np.array([[np.interp(fr,f[ix:],otf[ix,ix:]),
                      np.interp(fr,f[ix:],otf[ix:,ix])] for fr in freqs],dtype=float)
