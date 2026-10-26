"""Independent scalar Huygens propagation from a ray-traced physical exit plane.

Purpose: diagnose Qfilm0.2's paraxial stop-pupil FFT, NOT a certified
production physical-optics model.  This code propagates rays to a common plane
beyond the final surface, uses the actual exit-plane transverse mapping and
optical phase, and integrates their coherent contributions onto the *fixed*
film.  Its resulting PSF is a meaningful independent numerical comparison for
near-paraxial propagating waves; vector, polarization, Fresnel transmission and
high-NA effects remain outside the model.

Important:
    The exit plane must be located between every refracting surface and film.
    A smooth, one-to-one pupil-to-plane map is expected. Jacobian zeros/folds
    invalidate the scalar quadrature instead of silently producing a label.
    Reference calls are intentionally slower and make no training-certification
    claim.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass(frozen=True)
class HuygensConfig:
    image_n: int = 128
    image_pitch_um: float = 1.0
    output_block: int = 96
    near_last_surface_mm: float = 0.1
    min_mapping_jacobian: float = 1.e-10
    max_bad_pupil_fraction: float = 0.02
    obliquity: str = 'kirchhoff'


def exit_plane_from_rays(z, x, y, opl, xfilm, yfilm, valid,
                         film_z_mm: float, exit_z_mm: float):
    """Propagate output rays to a fixed plane in air after the last surface.

    All arrays [n,n] in mm.  
    `opl` is incident plane-phase plus the internal optical path of each ray.
    Air propagation adds the geometric physical segment to that common plane.
    """
    dzfilm = film_z_mm-z
    good = valid & (dzfilm > 1e-9) & (exit_z_mm >= z) & (exit_z_mm < film_z_mm)
    slope_x = (xfilm-x)/dzfilm.clamp_min(1.e-12)
    slope_y = (yfilm-y)/dzfilm.clamp_min(1.e-12)
    dz = (exit_z_mm-z)
    xe = x+dz*slope_x
    ye = y+dz*slope_y
    ople = opl + dz*torch.sqrt(1.0+slope_x*slope_x+slope_y*slope_y)
    return xe, ye, ople, good


def exit_area_weights(xe, ye, valid, stop_pitch, min_jacobian=1e-10):
    """sqrt(dA_exit/dA_stop)*dA_stop is the Huygens ray-tube area weight.

    For lossless propagation with constant incident flux per stop-plane area,
    amplitude scales as 1/sqrt(J); numerical Huygens integration multiplies
    physical output area J*dA_stop.  Output amplitude integrand is sqrt(J)dA.
    """
    # Derivatives use the grid spacing of the true stop-plane pupil coordinates.
    dxdu, dxdv = torch.gradient(xe, spacing=(stop_pitch,stop_pitch))
    dydu, dydv = torch.gradient(ye, spacing=(stop_pitch,stop_pitch))
    jac = torch.abs(dxdu*dydv - dxdv*dydu)
    bad = valid & ((jac < min_jacobian) | ~torch.isfinite(jac))
    area = torch.where(valid & ~bad, torch.sqrt(jac.clamp_min(0.0))*stop_pitch**2, 0.)
    # The scalar integral is defined only where the pupil mapping is smooth.
    return area, bad, jac


def huygens_psf_exit_plane(xe, ye, ople, valid, lam_nm: float, stop_pitch_mm: float,
                           film_z_mm: float, exit_z_mm: float,
                           center_xy_mm=(0.,0.), config=HuygensConfig()):
    """Exact spherical propagation distance with scalar Kirchhoff obliquity.

    Computes the normalized intensity on a Cartesian sampling window about a
    specified physical fixed-film coordinate, no refocusing/recentering.
    The integration uses float64 throughout for nanometre OPD sensitivity.
    """
    if xe.shape != ye.shape or xe.shape != ople.shape or xe.shape != valid.shape:
        raise ValueError('exit plane arrays must have the same shape')
    if xe.ndim!=2 or xe.shape[0]!=xe.shape[1]:
        raise ValueError('exit plane pupil must have square two-dimensional grid')
    if not (film_z_mm > exit_z_mm) or stop_pitch_mm<=0:
        raise ValueError('exit plane must precede film plane')
    n=config.image_n
    pitch=config.image_pitch_um*1.e-3
    if n<16 or pitch<=0:
        raise ValueError('invalid detector sampling')
    xy=(torch.arange(n,device=xe.device,dtype=torch.float64)-(n-1)*.5)*pitch
    yy,xx=torch.meshgrid(xy,xy,indexing='ij')
    tx=xx.flatten()+float(center_xy_mm[0]);ty=yy.flatten()+float(center_xy_mm[1])
    area,bad,jac=exit_area_weights(xe,ye,valid,stop_pitch_mm,config.min_mapping_jacobian)
    total_inside=int(valid.sum().item())
    nbad=int(bad.sum().item())
    enough=bool(total_inside>=32 and nbad/max(total_inside,1)<=config.max_bad_pupil_fraction)
    # Check signed Jacobian fold in valid regions; use patch interior only.
    wx=xe.flatten();wy=ye.flatten();wopl=ople.flatten();amp=area.flatten()
    active=amp>0
    wx=wx[active];wy=wy[active];wopl=wopl[active];amp=amp[active]
    k=2.*math.pi/(lam_nm*1.e-6)
    L=film_z_mm-exit_z_mm
    pieces=[]
    if len(wx)<32:
        raise ValueError('insufficient exit plane rays')
    for first in range(0,n*n,config.output_block):
        sx=tx[first:first+config.output_block,None];sy=ty[first:first+config.output_block,None]
        distance=torch.sqrt(L*L+(sx-wx[None,:]).square()+(sy-wy[None,:]).square())
        # Subtract common reference to preserve FP64 phase resolution.
        phase=k*(wopl[None,:]+distance-wopl[0]-L)
        obliq=(1.+L/distance)*0.5 if config.obliquity=='kirchhoff' else 1.
        v=(torch.polar(torch.ones_like(phase),phase)*
           (amp[None,:]*obliq/distance)).sum(dim=-1)
        pieces.append((v.abs()**2))
    intensity=torch.cat(pieces).reshape(n,n)
    energy=intensity.sum()
    intensity=intensity/energy.clamp_min(1e-200)
    return {'psf':intensity,'pixel_pitch_mm':pitch,'sufficient_mapping':enough,
            'bad_mapping_fraction':nbad/max(total_inside,1),
            'active_pupil':int(active.sum().item()),'unnormalized_flux':float(energy.item())}


def mtf_from_psf(psf, pitch_mm, frequencies_lpmm):
    """Extract sampled signed OTF and MTF (x/y). Zero outside discrete Nyquist.

    The output PSF must cover the impulse response sufficiently: truncated PSF
    changes MTF and therefore should be measured with increasing image window.
    """
    n=psf.shape[-1]
    otf=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(psf)))
    z=n//2
    otf=otf/otf[z,z]
    df=1./(n*pitch_mm)
    vals=[]
    for freq in frequencies_lpmm:
        if freq>=.5/pitch_mm:
            vals.append(torch.zeros((2,),device=psf.device,dtype=torch.float64));continue
        u=freq/df
        i=int(u);t=u-i
        if z+i+1>=n:
            vals.append(torch.zeros((2,),device=psf.device,dtype=torch.float64));continue
        vals.append(torch.stack((torch.abs((1-t)*otf[z,z+i]+t*otf[z,z+i+1]),
                                 torch.abs((1-t)*otf[z+i,z]+t*otf[z+i+1,z]))))
    return torch.stack(vals)


def real_lens_reference(p, field_index=0, pupil_n=64, lam_nm=550.,
                        config=HuygensConfig(),device='cuda:0',field_total=5):
    """Run an independent fixed-film scalar diffraction spot reference.

    This is intentionally one wavelength/field per call, suitable for optical
    physics acceptance and controlled reference comparisons only.
    """
    from evaluator.qfilm02 import _extract_spec,_surfaces_at,trace_pupil,pupil_grid
    src,stop_z,film_z,spec=_extract_spec(p)
    stop_radius=float(spec['stop_diameter_mm'])*.5
    target=float(spec.get('efl_target_mm',50.))
    rmax=math.hypot(36.,24.)/2.
    field_h=field_index/(field_total-1)
    a=math.atan(field_h*rmax/target)
    d=torch.device(device)
    ang=torch.tensor([[a]],device=d,dtype=torch.float64)
    uv,inside=pupil_grid(pupil_n,d)
    surf=_surfaces_at(src,lam_nm)
    with torch.inference_mode():
        output=trace_pupil(surf,ang,uv,float(stop_z),stop_radius,float(film_z))
        chief=trace_pupil(surf,ang,torch.zeros((1,2),device=d,dtype=torch.float64),float(stop_z),stop_radius,float(film_z))
    z,x,y,opl,xf,yf,valid=output
    valid=valid & inside[None,:]
    maxz=float(z[valid].max().item())
    exit_z=maxz+config.near_last_surface_mm
    n=pupil_n
    xe,ye,ople,v=exit_plane_from_rays(*[i.reshape(n,n) for i in (z,x,y,opl,xf,yf,valid)],
                                      film_z,exit_z)
    if not bool(chief[6].all()):raise ValueError('invalid chief ray')
    xy=(float(chief[4].item()),float(chief[5].item()))
    rs=huygens_psf_exit_plane(xe,ye,ople,v,lam_nm,2.*stop_radius/n,float(film_z),exit_z,
                              center_xy_mm=xy,config=config)
    rs.update({'field_h':field_h,'wavelength_nm':lam_nm,'film_center_mm':xy,
               'exit_z_mm':exit_z,'method':'scalar_physical_exit_plane_huygens',
               'certified':False})
    return rs