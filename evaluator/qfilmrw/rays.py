"""Thick-lens ray state. Geometry primitives are shared; old PSF solvers are not."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from evaluator.qfilm02 import _surfaces_at,pupil_grid,trace_pupil
from evaluator.qfilm04_wavefront import fixed_exit_plane

class RWError(ValueError):
    def __init__(self,code,message):super().__init__(message);self.code=code

def masked_gradient(value,valid,spacing,dim):
    """Centered interior, one-sided boundary derivatives; no pupil erosion."""
    forward=torch.roll(value,-1,dim);backward=torch.roll(value,1,dim)
    vf=torch.roll(valid,-1,dim).clone();vb=torch.roll(valid,1,dim).clone()
    end=[slice(None)]*2;end[dim]=-1;vf[tuple(end)]=False
    end[dim]=0;vb[tuple(end)]=False
    derivative=torch.where(vf&vb,(forward-backward)/(2*spacing),torch.where(vf,(forward-value)/spacing,(value-backward)/spacing))
    # Second-order one-sided formulas where two interior neighbors exist.
    vff=torch.roll(vf,-1,dim)&vf;vbb=torch.roll(vb,1,dim)&vb
    derivative=torch.where(vf&~vb&vff,(-3*value+4*forward-torch.roll(value,-2,dim))/(2*spacing),derivative)
    derivative=torch.where(vb&~vf&vbb,(3*value-4*backward+torch.roll(value,2,dim))/(2*spacing),derivative)
    good=valid&(vf|vb)&torch.isfinite(derivative)
    return torch.where(good,derivative,0.),good

def mapping_derivatives(x,y,valid,spacing):
    xu,a=masked_gradient(x,valid,spacing,1);xv,b=masked_gradient(x,valid,spacing,0)
    yu,c=masked_gradient(y,valid,spacing,1);yv,d=masked_gradient(y,valid,spacing,0)
    jac=xu*yv-xv*yu;good=a&b&c&d&torch.isfinite(jac)&(jac.abs()>1e-14)
    folded=False
    if bool((jac[good]>0).any()) and bool((jac[good]<0).any()):
        from scipy.ndimage import label
        # Keep singular nodes when identifying connected transmitted patches;
        # otherwise a caustic line splits one folded patch into two apparently
        # harmless components of opposite orientation.
        labels,count=label(valid.cpu().numpy());values=jac.cpu().numpy();usable=good.cpu().numpy()
        folded=any(np.any(values[(labels==i)&usable]>0) and np.any(values[(labels==i)&usable]<0) for i in range(1,count+1))
    return (xu,xv,yu,yv),jac,good,folded

@dataclass
class RayState:
    x:torch.Tensor
    y:torch.Tensor
    opl:torch.Tensor
    area:torch.Tensor
    field_amplitude:torch.Tensor
    ray_power:torch.Tensor
    active:torch.Tensor
    exit_z_mm:float
    film_z_mm:float
    wavelength_nm:float
    field_h:float
    center_mm:tuple
    metadata:dict
    @property
    def amplitude_area(self):return self.field_amplitude*self.area

@torch.inference_mode()
def trace_lens(p,wavelength_nm,field_h,pupil_n,device='cpu',center_mm=None,frame_mm=(36.,24.)):
    if not (0<=field_h<=1) or pupil_n<16:raise ValueError('invalid condition')
    device=torch.device(device)
    surfs=_surfaces_at(p['surfaces'],wavelength_nm)
    exit_z=fixed_exit_plane(p,wavelength_nm)
    film_z=float(p['image_z_mm']);stop_z=float(p['stop_z_mm'])
    rs=float(p['design_spec']['stop_diameter_mm'])/2
    alpha=math.atan(field_h*math.hypot(*frame_mm)/2/float(p['design_spec']['efl_target_mm']))
    angle=torch.tensor([[alpha]],dtype=torch.float64,device=device)
    uv,inside=pupil_grid(pupil_n,device)
    tr=trace_pupil(surfs,angle,uv,stop_z,rs,film_z)
    z,x,y,opl,xf,yf,valid=[v.reshape(pupil_n,pupil_n) for v in tr]
    mask=valid & inside.reshape(pupil_n,pupil_n)
    count=int(mask.sum())
    # Aiming/intersection failures and physical occlusion are not distinguishable
    # in the shared legacy trace. Do not turn zero rays into a lens-failure label.
    if count==0:raise RWError('RAY_AIMING_OR_OCCLUSION_UNRESOLVED','zero traced rays; requires independent ray-aiming/occlusion diagnosis')
    if count<32:raise RWError('SOURCE_SAMPLING_FAILURE','too few transmitted samples; not proof of physical blockage')
    tf=film_z-z
    sx=(xf-x)/tf;sy=(yf-y)/tf
    xe=x+(exit_z-z)*sx;ye=y+(exit_z-z)*sy
    oe=opl+(exit_z-z)*torch.sqrt(1+sx*sx+sy*sy)
    du=2*rs/pupil_n
    _,jac,good,fold=mapping_derivatives(xe,ye,mask,du)
    bad=mask & ~good
    if fold:raise RWError('UNSUPPORTED_EXIT_MAPPING','folded exit map needs a multi-branch field representation')
    active=good & torch.isfinite(oe)
    if int(active.sum())<32:raise RWError('SOURCE_SAMPLING_FAILURE','insufficient nonsingular derivative stencils')
    area=torch.where(active,jac.abs()*du**2,0.)
    power=torch.where(active,torch.full_like(area,du**2),0.)
    amplitude=torch.where(active,torch.sqrt(power/area.clamp_min(1e-300)),0.)
    chief=trace_pupil(surfs,angle,torch.zeros((1,2),device=device,dtype=torch.float64),stop_z,rs,film_z)
    if center_mm is None:
        if bool(chief[6].all()):center_mm=(float(chief[4].item()),float(chief[5].item()))
        else:
            center_mm=tuple(float((v[active]*power[active]).sum()/power[active].sum()) for v in (xf,yf))
    return RayState(xe,ye,oe,area,amplitude,power,active,exit_z,film_z,float(wavelength_nm),float(field_h),tuple(center_mm),
        {'pupil_n':pupil_n,'valid_rays':count,'active_nodes':int(active.sum()),'reference_ray_blocked':not bool(chief[6].all()),
         'geometric_fill':count/int(inside.sum()),'excluded_stencil_fraction':float((mask & ~active).sum())/count,
         'singular_stencil_count':int(bad.sum()),'transport':'uniform stop-plane scalar z-flux; no Fresnel/coating/polarization',
         'transport_validated':False,'mapping_orientation':int(torch.sign(jac[active].median())),
         'actual_device':str(device)})
