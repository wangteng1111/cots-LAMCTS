#!/usr/bin/env python3
"""Qfilm vs pinned DeepLens: IDENTICAL incoming rays, phase and exit pupil audit.

No independent random pupil sampling, no individual refocus or PSF recentering.
Separates actual ray OPL, WFE, pupil map, Jacobian and amplitude.
The ray comparison works upstream of image propagation and never changes J.
"""
from __future__ import annotations
import sys,os,json,math,traceback,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from evaluator.qfilm02 import _surfaces_at,pupil_grid,_start_for_stop,trace_pupil
from evaluator.qfilm02_huygens import exit_area_weights
from validation.qfilm_deeplens_same_prescription import thin_lens,internal_stop_positive_lens,conversion,BK7
from deeplens import GeoLens
from deeplens.light import Ray
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_common_ray"
OUT.mkdir(parents=True,exist_ok=True)
TARGET=OUT/"matched_incident_rays_opl_opd_exit_jacobian.json"
RADIUS=math.hypot(36.,24.)/2
CASES=[("singlet",550.,0.),("singlet",630.,0.),("singlet",670.,0.),("singlet",670.,.1),
       ("singlet",670.,.5),("internal_stop_asphere",550.,0.),
       ("internal_stop_asphere",630.,0.),("internal_stop_asphere",670.,0.),
       ("internal_stop_asphere",670.,.1),("internal_stop_asphere",670.,1.)]
def metrics(x,mask):
 v=x[mask]
 return {"n":int(v.numel()),"max_abs":float(v.abs().max()),"mean_abs":float(v.abs().mean()),
  "rms":float(v.square().mean().sqrt()),"p95_abs":float(torch.quantile(v.abs(),.95))}
def run_case(lens,p,w,h,n=64,device="cuda:0"):
 stop=float(p["stop_z_mm"]);rs=p["design_spec"]["stop_diameter_mm"]/2.
 film=float(p["image_z_mm"]);offset=max(0.,-min(0.,stop))
 zstart=min(stop,0.)-20.
 a=math.atan(h*RADIUS/p["design_spec"]["efl_target_mm"])
 ang=torch.tensor([[a]],device=device,dtype=torch.float64)
 uv,inside=pupil_grid(n,device)
 ss=_surfaces_at(p["surfaces"],w)
 with torch.inference_mode():
  x0,y0,valid0=_start_for_stop(ss,ang,uv,stop,rs,zstart)
  z,x,y,opl,xf,yf,vq=trace_pupil(ss,ang,uv,stop,rs,film)
  chief=trace_pupil(ss,ang,torch.zeros((1,2),dtype=torch.float64,device=device),stop,rs,film)
 ux=torch.sin(ang).expand_as(x0);uz=torch.cos(ang).expand_as(x0);uy=torch.zeros_like(ux)
 o=torch.stack((x0.flatten(),y0.flatten(),torch.full_like(x0.flatten(),zstart+offset)),dim=-1)
 d=torch.stack((ux.flatten(),uy.flatten(),uz.flatten()),dim=-1)
 with torch.inference_mode():
  ray=Ray(o,d,float(w)*1e-3,is_coherent=True,device=device)
  ray,_=lens.trace(ray)
  dv=ray.is_valid>0
  dx=ray.d[:,0];dy=ray.d[:,1];dz=ray.d[:,2]
  qmask=inside.flatten() & vq.flatten() & dv & valid0.flatten()
  # Qfilm optical path includes incident plane-wave eikonal at launch plane.
  # DeepLens coherent OPL starts at 0; restore exact same incident plane phase.
  ray_plane=(o[:,0]*d[:,0]+o[:,1]*d[:,1]+(o[:,2]-offset)*d[:,2])
  opl_deep=ray.opl.flatten()+ray_plane
  pq=torch.stack((x.flatten(),y.flatten(),z.flatten()),dim=-1)
  pd=ray.o.clone();pd[:,2]-=offset
  qz,zmax=z.flatten(),float(z.flatten()[qmask].max())
  dzfilm=film-z.flatten()
  qdx=(xf.flatten()-x.flatten())/dzfilm; qdy=(yf.flatten()-y.flatten())/dzfilm
  zexit=max(float(pd[:,2][qmask].max()),zmax)+.1
  assert zexit<film
  qxe=x.flatten()+(zexit-z.flatten())*qdx
  qye=y.flatten()+(zexit-z.flatten())*qdy
  qle=opl.flatten()+(zexit-z.flatten())*torch.sqrt(1+qdx*qdx+qdy*qdy)
  with torch.inference_mode():
   rem=(zexit-pd[:,2])/dz
   dxe=pd[:,0]+rem*dx;dye=pd[:,1]+rem*dy
   dle=opl_deep+rem
  pixelx=float(chief[4].item());pixely=float(chief[5].item())
  qr=torch.sqrt((film-zexit)**2+(qxe-pixelx)**2+(qye-pixely)**2)
  dr=torch.sqrt((film-zexit)**2+(dxe-pixelx)**2+(dye-pixely)**2)
  wq=qle+qr;wd=dle+dr
  # OPD piston removed per *matched common* entrance rays. Units µm.
  diffOPD=(wq-wd)
  diffOPD=diffOPD-diffOPD[qmask].mean()
  qOPD=wq-wq[qmask].mean()
  dOPD=wd-wd[qmask].mean()
  pitch=2*rs/n
  jq,_,jqa=exit_area_weights(qxe.reshape(n,n),qye.reshape(n,n),qmask.reshape(n,n),pitch)
  jd,_,jda=exit_area_weights(dxe.reshape(n,n),dye.reshape(n,n),qmask.reshape(n,n),pitch)
  # area amplitude sqrt(J), normalized RMS, remove photometric constant
  # Finite differences at a clipped-pupil edge read invalid neighboring
  # coordinates. Compare Jacobians only on the common valid stencil.
  common=qmask.reshape(n,n)
  interior=torch.zeros_like(common)
  interior[1:-1,1:-1]=(common[1:-1,1:-1]&common[:-2,1:-1]&common[2:,1:-1]
                       &common[1:-1,:-2]&common[1:-1,2:])
  aq=jq[interior];ad=jd[interior]
  an=(aq/aq.square().mean().sqrt())-(ad/ad.square().mean().sqrt())
  rr={
    "lambda_nm":w,"h":h,"pupil_n":n,"number_common_valid":int(qmask.sum()),
    "qfilm_valid":int((inside.flatten()&vq.flatten()).sum()),"deeplens_valid":int((inside.flatten()&dv).sum()),
    "last_refracting_surface_xy_position_mm":metrics(torch.linalg.vector_norm(pq-pd,dim=-1),qmask),
    "qfilm_deeplens_exit_coordinate_x_mm":metrics(qxe-dxe,qmask),
    "qfilm_deeplens_exit_coordinate_y_mm":metrics(qye-dye,qmask),
    "film_ray_intercept_x_mm":metrics(xf.flatten()-(pd[:,0]+(film-pd[:,2])*dx/dz),qmask),
    "corrected_OPL_last_surface_um":metrics(1e3*((opl.flatten()-opl_deep)-(opl.flatten()-opl_deep)[qmask].mean()),qmask),
    "corrected_OPD_full_film_um":metrics(diffOPD*1e3,qmask),
    "OPD_Qfilm_rms_um":float(qOPD[qmask].square().mean().sqrt()*1e3),
    "OPD_DeepLens_rms_um":float(dOPD[qmask].square().mean().sqrt()*1e3),
    "relative_complex_phase_phasor_max":metrics(torch.abs(torch.exp(2j*math.pi*diffOPD/(w*1e-6))-1),qmask),
    "jacobian_comparison_common_interior_rays":int(interior.sum()),
    "exit_Jacobian_map_ratio_p95":float(torch.quantile((jqa[interior]/jda[interior]).abs(),.95)),
    "exit_Jacobian_map_ratio_median":float(torch.quantile((jqa[interior]/jda[interior]).abs(),.5)),
    "normalized_exit_integrand_amplitude_difference":metrics(an,torch.ones_like(an,dtype=torch.bool)),
    "stop_center_film_reference_x_mm":pixelx,
    "exit_plane_mm":zexit}
  return rr
def main():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 systems={"singlet":p,"internal_stop_asphere":internal_stop_positive_lens()}
 out={"schema":"Qfilm0.3-common-ray-opl-opd-exit-jacobian-v1",
      "matched_incoming_ray_geometry":True,"wavelength_cases":[],"status":"not physics certified"}
 for name,p in systems.items():
  data,_=conversion(p,sensor_size=48.,sensor_res=9600,material_mode="bk7")
  path=OUT/(name+".json");path.write_text(json.dumps(data,indent=2))
  lens=GeoLens(filename=str(path),device="cuda:0",dtype=torch.float64)
  for cn,w,h in CASES:
   if cn!=name:continue
   try:entry={"lens":name,**run_case(lens,p,w,h)}
   except Exception as e:entry={"lens":name,"wavelength_nm":w,"h":h,"error":repr(e),"traceback":traceback.format_exc()[-4500:]}
   out["wavelength_cases"].append(entry)
   print("RAY_AUDIT",json.dumps(entry),flush=True)
   TARGET.write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":main()
