#!/usr/bin/env python3
"""Native DeepLens Huygens vs Qfilm Huygens on EXACT same deterministic rays/OPL.
Replaces source sampling ONLY in test, not upstream equations or ray tracing.
"""
from __future__ import annotations
import os,sys,json,math,types,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from deeplens.light import Ray
from validation.qfilm_deeplens_same_prescription import (thin_lens,internal_stop_positive_lens,conversion,mtf_from_psf,BK7)
from validation.qfilm03_huygens_safechunk import patch_huygens
from evaluator.qfilm02 import pupil_grid,_surfaces_at,_start_for_stop,trace_pupil
from evaluator.qfilm03_physical_huygens import Propagation,eval_fixed_film,mtf_from_psf_exact
from deeplens import GeoLens
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_identical_huygens"
OUT.mkdir(parents=True,exist_ok=True);FILE=OUT/"identical_rays_two_propagators.json"
F=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)
R=math.hypot(36.,24.)/2
CASES=[("singlet",670.,0.1),("singlet",670.,0.),("internal_stop_asphere",670.,.1),("internal_stop_asphere",630.,0.)]
def run_same(lens,p,w,h,n):
 stop=float(p["stop_z_mm"]);film=float(p["image_z_mm"]);rs=p["design_spec"]["stop_diameter_mm"]/2.
 offset=max(0.,-min(0.,stop));zstart=min(stop,0.)-20.
 angle=math.atan(h*R/p["design_spec"]["efl_target_mm"])
 ang=torch.tensor([[angle]],device="cuda:0",dtype=torch.float64)
 uv,inside=pupil_grid(n,"cuda:0")
 surf=_surfaces_at(p["surfaces"],w)
 with torch.inference_mode():
  x0,y0,v0=_start_for_stop(surf,ang,uv,stop,rs,zstart)
  chi=trace_pupil(surf,ang,torch.zeros((1,2),device="cuda:0",dtype=torch.float64),stop,rs,film)
  chief_x=float(chi[4].item())
  x0=x0.flatten();y0=y0.flatten()
  # +x Qfilm ray field rotated/reflected to -y native DeepLens source field
  o=torch.stack((y0,-x0,torch.full_like(x0,zstart+offset)),dim=-1)
  dirs=torch.zeros_like(o);dirs[:,1]=-math.sin(angle);dirs[:,2]=math.cos(angle)
  incoming_opl=(math.sin(angle)*x0+math.cos(angle)*zstart).reshape(-1,1)
  valid=(inside.flatten()&v0.flatten())
  def sample_fixed(self,points,num_rays=1000000,wvln=None,**kw):
   ray=Ray(o.clone(),dirs.clone(),float(wvln),is_coherent=True,device="cuda:0")
   ray.opl=incoming_opl.clone()
   ray.is_valid=valid.double().clone()
   return ray
  def center_fixed(self,points_obj,method="chief_ray",ray=None):
   return torch.tensor([[0.,chief_x]],device="cuda:0",dtype=torch.float64)
  real_sample=lens.sample_from_points;real_center=lens.psf_center
  lens.sample_from_points=types.MethodType(sample_fixed,lens)
  lens.psf_center=types.MethodType(center_fixed,lens)
  z_virtual=float(lens.get_exit_pupil()[0])-offset
  try:
   with torch.inference_mode():
    dpsf=lens.psf_huygens(points=[0.,math.tan(angle)*float(lens.foclen)/24.,-1.e6],
                         ks=128,wvln=w*.001,spp=n*n,recenter=True)
  finally:
   lens.sample_from_points=real_sample
   lens.psf_center=real_center
  tx,ty=mtf_from_psf(dpsf.cpu().numpy(),.005,F)
  mdeep=np.column_stack((tx,ty)) if h==0 else np.column_stack((ty,tx))
  with torch.inference_mode():
   result=eval_fixed_film(p,w,h,Propagation(pupil_n=n,pixel_n=128,pixel_pitch_um=5.,
          pixel_block=384,weight_mode="uniform_stop",diagnostic_virtual_exit_z_mm=z_virtual))
  mq=mtf_from_psf_exact(result["psf"],.005,F)
  err=np.abs(mdeep-mq)
  return {"pupil_n":n,"num_valid_rays":int(valid.sum()),"virtual_z_Qfilm_mm":z_virtual,
          "max_abs_MTF_between_identical_ray_integrals":float(err.max()),
          "mean_abs_MTF_between_identical_ray_integrals":float(err.mean()),
          "DeepLens_native_deterministic_MTF":mdeep.tolist(),
          "Qfilm_deterministic_MTF":mq.tolist()}
def main():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 systems={"singlet":p,"internal_stop_asphere":internal_stop_positive_lens()}
 old,new,patch=patch_huygens(batch=4096)
 GeoLens.psf_huygens=new
 rec={"schema":"Qfilm03-DeepLens-identical-Huygens-ray-source-v1",
      "same_incident_rays_and_phase":True,
      "physics_kernel_unmodified":True,
      "batch_only_patch":patch,"records":[]}
 for name,p in systems.items():
  data,_=conversion(p,sensor_size=48.,sensor_res=9600,material_mode="bk7")
  f=OUT/(name+".json");f.write_text(json.dumps(data))
  lens=GeoLens(filename=str(f),device="cuda:0",dtype=torch.float64)
  for cn,w,h in CASES:
   if cn!=name:continue
   for n in (128,256):
    try:v={"name":name,"lambda_nm":w,"h":h,**run_same(lens,p,w,h,n)}
    except Exception as e:v={"name":name,"lambda_nm":w,"h":h,"n":n,"error":repr(e),"traceback":traceback.format_exc()[-3000:]}
    rec["records"].append(v)
    print("IDENTICAL_KERNEL",name,w,h,n,"diff",v.get("max_abs_MTF_between_identical_ray_integrals"),"error",v.get("error"),flush=True)
    FILE.write_text(json.dumps(rec,indent=2)+"\n")
 print("IDENTICAL_KERNEL_SUMMARY",json.dumps([{k:v for k,v in r.items() if k in
    ("name","lambda_nm","h","pupil_n","max_abs_MTF_between_identical_ray_integrals","error")} for r in rec["records"]]),flush=True)
if __name__=="__main__":main()
