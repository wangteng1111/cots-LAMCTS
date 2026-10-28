#!/usr/bin/env python3
"""Root-cause of native Huygens red residual:
  independent chief centering; finite-source spherical vs plane-wave input;
  deterministic matched entrance rays vs native random entrance-pupil sampler.
Only visualization/source sampling shims, no DeepLens optical equation changes.
"""
from __future__ import annotations
import sys,os,json,math,types,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from validation.qfilm_deeplens_same_prescription import thin_lens,internal_stop_positive_lens,conversion,mtf_from_psf,BK7
from validation.qfilm03_huygens_safechunk import patch_huygens
from deeplens import GeoLens
from deeplens.light import Ray
from evaluator.qfilm02 import pupil_grid,_surfaces_at,_start_for_stop,trace_pupil
from evaluator.qfilm03_physical_huygens import Propagation,eval_fixed_film,mtf_from_psf_exact
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_source_wavefield"
OUT.mkdir(parents=True,exist_ok=True);FILE=OUT/"random_chief_and_source_curvature.json"
F=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)
R=math.hypot(36.,24.)/2
SCENES=[("singlet",670.,0.),("singlet",670.,.1),("internal_stop_asphere",670.,.1)]
def mtf(psf,h):
 a,b=mtf_from_psf(psf.cpu().numpy(),.005,F)
 return np.column_stack((a,b)) if h==0 else np.column_stack((b,a))
def main():
 old,new,meta=patch_huygens(batch=4096)
 GeoLens.psf_huygens=new
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 report={"schema":"Qfilm03-huygens-source-and-chief-isolation-v1","cases":[]}
 for name,p in (("singlet",p),("internal_stop_asphere",internal_stop_positive_lens())):
  data,_=conversion(p,sensor_size=48.,sensor_res=9600,material_mode="bk7")
  file=OUT/(name+".json");file.write_text(json.dumps(data))
  lens=GeoLens(filename=str(file),device="cuda:0",dtype=torch.float64)
  for nm,w,h in SCENES:
   if nm!=name:continue
   rec={"case":name,"lambda_nm":w,"field_h":h,"data":{}}
   report["cases"].append(rec)
   stop=p["stop_z_mm"];radius=p["design_spec"]["stop_diameter_mm"]/2
   zstart=min(stop,0.)-20;offset=max(0.,-min(0.,stop))
   alpha=math.atan(h*R/p["design_spec"]["efl_target_mm"])
   ang=torch.tensor([[alpha]],device="cuda:0",dtype=torch.float64)
   surfs=_surfaces_at(p["surfaces"],w)
   with torch.inference_mode():
    chi=trace_pupil(surfs,ang,torch.zeros((1,2),device="cuda:0",dtype=torch.float64),
          stop,radius,p["image_z_mm"])
   chiefx=float(chi[4].item())
   point=[0.,math.tan(alpha)*float(lens.foclen)/24.,-1e6]
   seed=int(20261028+w*17+100*h)
   orig_center=lens.psf_center
   orig_sample=lens.sample_from_points
   def exact_chief(self,points_obj,method="chief_ray",ray=None):
    return torch.tensor([[0.,chiefx]],device="cuda:0",dtype=torch.float64)
   try:
    for key,fix in (("native_random_chief",False),("native_exact_chief",True)):
     lens.psf_center=types.MethodType(exact_chief,lens) if fix else orig_center
     torch.manual_seed(seed)
     with torch.inference_mode():psf=lens.psf_huygens(points=point,ks=128,wvln=w*.001,spp=1000000,recenter=True)
     rec["data"][key]=mtf(psf,h).tolist()
    lens.psf_center=types.MethodType(exact_chief,lens)
    n=256
    uv,inside=pupil_grid(n,"cuda:0")
    with torch.inference_mode():x0,y0,valid0=_start_for_stop(surfs,ang,uv,stop,radius,zstart)
    x0=x0.flatten();y0=y0.flatten()
    o=torch.stack((y0,-x0,torch.full_like(x0,zstart+offset)),dim=-1)
    v=(inside.flatten()&valid0.flatten())
    for type_name in ("deterministic_plane_wave","deterministic_finite_point"):
     def sample(self,points,num_rays=0,wvln=None,**kwargs):
      if type_name=="deterministic_plane_wave":
       direc=torch.zeros_like(o);direc[:,1]=-math.sin(alpha);direc[:,2]=math.cos(alpha)
       init_opl=(math.sin(alpha)*x0+math.cos(alpha)*zstart).reshape(-1,1)
      else:
       obj=torch.tensor([0.,1.e6*math.tan(alpha),-1.e6],device="cuda:0",dtype=torch.float64)
       diff=o-obj[None,:]
       distance=torch.linalg.norm(diff,dim=-1)
       direc=diff/distance[:,None]
       init_opl=distance[:,None]
      ray=Ray(o.clone(),direc.clone(),float(wvln),is_coherent=True,device="cuda:0")
      ray.is_valid=v.double().clone();ray.opl=init_opl.clone()
      return ray
     lens.sample_from_points=types.MethodType(sample,lens)
     with torch.inference_mode():psf=lens.psf_huygens(points=point,ks=128,wvln=w*.001,spp=n*n,recenter=True)
     rec["data"][type_name]=mtf(psf,h).tolist()
    arr={k:np.asarray(v) for k,v in rec["data"].items()}
    def delta(a,b):return {"max":float(np.max(abs(arr[a]-arr[b]))),
       "mean":float(np.mean(abs(arr[a]-arr[b])))}
    rec["comparisons"]={
      "chief_only":delta("native_random_chief","native_exact_chief"),
      "finite_source_effect":delta("deterministic_plane_wave","deterministic_finite_point"),
      "random_vs_deterministic_plane":delta("native_exact_chief","deterministic_plane_wave"),
      "random_vs_deterministic_finite":delta("native_exact_chief","deterministic_finite_point")
      }
    print("SOURCE_AUDIT",name,w,h,json.dumps(rec["comparisons"]),flush=True)
   except Exception as e:
    rec["error"]=repr(e);rec["traceback"]=traceback.format_exc()[-4500:]
    print("SOURCE_FAIL",name,w,h,repr(e),flush=True)
   finally:
    lens.sample_from_points=orig_sample
    lens.psf_center=orig_center
   FILE.write_text(json.dumps(report,indent=2)+"\n")
 print("SOURCE_COMPLETE",json.dumps([{k:v for k,v in x.items() if k!="data"} for x in report["cases"]]),flush=True)
if __name__=="__main__":main()
