#!/usr/bin/env python3
"""Numerical convergence of corrective physical Huygens solver at red anomalies."""
from __future__ import annotations
import sys,os,json,time,math,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from evaluator.qfilm03_physical_huygens import Propagation,eval_fixed_film,mtf_from_psf_exact
from validation.qfilm_deeplens_same_prescription import thin_lens,internal_stop_positive_lens,conversion,mtf_from_psf,BK7
from deeplens import GeoLens
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_huygens_resolution"
OUT.mkdir(parents=True,exist_ok=True);FILE=OUT/"red_resolution_convergence.json"
F=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)
CASES=[("singlet",670.,0.),("singlet",670.,.1),
       ("internal_stop_asphere",630.,0.),("internal_stop_asphere",670.,.1)]
def main():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 systems={"singlet":p,"internal_stop_asphere":internal_stop_positive_lens()}
 data={"schema":"Qfilm0.3-Huygens-red-pupil-convergence-v1","cases":[],"certified":False}
 for name,p in systems.items():
  dd,_=conversion(p,sensor_size=48.,sensor_res=9600,material_mode="bk7")
  path=OUT/(name+".json");path.write_text(json.dumps(dd))
  lens=GeoLens(filename=str(path),device="cuda:0",dtype=torch.float64)
  for nm,w,h in CASES:
   if name!=nm:continue
   rec={"name":name,"lam_nm":w,"h":h,"levels":[]};data["cases"].append(rec)
   alpha=math.atan(h*math.hypot(36,24)/2/p["design_spec"]["efl_target_mm"])
   point=[0.,math.tan(alpha)*float(lens.foclen)/(float(lens.sensor_size[1])/2),-1e6]
   torch.manual_seed(20261028+int(w)*17+int(h*10)*100)
   with torch.inference_mode():ref=lens.psf_huygens(points=point,ks=128,wvln=w*.001,spp=1000000,recenter=True)
   x,y=mtf_from_psf(ref.detach().cpu().numpy(),.005,F)
   target=np.column_stack((x,y)) if h==0 else np.column_stack((y,x))
   rec["reference_mtf"]=target.tolist()
   for n in (128,192,256,384):
    if n==384 and (name!="singlet" or h==0.):continue
    entry={"pupil_n":n};rec["levels"].append(entry)
    for amp in ("uniform_stop","sqrt_exit_jac"):
     t=time.perf_counter()
     try:
      torch.cuda.reset_peak_memory_stats()
      with torch.inference_mode():
       rr=eval_fixed_film(p,w,h,Propagation(pupil_n=n,pixel_n=128,pixel_pitch_um=5.,
                                              pixel_block=384,weight_mode=amp))
      torch.cuda.synchronize()
      mtf=mtf_from_psf_exact(rr["psf"],.005,F)
      delta=np.abs(mtf-target)
      entry[amp]={"seconds":time.perf_counter()-t,"max_mtf_difference":float(delta.max()),
                  "mean_abs_mtf_difference":float(delta.mean()),
                  "MTF":mtf.tolist(),"bad_jacobian_fraction":rr["bad_jacobian_fraction"],
                  "active_rays":rr["valid_rays"],"peak_torch_mem_GiB":torch.cuda.max_memory_allocated()/1024**3}
      print("RED_CONVERGENCE",name,w,h,n,amp,"max",entry[amp]["max_mtf_difference"],
            "sec",entry[amp]["seconds"],flush=True)
     except Exception as e:
      entry[amp]={"error":repr(e),"traceback":traceback.format_exc()[-1200:]}
      print("RED_CONVERGENCE_ERROR",name,w,h,n,amp,repr(e),flush=True)
     FILE.write_text(json.dumps(data,indent=2)+"\n")
 print("RED_RESOLUTION_COMPLETE",json.dumps([{ "name":c["name"],"w":c["lam_nm"],"h":c["h"],
    "levels":[{"n":x["pupil_n"],"uniform":x.get("uniform_stop",{}).get("max_mtf_difference"),
        "jacobian":x.get("sqrt_exit_jac",{}).get("max_mtf_difference")}
        for x in c["levels"]]} for c in data["cases"]]),flush=True)
if __name__=="__main__":main()
