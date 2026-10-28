#!/usr/bin/env python3
"""Definitive DeepLens virtual exit pupil plane vs physical post-glass plane ablation.
Hold identical Qfilm refraction/OPL and Huygens spherical kernel; move source plane.
"""
from __future__ import annotations
import sys,os,json,math,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from evaluator.qfilm03_physical_huygens import Propagation,eval_fixed_film,mtf_from_psf_exact
from validation.qfilm_deeplens_same_prescription import thin_lens,internal_stop_positive_lens,conversion,mtf_from_psf,BK7
from deeplens import GeoLens
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_virtual_pupil_diagnostic"
OUT.mkdir(parents=True,exist_ok=True);FILE=OUT/"virtual_vs_postglass_huygens.json"
F=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)
SCENES=[("singlet",670.,0.),("singlet",670.,.1),("singlet",550.,0.),
       ("internal_stop_asphere",670.,0.),("internal_stop_asphere",670.,.1),
       ("internal_stop_asphere",630.,0.)]
def main():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 lenses={"singlet":p,"internal_stop_asphere":internal_stop_positive_lens()}
 report={"schema":"Qfilm03-real-vs-virtual-exit-pupil-physical-PSF-v1",
         "source_SHA":"911bb9d334fce7e20a4407c4506ac683b52ff246",
         "cases":[],"training_eligible":False}
 for name,p in lenses.items():
  d,_=conversion(p,sensor_size=48.,sensor_res=9600,material_mode="bk7")
  file=OUT/(name+".json");file.write_text(json.dumps(d))
  lens=GeoLens(filename=str(file),device="cuda:0",dtype=torch.float64)
  pupil_z=float(lens.get_exit_pupil()[0])
  offset=max(0.,-min(0.,p["stop_z_mm"]))
  virtual_z=pupil_z-offset
  for nm,w,h in SCENES:
   if name!=nm:continue
   angle=math.atan(h*math.hypot(36,24)/2/p["design_spec"]["efl_target_mm"])
   point=[0.,math.tan(angle)*float(lens.foclen)/(float(lens.sensor_size[1])/2),-1e6]
   torch.manual_seed(20261028+int(w)*17+int(h*10)*100)
   with torch.inference_mode():
    psf=lens.psf_huygens(points=point,ks=128,wvln=w*.001,spp=1000000,recenter=True)
   tx,ty=mtf_from_psf(psf.cpu().numpy(),.005,F)
   native=np.column_stack((tx,ty)) if h==0 else np.column_stack((ty,tx))
   rec={"name":name,"lambda_nm":w,"h":h,
        "deep_virtual_exit_z_mm_Qfilm_coordinates":virtual_z,
        "deep_virtual_exit_z_mm_absolute":pupil_z,
        "last_refracting_vertex_z_mm_Qfilm":sum(si["thickness"] for si in p["surfaces"][:-1]),
        "native_deeplens_mtf":native.tolist(),"comparisons":[]}
   report["cases"].append(rec)
   for n in (128,256):
    for amp in ("uniform_stop","sqrt_exit_jac"):
     for mode in ("after_last_glass","DeepLens_virtual_backextrapolation"):
      kw={} if mode=="after_last_glass" else {"diagnostic_virtual_exit_z_mm":virtual_z}
      t=time.perf_counter()
      try:
       with torch.inference_mode():
        result=eval_fixed_film(p,w,h,Propagation(pupil_n=n,pixel_n=128,
               pixel_pitch_um=5.,pixel_block=384,weight_mode=amp,**kw))
       mtf=mtf_from_psf_exact(result["psf"],.005,F)
       delta=np.abs(mtf-native)
       rr={"n":n,"amp":amp,"mode":mode,"max_abs_mtf":float(delta.max()),
           "mean_abs_mtf":float(delta.mean()),"seconds":time.perf_counter()-t,
           "PSF_source_z_mm":result["exit_z_mm"],"PSF_source_plane":result["source_plane_type"],
           "MTF":mtf.tolist()}
      except Exception as e:
       rr={"n":n,"amp":amp,"mode":mode,"error":repr(e),"traceback":traceback.format_exc()[-1500:]}
      rec["comparisons"].append(rr)
      print("VIRTUAL_PUPIL",name,w,h,n,amp,mode,"max",rr.get("max_abs_mtf"),"error",rr.get("error"),flush=True)
      FILE.write_text(json.dumps(report,indent=2)+"\n")
 print("PLANE_SUMMARY",json.dumps([{"name":x["name"],"w":x["lambda_nm"],"h":x["h"],
  "modes":[{"n":i["n"],"amp":i["amp"],"mode":i["mode"],"max":i.get("max_abs_mtf")} for i in x["comparisons"]]}
    for x in report["cases"]]),flush=True)
if __name__=="__main__":main()
