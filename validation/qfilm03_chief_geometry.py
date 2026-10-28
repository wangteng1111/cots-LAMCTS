#!/usr/bin/env python3
"""DeepLens vs Qfilm on absolute fixed-film chief-ray intercepts and chromatic shifts.

Native DeepLens chief is selected as sampled ray closest to physical stop center;
reports its aperture-centering residual and 262K-vs-1M sampling stability.
Do not reinterpret sampled chief as metrologically exact without convergence.
"""
from __future__ import annotations
import os,sys,json,math,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from evaluator.qfilm02 import Config,evaluate as qev
from validation.qfilm_deeplens_same_prescription import (thin_lens,internal_stop_positive_lens,
                                                            BK7,conversion)
from deeplens import GeoLens
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_geometry"
OUT.mkdir(parents=True,exist_ok=True)
FILE=OUT/"chief_ray_distortion_chromatic.json"
FIELDS=(0,1,2,3,4,5,6,7,8,9,10)
WAVES=(430.,470.,510.,550.,590.,630.,670.)
def one(case,p):
 d,_=conversion(p,sensor_size=12.8,sensor_res=1024,material_mode="bk7")
 lp=OUT/(case+".json");lp.write_text(json.dumps(d))
 lens=GeoLens(filename=str(lp),device="cuda:0",dtype=torch.float64)
 rec={"case":case,"lens_efl_mm":float(lens.foclen),"samples":[]}
 qdata={}
 for w in WAVES:
  cfg=Config(fields=11,pupil_n=64,fft_n=256,max_fields_per_chunk=11,
          wavelengths_nm=(w,),spectral_weights=(1.),
          frame_width_mm=8.,frame_height_mm=6.,
          target_efl_mm=float(p["design_spec"]["efl_target_mm"]))
  with torch.inference_mode():qdata[w]=qev(p,cfg,device="cuda:0")["Q_image"]
 for w in WAVES:
  for fi in FIELDS:
   h=fi/10.0
   alpha=math.atan(h*5./p["design_spec"]["efl_target_mm"])
   vals=[]
   for spp in (262144,1000000):
    torch.manual_seed(20261028+fi*131+w*17)
    t=time.perf_counter()
    chief=lens.calc_chief_ray(fov=math.degrees(alpha),plane="meridional",
                  num_rays=spp,wvln=w*.001)
    torch.cuda.synchronize()
    ry=float(chief.o[...,1].reshape(-1)[0])
    stopdist=float(chief.stop_dist.reshape(-1)[0])
    valid=float(chief.is_valid.reshape(-1)[0])
    vals.append({"spp":spp,"physical_y_mm":ry,"positive_film_height_mm":-ry,
        "stop_dist_normalized":stopdist,"valid":valid,"seconds":time.perf_counter()-t})
   q=float(qdata[w][fi,31])
   e={"wavelength_nm":w,"field_h":h,"angle_rad":alpha,
      "qfilm_chief_x_mm":q,
      "deeplens_262k":vals[0],"deeplens_1m":vals[1],
      "chief_abs_height_error_mm":abs(abs(q)-abs(vals[1]["positive_film_height_mm"])),
      "deeplens_sampling_change_mm":abs(vals[0]["physical_y_mm"]-vals[1]["physical_y_mm"])}
   rec["samples"].append(e)
   print("CHIEF",case,w,h,"q",round(q,6),"deeplens",round(vals[1]["positive_film_height_mm"],6),
        "err_mm",round(e["chief_abs_height_error_mm"],6),
        "stopdist",round(vals[1]["stop_dist_normalized"],6),flush=True)
 rec["summary"]={
   "max_film_height_abs_error_mm":max(x["chief_abs_height_error_mm"] for x in rec["samples"]),
   "p95_film_height_abs_error_mm":float(np.quantile([x["chief_abs_height_error_mm"] for x in rec["samples"]],.95)),
   "max_deeplens_stochastic_shift_mm":max(x["deeplens_sampling_change_mm"] for x in rec["samples"]),
   "max_stop_miscentering_normalized":max(x["deeplens_1m"]["stop_dist_normalized"] for x in rec["samples"]),
   "invalid_chief_1m":sum(x["deeplens_1m"]["valid"]<.5 for x in rec["samples"])
 }
 return rec
def main():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 rec={"schema":"qfilm03-chief-position-7wave-11field-v1","status":"uncertified","results":{}}
 for name,p in (("singlet",p),("internal_stop_asphere",internal_stop_positive_lens())):
  try:r=one(name,p)
  except Exception as e:r={"error":repr(e),"traceback":traceback.format_exc()[-3500:]}
  rec["results"][name]=r;FILE.write_text(json.dumps(rec,indent=2)+"\n")
  print("CASE_SUMMARY",name,json.dumps(r.get("summary",r.get("error"))),flush=True)
if __name__=="__main__":main()
