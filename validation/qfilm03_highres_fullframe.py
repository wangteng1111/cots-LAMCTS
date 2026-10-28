#!/usr/bin/env python3
"""Qfilm0.3 precision ceiling numerical: full 36x24mm, 3 fields, 7 wavelengths, n=512,1024.

High-resolution 1024 pupil and 4096 FFT; each field chunk=1 to bound memory.
The 3-field radial mesh is an independent local precision probe (center/mid/edge),
NOT a substitute for 100-field all-spectrum convergence.
"""
from __future__ import annotations
import sys,os,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from evaluator.qfilm02 import Config,WAVES,WEIGHTS,evaluate
from validation.qfilm_deeplens_same_prescription import thin_lens,internal_stop_positive_lens,BK7
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_highres"
OUT.mkdir(parents=True,exist_ok=True);RESULT=OUT/"fullframe_1024_convergence.json"
def main():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 rec={"schema":"qfilm03-fullframe-1024-3field-7wave-v1","cases":{},"not_physics_certified":True}
 for name,p in (("singlet",p),("internal_stop_asphere",internal_stop_positive_lens())):
  c={"results":{}};rec["cases"][name]=c;arr={}
  for n in (256,512,1024):
   conf=Config(fields=3,pupil_n=n,fft_n=4*n,max_fields_per_chunk=1,
      wavelengths_nm=WAVES,spectral_weights=WEIGHTS,
      frame_width_mm=36.,frame_height_mm=24.,
      target_efl_mm=p["design_spec"]["efl_target_mm"])
   try:
    torch.cuda.reset_peak_memory_stats()
    t=time.perf_counter()
    with torch.inference_mode():r=evaluate(p,conf,device="cuda:0")
    torch.cuda.synchronize()
    secs=time.perf_counter()-t
    arr[n]=r
    c["results"][str(n)]={"seconds":secs,"peak_torch_GiB":torch.cuda.max_memory_allocated()/1024**3,
        "J":r["quality"]["J_task"],"valid":r["quality"]["valid"],
        "field_valid":(np.asarray(r["Q_image"])[:,30]>.5).tolist(),
        "MTF_t":[row[10:20].tolist() for row in r["Q_image"]],
        "MTF_s":[row[20:30].tolist() for row in r["Q_image"]]}
    print("HIGHRES",name,n,"seconds",round(secs,2),"J",r["quality"]["J_task"],flush=True)
   except Exception as e:
    c["results"][str(n)]={"error":repr(e),"traceback":traceback.format_exc()[-2200:]}
    print("HIGHRES_ERROR",name,n,repr(e),flush=True)
   c["convergence"]={}
   for a,b in ((256,512),(512,1024)):
    if a not in arr or b not in arr:continue
    d=np.abs(np.asarray(arr[a]["Q_image"])[:,10:30]-np.asarray(arr[b]["Q_image"])[:,10:30])
    c["convergence"][f"{a}->{b}"]={"mtf_max":float(d.max()),"mtf_p95":float(np.quantile(d,.95)),
       "mtf_mean":float(d.mean()),"per_field_mtf_max":d.max(axis=1).tolist(),
       "J_abs":abs(float(arr[a]["quality"]["J_task"])-float(arr[b]["quality"]["J_task"]))}
   RESULT.write_text(json.dumps(rec,indent=2)+"\n")
 print("SUMMARY",json.dumps({k:v["convergence"] for k,v in rec["cases"].items()}),flush=True)
if __name__=="__main__":main()
