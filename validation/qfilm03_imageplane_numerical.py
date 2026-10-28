#!/usr/bin/env python3
"""Qfilm0.3 image-plane numeric error audit: ALL 100 fields, 7 wavelengths, adaptive grids.

No reference is assumed exact; report convergence deltas separately from physical
model agreement. Includes polynomial MTF, valid-field masks, J, color, timing and
GPU allocated peak. Does not certify training data.
"""
from __future__ import annotations
import sys,os,json,time,copy,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from dataclasses import replace
from evaluator.qfilm02 import Config,WAVES,WEIGHTS
from evaluator.qfilm03 import evaluate,ValidationConfig
from validation.qfilm_deeplens_same_prescription import thin_lens,internal_stop_positive_lens,BK7

OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_accuracy"
OUT.mkdir(parents=True,exist_ok=True)
TARGET=OUT/"numerical_100x7.json"
def lenses():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 q=copy.deepcopy(p);q["design_spec"]["stop_diameter_mm"]*=1.25
 return [("singlet",p),("singlet_large_stop",q),("internal_stop_asphere",internal_stop_positive_lens())]
def summary(r):
 q=np.asarray(r["Q_image"],dtype=float)
 return {"valid_fields":int(np.count_nonzero(q[:,30]>0.5)),
         "n_fields":int(q.shape[0]),"J":r["quality"]["J_task"],"J_valid":r["quality"]["valid"],
         "efl_mm":r["efl_mm"],"fno":r["fno"],
         "MTF_center_t":q[0,10:20].tolist(),"MTF_edge_t":q[-1,10:20].tolist(),
         "MTF_center_s":q[0,20:30].tolist(),"MTF_edge_s":q[-1,20:30].tolist(),
         "illuminance_edge_proxy":float(q[-1,1]),"area_fill_edge":float(q[-1,2])}
def compare(a,b):
 a,b=np.asarray(a["Q_image"],float),np.asarray(b["Q_image"],float)
 d=np.abs(a[:,10:30]-b[:,10:30])
 den=(a[:,10:30]+b[:,10:30])/2
 flags=(a[:,30]>.5)&(b[:,30]>.5)
 return {"mtf_max_all":float(d.max()),"mtf_p95_all":float(np.quantile(d,.95)),
         "mtf_mean_all":float(d.mean()),"mtf_max_both_valid":float(d[flags].max()) if flags.any() else None,
         "num_both_valid":int(flags.sum()),
         "field_max_mtf":d.max(axis=1).tolist(),
         "max_field_h":float(np.argmax(d.max(axis=1))/(len(a)-1)),
         "mtf_bad_over_001":int((d>.01).sum()),"mtf_bad_over_005":int((d>.05).sum()),
         "J_abs":float(abs(float(a["quality"]["J_task"])-float(b["quality"]["J_task"]))) if a["quality"]["J_task"] is not None and b["quality"]["J_task"] is not None else None,
         "validity_mismatch_fields":int(np.count_nonzero((a[:,30]>.5)!=(b[:,30]>.5)))}
def main():
 rec={"schema":"qfilm03-numerical-all-fields-v1",
      "workload":"100 radial fields x seven wavelengths; 64/128/256/512 pupil",
      "device":"cuda:0","gpu":torch.cuda.get_device_name(0),
      "lenses":{},"source":"evaluator/qfilm03.py fast Torch backend",
      "physics_certified":False}
 for name,p in lenses():
  rr={};result={}
  for n in (64,128,256,512):
   cfg=Config(fields=100,pupil_n=n,fft_n=4*n,
             max_fields_per_chunk=min(10,max(1,int(10*(128/n)**2))),
             wavelengths_nm=WAVES,spectral_weights=WEIGHTS,
             target_efl_mm=float(p["design_spec"]["efl_target_mm"]))
   t=time.perf_counter()
   try:
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():out=evaluate(p,cfg,device="cuda:0",
                validation=ValidationConfig(mode="fast",base_backend="torch"))
    torch.cuda.synchronize()
    result[str(n)]={"seconds":time.perf_counter()-t,
          "peak_torch_GiB":torch.cuda.max_memory_allocated()/1024**3,
          "summary":summary(out)}
    rr[n]=out
    print("RESOLUTION",name,n,"seconds",result[str(n)]["seconds"],
          "valid",result[str(n)]["summary"]["valid_fields"],flush=True)
   except Exception as e:
    result[str(n)]={"error":repr(e),"traceback":traceback.format_exc()[-2400:]}
    print("GRID_FAIL",name,n,repr(e),flush=True)
   rec["lenses"][name]={"grids":result,"pair_errors":{
      f"{x}->{y}":compare(rr[x],rr[y])
      for x,y in ((64,128),(128,256),(256,512)) if x in rr and y in rr}}
   TARGET.write_text(json.dumps(rec,indent=2)+"\n")
 print("NUMERIC_SUMMARY",json.dumps({k:{n:{'seconds':v.get('seconds'),'J':v.get('summary',{}).get('J'),'error':v.get('error')} for n,v in c["grids"].items()} for k,c in rec["lenses"].items()}),flush=True)
if __name__=="__main__":main()
