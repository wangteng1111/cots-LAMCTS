#!/usr/bin/env python3
"""Qfilm0.3 100-field/7-color GPU operational benchmark vs Qfilm0.2."""
from __future__ import annotations
import json,os,time,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from evaluator.qfilm02 import evaluate as q02,Config
from evaluator.qfilm03 import evaluate as q03,ValidationConfig
from validation.qfilm_deeplens_same_prescription import thin_lens,internal_stop_positive_lens,BK7
def run(fn,p,conf,reps):
 vals=[];out=None;mem=[]
 for _ in range(reps):
  torch.cuda.reset_peak_memory_stats()
  t=time.perf_counter()
  with torch.inference_mode():out=fn(p,conf)
  torch.cuda.synchronize()
  vals.append(time.perf_counter()-t);mem.append(torch.cuda.max_memory_allocated()/1024**3)
 return out,vals,mem
def main():
 torch.cuda.init()
 summary={'schema':'Qfilm0.3-100x7-speed-v1','gpu':torch.cuda.get_device_name(0),'cases':[]}
 for name,p in [('biconvex',thin_lens()),('internal_stop',internal_stop_positive_lens())]:
  if name=='biconvex':p['surfaces'][0]['sellmeier']=BK7
  conf=Config(fields=100,pupil_n=64,fft_n=256,max_fields_per_chunk=100 if name=='biconvex' else 10)
  v=ValidationConfig(mode='fast',base_backend='hybrid')
  r0,a,m0=run(lambda p,c:q02(p,c,device='cuda:0'),p,conf,3)
  r1,b,m1=run(lambda p,c:q03(p,c,device='cuda:0',validation=v),p,conf,3)
  d=float(np.max(np.abs(r0['Q_image']-r1['Q_image'])))
  entry={'name':name,'fields':100,'wavelengths':7,'pupil_n':64,
      'qfilm02_torch_runs_sec':a,'qfilm03_hybrid_runs_sec':b,
      'qfilm02_torch_median_sec':statistics.median(a),
      'qfilm03_hybrid_median_sec':statistics.median(b),
      'speedup':statistics.median(a)/statistics.median(b),
      'max_gpu_mem_qfilm02_GiB':max(m0),
      'max_gpu_mem_qfilm03_GiB':max(m1),
      'Q_max_abs_delta':d,'J_diff':abs(float(r0['quality']['J_task'])-float(r1['quality']['J_task'])),
      'training_eligible':r1['training_eligible'],'reference_status':r1['reference']['status']}
  summary['cases'].append(entry)
  print('SPEED',json.dumps(entry),flush=True)
 dest=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'))/'qfilm03_full_speed.json'
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
