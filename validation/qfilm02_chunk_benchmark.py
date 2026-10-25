#!/usr/bin/env python3
"""Qfilm2 100-field GPU batching/chunk-size performance benchmark."""
from __future__ import annotations
import sys,os,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from evaluator.qfilm02 import Config,evaluate as ref
from evaluator.qfilm02_fast import evaluate as fast
from validation.qfilm02_smoke import thin_lens
from validation.qfilm02_accelerated import rich_lens
def run(lens,name,n,fields,waves,chunks):
 weights=tuple([1.0/len(waves)]*len(waves))
 data=[]
 baseline_config=Config(fields=fields,pupil_n=n,fft_n=4*n,max_fields_per_chunk=min(20,fields),
    wavelengths_nm=waves,spectral_weights=weights,
    target_efl_mm=lens['design_spec']['efl_target_mm'])
 with torch.inference_mode():
  a=time.perf_counter();rr=ref(lens,baseline_config,device='cuda:0');torch.cuda.synchronize()
  baseline=time.perf_counter()-a
 for chunk in chunks:
  config=Config(fields=fields,pupil_n=n,fft_n=4*n,max_fields_per_chunk=chunk,
      wavelengths_nm=waves,spectral_weights=weights,
      target_efl_mm=lens['design_spec']['efl_target_mm'])
  with torch.inference_mode():
   aa=fast(lens,config,device='cuda:0');torch.cuda.synchronize()
   samples=[]
   for _ in range(2):
    t=time.perf_counter();bb=fast(lens,config,device='cuda:0');torch.cuda.synchronize()
    samples.append(time.perf_counter()-t)
  diff=float(np.max(np.abs(bb['Q_image']-rr['Q_image'])))
  rec={'pupil_n':n,'fields':fields,'waves':len(waves),'chunk':chunk,
       'baseline_sec':baseline,'fast_sec_best':min(samples),'fast_sec_mean':sum(samples)/len(samples),
       'speedup':baseline/min(samples),'max_Q_abs':diff,
       'j_abs':abs(float(bb['quality']['J_task'])-float(rr['quality']['J_task'])),
       'backend_mode':bb.get('backend_mode','triton_fused')}
  data.append(rec);print('CHUNK',name,json.dumps(rec),flush=True)
 return {'name':name,'measurements':data}
def main():
 waves=(430.,470.,510.,550.,590.,630.,670.)
 res={'schema':'qfilm02-chunk-benchmark-v1','cases':[
   run(thin_lens(),'front_stop_64',64,100,waves,(10,25,50,100)),
   run(thin_lens(),'front_stop_128',128,100,waves,(10,25,50,100)),
   run(rich_lens(),'internal_stop_64',64,15,(510.,550.,630.),(15,)),
 ]}
 path=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'))/'qfilm02_chunk_benchmark.json'
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(res,indent=2)+'\n')
 print('SUMMARY',json.dumps({'cases':len(res['cases'])}),flush=True)
if __name__=='__main__':main()
