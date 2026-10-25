#!/usr/bin/env python3
"""Fair warmed Qfilm2 CUDA PyTorch/Triton steady-state timing and exactness."""
from __future__ import annotations
import os,sys,time,json,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from evaluator.qfilm02 import Config,evaluate as ref
from evaluator.qfilm02_fast import evaluate as fast
from validation.qfilm02_smoke import thin_lens
def measure(fn,p,conf,rounds=4):
 out=[];r=None
 for _ in range(rounds):
  t=time.perf_counter()
  with torch.inference_mode():r=fn(p,conf,device='cuda:0')
  torch.cuda.synchronize()
  out.append(time.perf_counter()-t)
 return out,r
def run(n):
 p=thin_lens();waves=(430.,470.,510.,550.,590.,630.,670.);weights=(.075,.125,.16,.28,.16,.125,.075)
 base=Config(fields=100,pupil_n=n,fft_n=4*n,max_fields_per_chunk=20,wavelengths_nm=waves,spectral_weights=weights)
 fused=Config(fields=100,pupil_n=n,fft_n=4*n,max_fields_per_chunk=100,wavelengths_nm=waves,spectral_weights=weights)
 # Explicit warmups to amortize GPU context and JIT compilation on both paths.
 measure(ref,p,base,2);measure(fast,p,fused,2)
 a=[];b=[]
 for _ in range(4):
  ta,ra=measure(ref,p,base,1)
  tb,rb=measure(fast,p,fused,1)
  a+=ta;b+=tb
 ma=statistics.median(a);mb=statistics.median(b)
 result={'pupil_n':n,'fields':100,'wavelengths':7,
         'torch_median_s':ma,'triton_median_s':mb,'speedup':ma/mb,
         'torch_samples_s':a,'triton_samples_s':b,
         'max_Q_abs':float(np.abs(ra['Q_image']-rb['Q_image']).max()),
         'J_abs':abs(float(ra['quality']['J_task'])-float(rb['quality']['J_task']))}
 assert result['max_Q_abs']<1e-5 and result['J_abs']<1e-6
 print('STEADY',json.dumps(result),flush=True)
 return result
def main():
 results={'schema':'qfilm02-steady-benchmark-v1','devices':['cuda:0'],'cases':[run(64),run(128)]}
 path=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'))/'qfilm02_steady_benchmark.json'
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
