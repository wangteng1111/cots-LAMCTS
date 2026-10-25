#!/usr/bin/env python3
"""Higher-resolution Qfilm pupil convergence 64 / 128 / 256, with field-specific MTF error."""
from __future__ import annotations
import os,sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from evaluator.qfilm02 import evaluate,Config
from validation.qfilm02_smoke import thin_lens
from validation.qfilm02_accelerated import rich_lens

def run(label,lens):
 rows=[]
 for n in [256,512,1024]:
  conf=Config(fields=5,pupil_n=n,fft_n=4*n,max_fields_per_chunk=1,
              wavelengths_nm=(510.,550.,630.),spectral_weights=(.25,.5,.25),
              target_efl_mm=lens['design_spec']['efl_target_mm'])
  t=time.perf_counter()
  try:
   with torch.inference_mode():r=evaluate(lens,conf,device='cuda:0')
   torch.cuda.synchronize()
   rows.append({'n':n,'seconds':time.perf_counter()-t,'q':r['Q_image'],
                'wave':r['Q_wavefront'],'J':r['quality']['J_task']})
   print(label,'GRID',n,'SECONDS',rows[-1]['seconds'],'J',rows[-1]['J'],flush=True)
  except Exception as e:
   rows.append({'n':n,'error':repr(e),'traceback':traceback.format_exc()});print(label,'FAIL',n,repr(e),flush=True)
 out={'case':label,'grids':[{'n':x['n'],'seconds':x.get('seconds'),'J':x.get('J'),'error':x.get('error')} for x in rows],'comparisons':[]}
 for a,b in zip(rows,rows[1:]):
  if 'q' not in a or 'q' not in b:continue
  err=np.abs(a['q'][:,10:30]-b['q'][:,10:30])
  out['comparisons'].append({'from':a['n'],'to':b['n'],'mtf_max_abs':float(err.max()),
           'mtf_p95_abs':float(np.quantile(err,.95)),
           'mtf_mean_abs':float(np.mean(err)),
           'J_abs':float(abs(a['J']-b['J'])),
           'field_max_abs':np.max(err,axis=1).tolist(),
           'worst_field_i':int(np.argmax(np.max(err,axis=1)))})
 print('RESULT',json.dumps(out),flush=True)
 return out

def main():
 result={'schema':'Qfilm0.2-fine-convergence-v1','cases':[run('simple_biconvex',thin_lens()),
                                                          run('internal_stop',rich_lens())]}
 out=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'))/'qfilm02_megagrid_convergence.json'
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
