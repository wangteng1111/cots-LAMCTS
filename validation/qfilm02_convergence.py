#!/usr/bin/env python3
"""Qfilm0.2 full lens pupil-grid convergence test; no physics certification implied."""
from __future__ import annotations
import json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,torch
from evaluator.qfilm02 import Config,evaluate,fft_pupil_metrics,pupil_grid
from validation.qfilm02_smoke import thin_lens
from validation.qfilm02_accelerated import rich_lens

def run_case(name,lens,fields=21,waves=(510.,550.,630.)):
    rows=[]
    for n in (32,64,128):
        conf=Config(fields=fields,pupil_n=n,fft_n=4*n,
                    max_fields_per_chunk=min(fields,5),wavelengths_nm=waves,
                    spectral_weights=tuple(1/len(waves) for _ in waves),
                    target_efl_mm=lens['design_spec']['efl_target_mm'])
        a=time.perf_counter()
        try:
            with torch.inference_mode():r=evaluate(lens,conf,device='cuda:0')
            torch.cuda.synchronize()
            rows.append({'n':n,'seconds':time.perf_counter()-a,'j':r['quality']['J_task'],'q':r['Q_image'],
                         'wave':r['Q_wavefront']})
            print('GRID',name,n,'seconds',rows[-1]['seconds'],'J',rows[-1]['j'],flush=True)
        except Exception as e:
            rows.append({'n':n,'error':repr(e),'traceback':traceback.format_exc()})
            print('GRID_FAIL',name,n,repr(e),flush=True)
    out={'name':name,'rows':[{'n':r['n'],'seconds':r.get('seconds'),'J':r.get('j'),'error':r.get('error')} for r in rows]}
    good=[r for r in rows if 'q' in r]
    compare=[]
    for a,b in zip(good,good[1:]):
        q1,q2=a['q'],b['q']
        dt=np.abs(q1[:,10:30]-q2[:,10:30])
        compare.append({'from_n':a['n'],'to_n':b['n'],'mtf_max_abs':float(np.max(dt)),
                        'mtf_mean_abs':float(np.mean(dt)),
                        'mtf_p95_abs':float(np.quantile(dt,.95)),
                        'J_abs':abs(a['j']-b['j']) if a['j'] is not None and b['j'] is not None else None,
                        'center_max_abs':float(np.max(dt[0])),
                        'edge_max_abs':float(np.max(dt[-1]))})
    out['comparisons']=compare
    if len(good)==3:
        out['passed_64_128']=compare[-1]['mtf_max_abs']<.01 and (compare[-1]['J_abs'] or 0)<.02
    print('CONVERGENCE',json.dumps(out),flush=True)
    return out

def main():
    out={'schema':'Qfilm0.2-convergence-v1','results':[]}
    for name,lens,fields in [('simple_biconvex',thin_lens(),21),('four_surface_internal_stop',rich_lens(),9)]:
        out['results'].append(run_case(name,lens,fields))
        dest=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'))/'qfilm02_convergence.json'
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2)+'\n')
    return 0
if __name__=='__main__':raise SystemExit(main())
