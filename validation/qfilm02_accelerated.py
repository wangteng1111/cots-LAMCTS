#!/usr/bin/env python3
"""Compare experimental fused Triton path to baseline Torch tracer across physical cases."""
from __future__ import annotations
import os,sys,time,json,traceback
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np,torch
from evaluator.qfilm02 import Config,evaluate as reference,pupil_grid,trace_pupil,_surfaces_at
from evaluator.qfilm02_fast import evaluate as fast_evaluate
from evaluator.qfilm02_triton import trace_wavelengths
from validation.qfilm02_smoke import thin_lens

def rich_lens():
    return {'surfaces':[
     {'radius':65.,'thickness':4.5,'n_after':1.5168,'v_after':64.17,'clear_aperture':34.},
     {'radius':-95.,'thickness':3.,'n_after':1.,'v_after':0.,'clear_aperture':34.},
     {'radius':-56.,'thickness':3.7,'n_after':1.620,'v_after':36.,'clear_aperture':30.,'conic':-.3,'asphere':(1.5e-7,-3.e-10)},
     {'radius':115.,'thickness':4.,'n_after':1.,'v_after':0.,'clear_aperture':30.}
    ],'stop_z_mm':5.5,'image_z_mm':68.,'design_spec':{'stop_diameter_mm':9.,'efl_target_mm':53.}}

def one_case(name,lens,pupil_n=32,fields=5,waves=(510.,550.,630.)):
    weights=tuple([1./len(waves)]*len(waves))
    cf=Config(fields=fields,pupil_n=pupil_n,fft_n=4*pupil_n,
        wavelengths_nm=waves,spectral_weights=weights,max_fields_per_chunk=min(fields,10))
    results={'name':name,'fields':fields,'pupil':pupil_n,'wavelengths':len(waves)}
    st=time.perf_counter()
    try:
        with torch.no_grad():ref=reference(lens,cf,device='cuda:0')
        torch.cuda.synchronize();results['reference_secs']=time.perf_counter()-st
        st=time.perf_counter()
        with torch.no_grad():fast=fast_evaluate(lens,cf,device='cuda:0')
        torch.cuda.synchronize();results['fast_secs']=time.perf_counter()-st
        for key in ('Q_image','Q_wavefront'):
            a=ref[key];b=fast[key]
            results[key+'_max_abs']=float(np.max(np.abs(a-b)))
            results[key+'_mean_abs']=float(np.mean(np.abs(a-b)))
            results[key+'_valid']=bool(np.all(np.isfinite(b)))
        results['J_ref']=ref['quality']['J_task'];results['J_fast']=fast['quality']['J_task']
        results['J_abs']=abs(float(results['J_ref'])-float(results['J_fast']))
        results['speedup']=results['reference_secs']/results['fast_secs']
        # Optical image MTF must numerically agree; wavefront Zernike coefficients may amplify residual differences.
        results['passed']=results['Q_image_max_abs']<1e-5 and results['J_abs']<1e-6
    except Exception:
        results['passed']=False;results['traceback']=traceback.format_exc()
    print('CASE '+json.dumps(results),flush=True)
    return results

def main():
    print('GPU',torch.cuda.get_device_name(0),'TRITON',__import__('triton').__version__,flush=True)
    results=[one_case('thin_pupil32',thin_lens(),32,5),one_case('multi_front_pupil32',rich_lens(),32,5)]
    if all(x['passed'] for x in results):
        results.extend([one_case('thin_full_pupil64',thin_lens(),64,100,(430.,470.,510.,550.,590.,630.,670.)),
                        one_case('multi_front_pupil64',rich_lens(),64,15)])
    output={'schema':'Qfilm0.2-triton-comparison-v1','passed':all(x['passed'] for x in results),'cases':results}
    path=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'))/'qfilm02_triton_compare.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(output,indent=2)+'\n')
    print('SUMMARY '+json.dumps({'passed':output['passed'],'cases':len(results)}),flush=True)
    if not output['passed']:raise SystemExit(2)

if __name__=='__main__':main()