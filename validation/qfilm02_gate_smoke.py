#!/usr/bin/env python3
"""Qfilm0.2 conservative numeric convergence and training-label gate tests."""
from __future__ import annotations
import os,sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from evaluator.qfilm02_convergence import compare_qfilm,evaluate_multigrid
from evaluator.qfilm02 import Config
from validation.qfilm02_smoke import thin_lens

def fake():
 q=np.zeros((100,32),dtype=float);q[:,30]=1;q[:,10:30]=.7
 return {'Q_image':q,'quality':{'valid':True,'J_task':0.7}}
a=fake();b=copy.deepcopy(a);same=compare_qfilm(a,b)
assert same['numerically_converged']
b['Q_image'][34,14]=.5
different=compare_qfilm(a,b)
assert not different['numerically_converged']
assert different['mtf_max_abs']>.19
p=thin_lens()
config=Config(fields=5,pupil_n=32,fft_n=128,max_fields_per_chunk=5,
              wavelengths_nm=(510.,550.,630.),spectral_weights=(.25,.5,.25))
r=evaluate_multigrid(p,config,device='cpu',levels=(32,64))
z=r['numerical_certification']
assert not z['physics_certified'] and not z['training_eligible'] and not r['certified']
assert z['checked_levels']==[32,64]
result={'schema':'qfilm02-label-gate-v1','identical_case':same['numerically_converged'],
        'large_error_rejected':not different['numerically_converged'],
        'real_prescription_gate':{k:v for k,v in z.items() if k!='convergence_checks'}}
print(json.dumps(result,indent=2),flush=True)
p=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'))/'qfilm02_gate_test.json'
p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
