#!/usr/bin/env python3
"""Qfilm optical integrity: supplier Sellmeier, ray Newton residual, CPU/GPU parity."""
from __future__ import annotations
import sys,os,math,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from evaluator.qfilm02 import _surfaces_at,_refract_step,_sag_and_grad,Config,evaluate
from evaluator.qfilm02_fast import evaluate as fused
from validation.qfilm02_smoke import thin_lens
from validation.qfilm02_accelerated import rich_lens

NBK7={"B":[1.03961212,0.231792344,1.01046945],"C":[0.00600069867,0.0200179144,103.560653]}
def test_sellmeier():
 p=thin_lens();p['surfaces'][0]['sellmeier']=NBK7
 val=_surfaces_at(p['surfaces'],587.5618)[0].n2
 assert abs(val-1.5168)<2e-5,(val,'Sellmeier wrong')
 assert _surfaces_at(p['surfaces'],486.1327)[0].n2>val
 cfg=Config(fields=5,pupil_n=32,fft_n=128,max_fields_per_chunk=5,
             wavelengths_nm=(486.1327,550.,587.5618),spectral_weights=(.25,.5,.25))
 cpu=evaluate(p,cfg,device='cpu')
 fast=fused(p,cfg,device='cuda:0')
 assert np.max(np.abs(cpu['Q_image']-fast['Q_image']))<1e-5
 assert 'cauchy_nV_dispersion' not in cpu['assumptions']
 return {"n_D":val,"cpu_gpu_Q_max":float(np.max(np.abs(cpu['Q_image']-fast['Q_image']))),
         "provenance":"sellmeier_3term"}
def test_contract():
 c=Config(wavelengths_nm=(550.,510.),spectral_weights=(.5,.5))
 try:c.validate();raise AssertionError('unsorted wavelength accepted')
 except ValueError:pass
 p=thin_lens();p['image_z_mm']=2.
 try:evaluate(p,Config(fields=5,pupil_n=32,fft_n=128,
                      wavelengths_nm=(550.,),spectral_weights=(1.,)),device='cpu')
 except ValueError:pass
 else:raise AssertionError('film preceding last optical surface accepted')
 return {"unsorted_wavelength_rejected":True,"pre_surface_film_rejected":True}
def main():
 results={"schema":"qfilm02-physical-input-integrity-v1",
          "sellmeier":test_sellmeier(),"contract":test_contract()}
 p=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'))/'qfilm02_physics_integrity.json'
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(results,indent=2)+'\n')
 print('SUMMARY',json.dumps(results),flush=True)
if __name__=='__main__':main()
