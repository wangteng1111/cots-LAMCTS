"""Public QfilmRW API. Diagnostic MTFs never silently become task labels."""
from dataclasses import asdict
import time
import numpy as np
import torch
from .contracts import VERSION,Task,Policy,WAVES,WEIGHTS
from .solver import solve_condition,polychromatic_otf,assemble_score


def evaluate(p,fields=(0.,.5,1.),wavelengths_nm=WAVES,spectral_weights=WEIGHTS,
             task=Task(),policy=Policy(),backend='c',gpu_index=0,mode='verify',checkpoint=None,progress=None):
    if backend not in ('c','g'):raise ValueError("backend must be c or g")
    if backend=='g' and not torch.cuda.is_available():raise RuntimeError('CUDA explicitly requested but unavailable')
    if len(wavelengths_nm)!=len(spectral_weights) or not wavelengths_nm:raise ValueError('invalid spectrum')
    device='cpu' if backend=='c' else f'cuda:{gpu_index}';t0=time.monotonic();conditions=[];field_outputs=[]
    for h in fields:
        mono=[]
        for wave in wavelengths_nm:
            d=solve_condition(p,float(wave),float(h),task,policy,device,progress=progress,mode=mode)
            conditions.append(d)
            if checkpoint:checkpoint(d)
            if 'objective_otf' in d:
                mono.append({'otf':d['objective_otf'],'center_film_mm':d['selected']['center_film_mm']})
        field={'field_h':h,'diagnostic_only':True,'polychromatic_otf':None}
        if len(mono)==len(wavelengths_nm):
            ref=min(range(len(wavelengths_nm)),key=lambda i:abs(wavelengths_nm[i]-550.))
            field['polychromatic_otf']=polychromatic_otf(mono,task.frequencies,spectral_weights,mono[ref]['center_film_mm'])
            field['mtf']=abs(field['polychromatic_otf'])
            field['reference_center_mm']=mono[ref]['center_film_mm']
        field_outputs.append(field)
    return {'schema':'QfilmRWResult-v1','version':VERSION,'variant':f'QfilmRW-{backend}',
        'numerical_execution_device':device,'task':asdict(task),'policy':asdict(policy),'conditions':conditions,
        'fields':field_outputs,'quality':assemble_score(conditions),'J_task':None,'training_eligible':False,
        'model_scope_validated':False,'empirically_calibrated':False,'elapsed_seconds':time.monotonic()-t0}
