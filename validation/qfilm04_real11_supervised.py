#!/usr/bin/env python3
"""Corrected-kernel 11-prescription matrix with honest resource admission.

Screening tiles test arithmetic only. Full acceptance requires independent
CPU/CUDA verification of pupil, support, pitch and joint refinement. A task
may finish normally while every full condition remains unusable.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time, traceback
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'infra')]
import numpy as np
import torch
from evaluator.qfilm02 import WAVES,FREQS
from evaluator.qfilm04_io import atomic_json
from evaluator.qfilm04_sampling import ConvergencePolicy,phase_guard,make_plan,solve_condition
from evaluator.qfilm04_wavefront import fixed_exit_plane,prepare_wavefront_bundle,integrate_fixed_film,WavefrontError
from validation.qfilm04_fine_11_cpu_gpu import import_case,DEEP_SHA
from validation.qfilm04_real_prescription_import import CAMERA_SAMPLES
from infra.qfilm_convergence_status import verify_pair

FIELDS=(0.,.5,1.)

def budget_candidates(policy):
    """Necessary six-grid sets, using the solver's conservative n² cost.

    Enumerate all selected grids that could pass the independent verifier.
    This is a lower bound: adaptive exploration can consume extra work.
    Failure of this admission test cannot hide a possible verified solution
    under this finite grid policy. Admission never implies convergence.
    """
    rows=[]
    for si in range(1,len(policy.support_candidates_mm)-1):
        for n in policy.pupil_candidates:
            if n//2 not in policy.pupil_candidates or n*2 not in policy.pupil_candidates:continue
            for previous in policy.pupil_candidates:
                if previous//2 not in policy.pupil_candidates:continue
                grids={(previous//2,si-1,policy.pitch_um),(previous,si-1,policy.pitch_um),
                       (n//2,si,policy.pitch_um),(n,si,policy.pitch_um),
                       (n,si,policy.pitch_um/2),(n*2,si+1,policy.pitch_um/2)}
                cost=sum(nn*nn*int(round(policy.support_candidates_mm[s]/(pitch*.001)))**2
                         for nn,s,pitch in grids)
                rows.append({'selected_n':n,'selected_support_index':si,'prior_selected_n':previous,
                             'minimum_pair_evaluations':cost,'grids':[list(g) for g in sorted(grids)]})
    return rows


def admit(p,wave,h,policy,exit_z,center,initial_bundle,progress):
    all_candidates=budget_candidates(policy)
    affordable=[c for c in all_candidates if c['minimum_pair_evaluations']<=policy.max_pair_evaluations]
    guards={};checks=[]
    for candidate in affordable:
        rejected=None
        for n,si,_ in candidate['grids']:
            key=(n,si)
            if key not in guards:
                b=initial_bundle if n==256 else prepare_wavefront_bundle(p,wave,h,n,exit_z,center)
                guards[key]=phase_guard(p,b,policy.support_candidates_mm[si],policy)
                progress({'stage':'admission_phase_guard','pupil_n':n,'support_index':si})
                if b is not initial_bundle:del b
            if not guards[key]['passed']:
                rejected={'pupil_n':n,'support_index':si};break
        checks.append({**candidate,'failed_phase_grid':rejected})
    accepted=[c for c in checks if c['failed_phase_grid'] is None]
    return {'admitted':bool(accepted),'budget':policy.max_pair_evaluations,
            'budget_candidate_count':len(affordable),'candidate_checks':checks,
            'phase_guards':[{'pupil_n':n,'support_index':si,**g} for (n,si),g in guards.items()],
            'minimum_unconstrained_pair_evaluations':min((c['minimum_pair_evaluations'] for c in all_candidates),default=None),
            'meaning':'necessary conditions only; rejected means no independently verifiable full result under declared policy'}


def require_supervisor():
    bridge=Path(os.environ.get('COTS_BRIDGE_ROOT','/var/lib/cots-lamcts'))
    d=json.loads((bridge/'state/daemon_identity_v06.json').read_text());proc=Path('/proc')/str(d['pid'])
    if (d['version']!='0.6.0' or (proc/'stat').read_text().rsplit(')',1)[1].split()[19]!=d['start_ticks']
        or b'cots_workstation_bridge_v06.py' not in (proc/'cmdline').read_bytes()):
        raise RuntimeError('live Bridge v0.6 required for workstation computation')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',choices=CAMERA_SAMPLES,nargs='+',default=list(CAMERA_SAMPLES))
    ap.add_argument('--mode',choices=('screen','complete'),default='screen')
    ap.add_argument('--deeplens-root',type=Path,default=Path('/var/lib/cots-lamcts/deeplens_upstream'))
    ap.add_argument('--output-dir',type=Path,default=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts/real11')))
    ap.add_argument('--cpu-only',action='store_true')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--tile-pixels',type=int,choices=(16,32),default=32)
    ap.add_argument('--fields',nargs='+',type=float,default=list(FIELDS))
    ap.add_argument('--waves',nargs='+',type=float,default=list(WAVES))
    a=ap.parse_args()
    if not set(a.fields)<=set(FIELDS) or not set(a.waves)<=set(WAVES):ap.error('unsupported matrix subset')
    if a.mode=='complete':require_supervisor()
    torch.set_num_threads(a.threads)
    if subprocess.check_output(['git','-C',str(a.deeplens_root),'rev-parse','HEAD'],text=True).strip()!=DEEP_SHA:
        raise RuntimeError('DeepLens commit mismatch')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    policy=ConvergencePolicy();policy.validate()
    source_manifest=json.loads((ROOT/'validation/fixtures/qfilm04_real11_source_sha256.json').read_text())
    if source_manifest['source_commit']!=DEEP_SHA:raise RuntimeError('source manifest commit mismatch')
    report={'schema':'Qfilm04Real11Supervised-v1','status':'running','mode':a.mode,'source_commit':DEEP_SHA,
            'names':a.name,'fields_h':a.fields,'wavelengths_nm':a.waves,'policy':asdict(policy),
            'training_eligible':False,'physics_certified':False,'empirically_calibrated':False,
            'full_matrix_requested':a.fields==list(FIELDS) and a.waves==list(WAVES),
            'torch_version':str(torch.__version__),'cuda_available':torch.cuda.is_available(),
            'code_commit':os.environ.get('COTS_GIT_COMMIT'),'conditions':[],
            'evaluator_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'evaluator').glob('qfilm04*.py')}}
    last=[0.]
    def progress(value):
        if time.monotonic()-last[0]>=30 or isinstance(value,dict):
            last[0]=time.monotonic()
            atomic_json(a.output_dir/'progress.json',{'updated_unix':time.time(),'message':value})
            print('REAL11_PROGRESS',json.dumps(value),flush=True)
    for name in a.name:
        import_error=None;source_sha=None
        try:
            p,source_sha=import_case(a.deeplens_root,name)
            if source_sha!=source_manifest['source_sha256'][name]:raise RuntimeError('prescription bytes changed')
            exit_z=fixed_exit_plane(p)
        except Exception as exc:
            import_error=repr(exc)
        for h in a.fields:
            for wave in a.waves:
                ident=f'{name}-h{h:g}-w{wave:g}'
                out=a.output_dir/ident;out.mkdir(parents=True,exist_ok=True)
                row={'id':ident,'name':name,'field_h':h,'wavelength_nm':wave,'source_sha256':source_sha,
                     'started_unix':time.time(),'full_condition_verified':False,'J_task':None}
                report['conditions'].append(row)
                progress({'condition':ident,'stage':'starting'})
                try:
                    if import_error:raise RuntimeError(import_error)
                    b=prepare_wavefront_bundle(p,wave,h,256,exit_z)
                    row['wavefront']=b.metadata
                    # Full support is unchanged. This 8um tile is ONLY arithmetic.
                    c=integrate_fixed_film(b,a.tile_pixels,.25,FREQS,16,retain_intensity=True)
                    row['arithmetic_tile']={'pupil_n':256,'pixel_n':a.tile_pixels,'pitch_um':.25,'support_mm':a.tile_pixels*.00025,
                        'full_psf_validation':False,'cpu_integral':c['raw_intensity_integral'],'cpu_device':'cpu'}
                    if torch.cuda.is_available() and not a.cpu_only:
                        gb=prepare_wavefront_bundle(p,wave,h,256,exit_z,b.center_mm,'cuda:0')
                        g=integrate_fixed_film(gb,a.tile_pixels,.25,FREQS,16,retain_intensity=True)
                        delta=float(np.max(abs(c['otf']-g['otf'])))
                        intensity=float(np.max(abs(c['raw_intensity']-g['raw_intensity']))/max(float(np.max(c['raw_intensity'])),1e-300))
                        flux=abs(c['raw_intensity_integral']-g['raw_intensity_integral'])/max(g['raw_intensity_integral'],1e-300)
                        active=b.metadata['active_nodes']==gb.metadata['active_nodes']
                        row['arithmetic_tile'].update(gpu_device='cuda:0',complex_otf_max_abs=delta,
                            raw_intensity_relative_max_abs=intensity,relative_integral_difference=flux,
                            active_nodes_equal=active,passed=bool(delta<=1e-6 and intensity<=1e-6 and flux<=1e-6 and active))
                        del gb,g
                    else:row['arithmetic_tile'].update(state='CPU_ONLY' if a.cpu_only else 'CUDA_UNAVAILABLE',passed=None)
                    del c
                    row['admission']=admit(p,wave,h,policy,exit_z,b.center_mm,b,progress)
                    del b
                    row['state']='screened_admissible' if row['admission']['admitted'] else 'POLICY_RESOURCE_INFEASIBLE'
                    if a.mode=='complete' and row['admission']['admitted']:
                        if row['arithmetic_tile'].get('passed') is not True:
                            row['state']='ARITHMETIC_GATE_NOT_PASSED'
                        else:
                            plan=make_plan(p,wave,h,FREQS,policy,progress_callback=progress)
                            atomic_json(out/'phase-plan.json',plan,immutable=True)
                            states={}
                            for backend,device in (('g','cuda:0'),('c','cpu')):
                                raw_count=[0]
                                def save_raw(image,metadata):
                                    raw_count[0]+=1
                                    path=out/f'{ident}-{backend}-raw-{raw_count[0]:03d}.npy'
                                    np.save(path,image,allow_pickle=False)
                                    checksum=hashlib.sha256()
                                    with path.open('rb') as stream:
                                        for chunk in iter(lambda:stream.read(1024*1024),b''):checksum.update(chunk)
                                    return {'file':path.name,'sha256':checksum.hexdigest(),'bytes':path.stat().st_size,
                                            'shape':list(image.shape),'dtype':str(image.dtype),**metadata}
                                result=solve_condition(p,plan,policy,device,32,
                                    lambda d:atomic_json(out/f'{ident}-{backend}-checkpoint.json',d),
                                    lambda done,total:progress([ident,backend,done,total]),save_raw)
                                plan=result['resolved_sampling_plan']
                                atomic_json(out/f'{ident}-{backend}-terminal.json',result,immutable=True)
                                states[backend]=result['state']
                            atomic_json(out/'summary-terminal.json',{'source_commit':DEEP_SHA,
                                'cases':[{'id':ident,'source_sha256':source_sha}]},immutable=True)
                            row['execution_states']=states
                            try:
                                row['independent_verification']=verify_pair(out,ident,expected=(name,wave,source_sha),field_h=h)
                                row.update(full_condition_verified=True,state='numerically_converged_cpu_gpu_verified')
                            except (OSError,ValueError,KeyError,AssertionError,StopIteration,TypeError,ZeroDivisionError) as exc:
                                row['state']='FULL_CONDITION_NOT_VERIFIED';row['verification_error']=type(exc).__name__
                except WavefrontError as exc:row.update(state=exc.code,error=str(exc))
                except Exception as exc:row.update(state='error',error=repr(exc),traceback=traceback.format_exc())
                row['finished_unix']=time.time()
                atomic_json(out/'condition-terminal.json',row,immutable=True)
                atomic_json(a.output_dir/'matrix-checkpoint.json',report)
                print('REAL11_CONDITION',ident,row['state'],flush=True)
    rows=report['conditions']
    report.update(status='completed' if not any(r['state']=='error' for r in rows) else 'completed_with_errors',
        finished_unix=time.time(),condition_count=len(rows),
        verified_full_conditions=sum(r['full_condition_verified'] for r in rows),
        arithmetic_pairs_passed=sum(r.get('arithmetic_tile',{}).get('passed') is True for r in rows),
        all_requested_conditions_verified=len(rows)==len(a.name)*len(a.fields)*len(a.waves) and all(r['full_condition_verified'] for r in rows))
    atomic_json(a.output_dir/'matrix-terminal.json',report,immutable=True)
    return 2 if report['status']=='completed_with_errors' else 0

if __name__=='__main__':raise SystemExit(main())
