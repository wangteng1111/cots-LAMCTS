#!/usr/bin/env python3
"""11 pinned product-named prescriptions, 3 fields x 7 wavelengths.

The active driver runs two independent GPU lanes; the CPU solver is retired. Network
publication is handled by Bridge, never inside the solver. Per-condition
terminals and immutable summaries distinguish execution from acceptance.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from evaluator.qfilmrw import VERSION,Task,Policy,solve_condition
from evaluator.qfilmrw.contracts import WAVES,digest
from evaluator.qfilmrw.solver import IMPLEMENTATION_SIGNATURE
from evaluator.qfilm04_io import atomic_json
from validation.qfilm04_real_prescription_import import CAMERA_SAMPLES

FIXTURE_SHA='09f589d4859070a036de73fca4a7c4287edf393abcd484d009e6534f81ac7954'

def cases():
    file=ROOT/'validation/fixtures/qfilmrw_real11.json'
    if hashlib.sha256(file.read_bytes()).hexdigest()!=FIXTURE_SHA:raise RuntimeError('frozen prescription fixture changed')
    d=json.loads(file.read_text())
    if d['source_commit']!='911bb9d334fce7e20a4407c4506ac683b52ff246':raise RuntimeError('source commit mismatch')
    if tuple(c['name'] for c in d['cases'])!=CAMERA_SAMPLES:raise RuntimeError('wrong 11 prescriptions')
    old=json.loads((ROOT/'validation/fixtures/qfilm04_real11_source_sha256.json').read_text())
    if any(c['source_sha256']!=old['source_sha256'][c['name']] for c in d['cases']):raise RuntimeError('old/new source identity mismatch')
    return d['cases']

def require_handover():
    bridge=Path(os.environ.get('COTS_BRIDGE_ROOT','/var/lib/cots-lamcts'))
    identity=json.loads((bridge/'state/daemon_identity_v06.json').read_text());proc=Path('/proc')/str(identity['pid'])
    if identity['version']!='0.6.1' or (proc/'stat').read_text().rsplit(')',1)[1].split()[19]!=identity['start_ticks']:
        raise RuntimeError('live Bridge 0.6.1 required')
    if b'cots_workstation_bridge_v06.py' not in (proc/'cmdline').read_bytes():raise RuntimeError('unexpected supervisor process')
    old=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            args=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(a.startswith(str(bridge/'jobs')+'/') and '/src/' in a and a.endswith('.py') and 'qfilmrw' not in a.lower() and '000-000-020-qfilmrw-' not in a for a in args):old.append(entry.name)
        except OSError:pass
    if old:raise RuntimeError('legacy job processes remain; refusing overlapping test: '+','.join(old))

def decode_otf(value):
    return np.array([[complex(v['real'],v['imag']) for v in row] for row in value])

def check_condition(value):
    """Recompute reported projections/OTF, integral and gate decisions.

    Used on local artifacts, not on success flags alone. Model-scope validation
    is deliberately false for this first local-Fourier implementation.
    """
    from evaluator.qfilm04_wavefront import otf_from_projections
    if digest(value['plan'])!=value['plan_hash']:raise ValueError('plan hash mismatch')
    for run in value['runs']:
        px=np.asarray(run['raw_projection_x']);py=np.asarray(run['raw_projection_y']);pitch=run['pitch_um']*.001
        got=otf_from_projections(px,py,pitch,value['plan']['task']['frequencies'])
        if not np.allclose(got,decode_otf(run['otf']),atol=2e-11,rtol=0):raise ValueError('OTF/projection mismatch')
        if not np.isclose(px.sum()*pitch*pitch,run['raw_integral'],rtol=2e-11):raise ValueError('raw integral mismatch')
    policy=value['plan']['policy'];by_axis={r['axis']:r for r in value['runs']}
    pupils=[r for r in value['runs'] if r['axis']=='pupil'];pi=0
    for comparison in value['comparisons']:
        axis=comparison['axis']
        if axis=='pupil':a,b=pupils[pi:pi+2];pi+=1;tol=policy['pupil_otf_tolerance']
        elif axis=='angular_grid':a,b=pupils[-1],by_axis[axis];tol=policy['grid_otf_tolerance']
        else:
            a,b=by_axis['angular_grid'],by_axis[axis]
            tol=policy[axis+'_otf_tolerance']
        delta=float(np.max(abs(decode_otf(a['otf'])-decode_otf(b['otf']))))
        integral=abs(a['raw_integral']-b['raw_integral'])/max(abs(b['raw_integral']),1e-300)
        passed=delta<=tol and integral<=policy['integral_relative_tolerance']
        if passed!=comparison['passed'] or not np.isclose(delta,comparison['complex_otf_max_abs'],rtol=1e-10,atol=1e-14):raise ValueError('refinement comparison mismatch')
    for name in ('reference_probe','joint_reference_probe'):
        if name not in value:continue
        probe=value[name]
        exact=np.asarray(probe['exact_real'])+1j*np.asarray(probe['exact_imag'])
        approx=np.asarray(probe['approx_real'])+1j*np.asarray(probe['approx_imag'])
        delta=np.linalg.norm(exact-approx)/max(np.linalg.norm(exact),1e-300)
        if not np.isclose(delta,probe['relative_complex_field_l2'],rtol=1e-10,atol=1e-14):raise ValueError('direct reference discrepancy mismatch')
    if value.get('numerically_verified'):
        if not all(value['flags'].values()):raise ValueError('false numerical pass')
        if not {'pupil','angular_grid','support','pitch','joint'} <= {d['axis'] for d in value['comparisons'] if d['passed']}:raise ValueError('missing refinement')
        if pupils[-1]['residual_phase_neighbor_cycles']>policy['residual_phase_target_cycles'] or by_axis['joint']['residual_phase_neighbor_cycles']>policy['residual_phase_target_cycles']:raise ValueError('phase sampling gate mismatch')
        for name in ('reference_probe','joint_reference_probe'):
            if value[name]['relative_complex_field_l2']>policy['reference_relative_l2_tolerance']:raise ValueError('reference gate mismatch')
    if value.get('training_eligible') or value.get('J_task') is not None:raise ValueError('unvalidated local model cannot generate J labels')
    return True

def worker(a):
    torch.set_num_threads(a.threads)
    from evaluator.qfilmrw.field_audit import cuda_device
    cuda_device(a.device)
    selected=[c for c in cases() if c['name'] in a.names]
    out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    pupils=tuple(a.pupils) if a.pupils else ((64,128) if a.mode=='survey' else (128,256,512,1024))
    policy=Policy(pupils=pupils,angular_grid_n=a.grid_n,max_seconds=a.condition_seconds,max_live_elements=a.max_live_elements)
    task=Task();task.validate();policy.validate()
    audit=json.loads(a.field_contract.read_text()) if a.field_contract else None
    if any(h!=0 for h in a.fields) and audit is None:raise RuntimeError('GPU off-axis verification requires an audited field contract')
    if audit and (audit.get('status')!='completed' or audit.get('gpu_only_optics') is not True):raise RuntimeError('completed GPU field audit required')
    domains={c['name']:c for c in audit['cases']} if audit else {}
    expected_domains={c['name']:c for c in json.loads((ROOT/'validation/fixtures/qfilmrw_field_domains.json').read_text())['sources']}
    report={'schema':'QfilmRWGPUOnlyMatrix-v2','version':VERSION,'status':'running','actual_device':a.device,'execution_policy':'GPU_ONLY_NO_CPU_REFERENCE',
        'code_commit':os.environ.get('COTS_GIT_COMMIT'),'fixture_sha256':FIXTURE_SHA,'source_commit':'911bb9d334fce7e20a4407c4506ac683b52ff246',
        'fields_h':a.fields,'wavelengths_nm':a.waves,'task_hash':task.signature,'policy':policy.__dict__,
        'mode':a.mode,'started_unix':time.time(),'conditions':[],'J_task':None,'training_eligible':False}
    report['code_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [*sorted((ROOT/'evaluator/qfilmrw').glob('*.py')),Path(__file__).resolve(),ROOT/'evaluator/qfilm02.py',ROOT/'evaluator/qfilm04_wavefront.py']}
    for case in selected:
        for h in a.fields:
            for w in a.waves:
                ident=f"{case['name']}-h{h:g}-w{w:g}"
                folder=out/ident;folder.mkdir(parents=True,exist_ok=True)
                def progress(d):
                    atomic_json(out/'progress.json',{'updated_unix':time.time(),'condition':ident,**d})
                    print('QFILMRW',ident,d['stage'],flush=True)
                raw_index=[0]
                def save_raw(image,metadata):
                    raw_index[0]+=1;file=folder/f'psf-{raw_index[0]:02d}.npy'
                    np.save(file,image,allow_pickle=False)
                    sha=hashlib.sha256()
                    with file.open('rb') as stream:
                        for chunk in iter(lambda:stream.read(1024*1024),b''):sha.update(chunk)
                    return {'file':file.name,'sha256':sha.hexdigest(),'shape':list(image.shape),'dtype':str(image.dtype),**metadata}
                domain=domains.get(case['name'])
                calibrated=next((f for f in domain['native_fields'] if f['field_h']==h),None) if domain else None
                if domain and domain['source_sha256']!=case['source_sha256']:raise RuntimeError('audited field contract source mismatch')
                if domain and domain['metadata']!=expected_domains[case['name']]:raise RuntimeError('audited field domain metadata mismatch')
                if calibrated and calibrated['state']=='calibrated':
                    if not math.isfinite(calibrated['object_angle_rad']) or calibrated['residual_mm']>.001:raise RuntimeError('field coordinate calibration failed')
                    if abs(calibrated['target_image_height_mm']-h*domain['metadata']['source_r_sensor_mm'])>1e-8:raise RuntimeError('field target is not the declared native image height')
                if h!=0 and (calibrated is None or calibrated['state']!='calibrated'):
                    value={'schema':'QfilmRWCondition-v2','state':'FIELD_CALIBRATION_UNRESOLVED','runs':[],'J_task':None,
                      'numerically_verified':False,'model_scope_validated':False,'training_eligible':False,
                      'actual_device':a.device,'plan_hash':None,'elapsed_seconds':0.,'flags':{k:False for k in ('pupil','angular_grid','support','pitch','joint','reference_spotcheck')}}
                else:
                    theta=calibrated['object_angle_rad'] if calibrated else 0.
                    contract=None if calibrated is None else {'scope':'native_prescription_image_circle','target_image_height_mm':calibrated['target_image_height_mm'],
                        'source_r_sensor_mm':domain['metadata']['source_r_sensor_mm'],'calibration_angle_rad':theta,'source_sha256':case['source_sha256']}
                    value=solve_condition(case['prescription'],w,h,task,policy,a.device,progress,save_raw if a.save_raw else None,a.mode,object_angle_rad=theta,field_contract=contract)
                value.update(id=ident,name=case['name'],source_sha256=case['source_sha256'],code_commit=os.environ.get('COTS_GIT_COMMIT'),finished_unix=time.time())
                atomic_json(folder/'condition-terminal.json',value,immutable=True)
                if value['runs']:check_condition(json.loads((folder/'condition-terminal.json').read_text()))
                row={k:value[k] for k in ('id','name','state','actual_device','source_sha256','plan_hash','elapsed_seconds','numerically_verified','model_scope_validated')}
                row.update(completed_psf_grids=len(value['runs']),native_terminal=f'{ident}/condition-terminal.json',flags=value['flags'],
                    reference_relative_l2=value.get('reference_probe',{}).get('relative_complex_field_l2'))
                report['conditions'].append(row)
                atomic_json(out/'matrix-checkpoint.json',report)
                print('QFILMRW_TERMINAL',ident,value['state'],flush=True)
    report.update(status='completed_with_errors' if any(c['state']=='NUMERICAL_FAILURE' for c in report['conditions']) else 'completed',
        finished_unix=time.time(),condition_count=len(report['conditions']),
        completed_psf_grids=sum(c['completed_psf_grids'] for c in report['conditions']),
        numerical_conditions=sum(c['numerically_verified'] for c in report['conditions']),
        model_validated_conditions=sum(c['model_scope_validated'] for c in report['conditions']))
    atomic_json(out/'matrix-terminal.json',report,immutable=True)
    return 0 if report['status']=='completed' else 2

def collect_legacy_cpu_gpu(root):
    """Compare actual CPU/GPU executions, retaining missing/error states."""
    rows=[];cpu_root=root/'cpu';gpu_roots=[root/'cuda0',root/'cuda1']
    for case in cases():
        for h in (0.,.5,1.):
            for w in WAVES:
                ident=f"{case['name']}-h{h:g}-w{w:g}";cf=cpu_root/ident/'condition-terminal.json'
                gf=next((p/ident/'condition-terminal.json' for p in gpu_roots if (p/ident/'condition-terminal.json').exists()),None)
                row={'id':ident,'state':'MISSING','arithmetic_pair_verified':False,'full_condition_accepted':False}
                rows.append(row)
                if not cf.exists() or gf is None:continue
                try:
                    c=json.loads(cf.read_text());g=json.loads(gf.read_text());check_condition(c);check_condition(g)
                    row.update(cpu_state=c['state'],gpu_state=g['state'],gpu_device=g['actual_device'])
                    if c['plan_hash']!=g['plan_hash'] or c['source_sha256']!=g['source_sha256'] or c['source_sha256']!=case['source_sha256']:raise ValueError('condition identity mismatch')
                    expected={'version':VERSION,'task':Task().__dict__,'policy':Policy().__dict__,'wavelength_nm':w,'field_h':h,'mode':'verify','prescription_hash':digest(case['prescription']),'implementation_signature':IMPLEMENTATION_SIGNATURE}
                    if digest(expected)!=c['plan_hash']:raise ValueError('unexpected task or relaxed numerical policy')
                    if c.get('code_commit')!=g.get('code_commit'):raise ValueError('CPU/GPU code commit mismatch')
                    if c['actual_device']!='cpu' or not g['actual_device'].startswith('cuda:'):raise ValueError('device mismatch')
                    if len(c['runs'])!=len(g['runs']):raise ValueError('different actual grid schedules')
                    deltas=[]
                    for cr,gr in zip(c['runs'],g['runs']):
                        for k in ('pupil_n','angular_grid_n','pixel_n','pitch_um','axis'):
                            if cr[k]!=gr[k]:raise ValueError('grid mismatch')
                        oc=decode_otf(cr['otf']);og=decode_otf(gr['otf'])
                        oi=float(np.max(np.abs(oc-og)));ri=abs(cr['raw_integral']-gr['raw_integral'])/max(abs(cr['raw_integral']),1e-300)
                        proj=max(float(np.max(np.abs(np.array(cr[k])-np.array(gr[k]))))/max(float(np.max(cr[k])),1e-300) for k in ('raw_projection_x','raw_projection_y'))
                        for rr,folder in ((cr,cf.parent),(gr,gf.parent)):
                            if 'raw_artifact' in rr:
                                raw=rr['raw_artifact'];f=folder/raw['file'];sha=hashlib.sha256()
                                with f.open('rb') as stream:
                                    for chunk in iter(lambda:stream.read(1024*1024),b''):sha.update(chunk)
                                if sha.hexdigest()!=raw['sha256']:raise ValueError('PSF checksum mismatch')
                                image=np.load(f,mmap_mode='r',allow_pickle=False)
                                if not np.allclose(image.sum(0),rr['raw_projection_x'],rtol=1e-10,atol=1e-14):raise ValueError('PSF/projection mismatch')
                                if not np.allclose(image.sum(1),rr['raw_projection_y'],rtol=1e-10,atol=1e-14):raise ValueError('PSF/projection mismatch')
                        deltas.append({'complex_otf_max_abs':oi,'integral_relative':ri,'projection_relative_max_abs':proj,'passed':oi<=1e-6 and ri<=1e-6 and proj<=1e-6})
                    row.update(state='CHECKED' if deltas else 'NO_PSF_ON_EITHER_DEVICE',grid_deltas=deltas,
                        arithmetic_pair_verified=bool(deltas) and all(d['passed'] for d in deltas),
                        full_condition_accepted=bool(c['numerically_verified'] and g['numerically_verified'] and c['model_scope_validated'] and g['model_scope_validated'] and deltas and all(d['passed'] for d in deltas)))
                except Exception as exc:row.update(state='VERIFICATION_FAILED',error=repr(exc))
    d={'schema':'QfilmRWIndependentMatrixVerification-v1','status':'completed','observed_unix':time.time(),
        'condition_count':len(rows),'arithmetic_pairs_verified':sum(r['arithmetic_pair_verified'] for r in rows),
        'full_conditions_accepted':sum(r['full_condition_accepted'] for r in rows),'conditions':rows,
        'training_eligible':False,'J_task':None,'all_11_usable':False}
    atomic_json(root/'usability-terminal.json',d,immutable=True);return d

def collect(root):
    from evaluator.qfilmrw.field_audit import cuda_device
    rows=[]
    for case in cases():
        for h in (0.,.5,1.):
            for wave in WAVES:
                ident=f"{case['name']}-h{h:g}-w{wave:g}"
                paths=[p/ident/'condition-terminal.json' for p in (root/'cuda0',root/'cuda1')]
                found=[p for p in paths if p.exists()]
                row={'id':ident,'state':'MISSING','numerical_verified':False,'full_condition_accepted':False};rows.append(row)
                if not found:continue
                try:
                    if len(found)!=1:raise ValueError('duplicate primary GPU result')
                    file=found[0];d=json.loads(file.read_text())
                    if not d['actual_device'].startswith('cuda:'):raise ValueError('CPU record rejected by GPU-only collector')
                    if d['source_sha256']!=case['source_sha256']:raise ValueError('source mismatch')
                    if d['runs']:
                        check_condition(d)
                        plan=d['plan']
                        if plan['version']!=VERSION or plan['implementation_signature']!=IMPLEMENTATION_SIGNATURE:raise ValueError('GPU kernel identity mismatch')
                        if digest(plan['task'])!=digest(Task().__dict__) or digest(plan['policy'])!=digest(Policy().__dict__):raise ValueError('unexpected or relaxed GPU numerical policy')
                        if plan['prescription_hash']!=digest(case['prescription']) or plan['wavelength_nm']!=wave or plan['field_h']!=h:raise ValueError('GPU optical condition mismatch')
                        if plan.get('object_angle_rad') is None:raise ValueError('missing audited field angle')
                        if h and (not plan.get('field_contract') or plan['field_contract']['source_sha256']!=case['source_sha256']):raise ValueError('missing source-bound field contract')
                    for run in d['runs']:
                        if 'raw_artifact' not in run:continue
                        record=run['raw_artifact'];path=file.parent/record['file'];sha=hashlib.sha256()
                        with path.open('rb') as stream:
                            for chunk in iter(lambda:stream.read(1024*1024),b''):sha.update(chunk)
                        if sha.hexdigest()!=record['sha256']:raise ValueError('raw PSF checksum mismatch')
                        image=np.load(path,mmap_mode='r',allow_pickle=False)
                        tensor=torch.as_tensor(np.array(image,copy=True),device=cuda_device(d['actual_device']))
                        for axis,key in ((0,'raw_projection_x'),(1,'raw_projection_y')):
                            expected=torch.as_tensor(run[key],device=tensor.device,dtype=torch.float64)
                            if not bool(torch.allclose(tensor.sum(axis),expected,rtol=1e-10,atol=1e-14)):raise ValueError('GPU raw PSF projection mismatch')
                        del tensor
                    row.update(state=d['state'],device=d['actual_device'],numerical_verified=d['numerically_verified'],
                        full_condition_accepted=bool(d['numerically_verified'] and d['model_scope_validated']))
                except Exception as exc:row.update(state='VERIFICATION_FAILED',error=repr(exc))
    data={'schema':'QfilmRWGPUOnlyVerification-v2','status':'completed','observed_unix':time.time(),'execution_policy':'GPU_ONLY_NO_CPU_REFERENCE',
        'condition_count':len(rows),'numerical_conditions_verified':sum(r['numerical_verified'] for r in rows),
        'full_conditions_accepted':sum(r['full_condition_accepted'] for r in rows),'conditions':rows,'J_task':None,'training_eligible':False,
        'cpu_gpu_parity_required':False,'cross_gpu_repeatability':'not part of this primary matrix'}
    atomic_json(root/'usability-terminal.json',data,immutable=True);return data

def orchestrate(a):
    require_handover()
    if torch.cuda.device_count()<2:raise RuntimeError('two actual CUDA devices required for this benchmark')
    if a.field_contract is None:raise RuntimeError('audited field contract required; run qfilmrw_field_audit_gpu first')
    lanes=[('cuda0','cuda:0',list(CAMERA_SAMPLES[::2])),('cuda1','cuda:1',list(CAMERA_SAMPLES[1::2]))]
    a.output_dir.mkdir(parents=True,exist_ok=True);children=[]
    tests=subprocess.run([sys.executable,str(ROOT/'validation/qfilmrw_tests.py')],capture_output=True,text=True,timeout=300)
    (a.output_dir/'qfilmrw-tests.log').write_text(tests.stdout+tests.stderr)
    atomic_json(a.output_dir/'preflight-terminal.json',{'status':'completed' if tests.returncode==0 else 'failed','returncode':tests.returncode,
        'observed_unix':time.time(),'torch_version':str(torch.__version__),'cuda_devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    if tests.returncode:raise RuntimeError('QfilmRW preflight tests failed; benchmark not launched')
    try:
        for label,device,names in lanes:
            folder=a.output_dir/label;folder.mkdir(exist_ok=True)
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--device',device,'--mode','verify','--output-dir',str(folder),'--names',*names,'--grid-n',str(a.grid_n),'--threads',str(a.threads),'--condition-seconds',str(a.condition_seconds),'--save-raw','--field-contract',str(a.field_contract)]
            log=(a.output_dir/(label+'.log')).open('w');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            children.append((label,p,log))
        while any(p.poll() is None for _,p,_ in children):
            atomic_json(a.output_dir/'driver-progress.json',{'updated_unix':time.time(),
                'lanes':[{'lane':label,'pid':p.pid,'returncode':p.poll(),'progress':json.loads((a.output_dir/label/'progress.json').read_text()) if (a.output_dir/label/'progress.json').exists() else None} for label,p,_ in children]})
            time.sleep(10)
        d=collect(a.output_dir)
        atomic_json(a.output_dir/'driver-terminal.json',{'status':'completed' if all(p.returncode==0 for _,p,_ in children) else 'completed_with_errors',
            'finished_unix':time.time(),'children':[{'lane':label,'returncode':p.returncode} for label,p,_ in children],
            'execution_policy':'GPU_ONLY_NO_CPU_REFERENCE','numerical_conditions_verified':d['numerical_conditions_verified'],'full_conditions_accepted':d['full_conditions_accepted']},immutable=True)
        return 0 if all(p.returncode==0 for _,p,_ in children) else 2
    finally:
        for _,p,log in children:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--orchestrate',action='store_true');ap.add_argument('--collect',action='store_true')
    ap.add_argument('--device',choices=('cuda:0','cuda:1'),default='cuda:0');ap.add_argument('--names',choices=CAMERA_SAMPLES,nargs='+',default=list(CAMERA_SAMPLES))
    ap.add_argument('--mode',choices=('survey','verify'),default='verify')
    ap.add_argument('--output-dir',type=Path,default=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts/qfilmrw_local')))
    ap.add_argument('--fields',nargs='+',type=float,default=[0.,.5,1.]);ap.add_argument('--waves',nargs='+',type=float,default=list(WAVES))
    ap.add_argument('--pupils',nargs='+',type=int);ap.add_argument('--grid-n',type=int,default=1024);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--field-contract',type=Path,default=ROOT/'validation/fixtures/qfilmrw_audited_fields_20261031.json',help='completed GPU field-audit JSON or its exported native field contract')
    ap.add_argument('--condition-seconds',type=float,default=7200.);ap.add_argument('--save-raw',action='store_true')
    ap.add_argument('--max-live-elements',type=int,default=300_000_000)
    a=ap.parse_args()
    if not set(a.fields)<={0.,.5,1.} or not set(a.waves)<=set(WAVES):ap.error('unsupported matrix subset')
    if a.collect:collect(a.output_dir);return 0
    return orchestrate(a) if a.orchestrate else worker(a)

if __name__=='__main__':raise SystemExit(main())
