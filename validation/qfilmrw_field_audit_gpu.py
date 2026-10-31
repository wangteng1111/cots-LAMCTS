#!/usr/bin/env python3
"""GPU-only 11-prescription field-domain, visibility and aiming audit."""
from __future__ import annotations
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from evaluator.qfilmrw.field_audit import bundle,efl_gpu,cuda_device,aim,propagate
from evaluator.qfilm02 import _surfaces_at
from evaluator.qfilm04_io import atomic_json
from validation.qfilmrw_real11 import cases,require_handover


def geometric_chief(p,angles,device):
    dev=cuda_device(device);surfs=_surfaces_at(p['surfaces'],550.);angle=torch.tensor(angles,device=dev,dtype=torch.float64).reshape(-1,1)
    uv=torch.zeros((1,2),device=dev,dtype=torch.float64);stop=p['stop_z_mm'];rs=p['design_spec']['stop_diameter_mm']/2;zs=min(stop,surfs[0].z)-20
    x,y,v,_=aim(surfs,angle,uv,stop,rs,zs);tr,_=propagate(surfs,angle,x,y,zs,stop,rs,p['image_z_mm'],v,clip=False)
    return [float(tr[4][i,0]) if bool(tr[6][i,0]) else None for i in range(len(angles))]


def calibrate(p,sweep,target,device):
    # First geometric-chief crossing on the initial increasing branch. Physical
    # transmission is checked independently; it cannot be inferred from this
    # virtual (aperture-free) coordinate ray.
    if target==0:return {'state':'calibrated','target_image_height_mm':0.,'object_angle_rad':0.,'geometric_chief_height_mm':0.,'residual_mm':0.}
    branch=[];last=None
    for r in sweep:
        xy=r['geometric_chief_mm']
        if xy is None:break
        height=xy[0]
        if last is not None and height<last-.05:break
        branch.append((math.radians(r['object_half_angle_deg']),height));last=height
    bracket=next(((a,b) for a,b in zip(branch,branch[1:]) if a[1]<=target<=b[1]),None)
    if bracket is None:return {'state':'TARGET_NOT_REACHED_ON_CALIBRATED_BRANCH','target_image_height_mm':target,'largest_sampled_chief_height_mm':max((v for _,v in branch),default=None)}
    (lo,_),(hi,_)=bracket
    for _ in range(17):
        mid=(lo+hi)/2;y=geometric_chief(p,[mid],device)[0]
        if y is None:return {'state':'GEOMETRIC_CHIEF_ROOT_FAILED','target_image_height_mm':target}
        if y<target:lo=mid
        else:hi=mid
    theta=(lo+hi)/2;y=geometric_chief(p,[theta],device)[0]
    return {'state':'calibrated' if y is not None and abs(y-target)<=.001 else 'FIELD_CALIBRATION_UNRESOLVED',
        'target_image_height_mm':target,'object_angle_rad':theta,'geometric_chief_height_mm':y,'residual_mm':abs(y-target) if y is not None else None,
        'coordinate_definition':'aperture-free geometric chief at 550nm; physical visibility independently audited'}


def preflight(device):
    p={'surfaces':[{'radius':'inf','thickness':5.,'n_after':1.,'v_after':50.,'clear_aperture':5.},{'radius':'inf','thickness':15.,'n_after':1.,'v_after':50.,'clear_aperture':5.}],
        'stop_z_mm':2.,'image_z_mm':20.,'design_spec':{'stop_diameter_mm':2.,'efl_target_mm':50.}}
    theta=.1;row=bundle(p,550.,[theta],16,device)[0]
    if row['physical_transmitted']!=row['geometric_aim_converged']:raise AssertionError('plane visibility preflight')
    if abs(row['geometric_chief_mm'][0]-18*math.tan(theta))>1e-7:raise AssertionError('chief-coordinate preflight')
    if not row['entry_raster_transmitted']:raise AssertionError('independent entrance raster preflight')
    return {'device':device,'passed':True,'plane_chief_mm':row['geometric_chief_mm'][0]}


def worker(a):
    cuda_device(a.device);torch.set_num_threads(2);a.output_dir.mkdir(parents=True,exist_ok=True)
    source=json.loads((ROOT/'validation/fixtures/qfilmrw_field_domains.json').read_text());sources={r['name']:r for r in source['sources']}
    report={'schema':'QfilmRWGPUFieldAudit-v1','status':'running','device':a.device,'gpu_only_optics':True,'code_commit':os.environ.get('COTS_GIT_COMMIT'),
        'started_unix':time.time(),'preflight':preflight(a.device),'cases':[]}
    for case in cases():
        if case['name'] not in a.names:continue
        name=case['name'];p=case['prescription'];m=sources[name];rs=float(m['source_r_sensor_mm']);frame=math.hypot(36,24)/2;ef=efl_gpu(p,550.,a.device)
        t=time.monotonic();old=[math.atan(h*frame/p['design_spec']['efl_target_mm']) for h in (0.,.5,1.)]
        hi=min(math.radians(85),max(math.atan(rs/ef)*1.8,old[-1]*1.3))
        angles=sorted(set([i*hi/32 for i in range(33)]+old+[math.atan(rs/ef)]))
        row={'name':name,'metadata':m,'source_sha256':case['source_sha256'],'efl_independent_gpu_mm':ef,
            'efl_import_relative_difference':abs(ef-p['design_spec']['efl_target_mm'])/abs(ef),
            'fullframe_half_diagonal_mm':frame,'radius_scope':'SOURCE_RADIUS_SMALLER_THAN_FULLFRAME' if frame-rs>.05 else 'SOURCE_RADIUS_ROUNDED_NEAR_FULLFRAME' if rs<frame else 'SOURCE_RADIUS_COVERS_FULLFRAME',
            'sweep':[],'native_fields':[]}
        report['cases'].append(row)
        for start in range(0,len(angles),8):
            row['sweep'].extend(bundle(p,550.,angles[start:start+8],64,a.device))
            atomic_json(a.output_dir/'progress.json',{'updated_unix':time.time(),'name':name,'stage':'angular_sweep','angles_completed':len(row['sweep']),'angles_total':len(angles),'device':a.device})
            print('FIELD_AUDIT',name,'sweep',len(row['sweep']),len(angles),flush=True)
        row['legacy_fields_550']=[min(row['sweep'],key=lambda r:abs(r['object_half_angle_deg']-math.degrees(th))) for th in old]
        for h in (0.,.5,1.):
            f={'field_h':h,**calibrate(p,row['sweep'],h*rs,a.device)}
            row['native_fields'].append(f)
            if f['state']=='calibrated':
                f['wavelength_visibility']=[bundle(p,w,[f['object_angle_rad']],128,a.device)[0] for w in (430.,470.,510.,550.,590.,630.,670.)]
                f['all_waves_have_transmitted_samples']=all(v['physical_transmitted']>0 for v in f['wavelength_visibility'])
        row['fullframe_chief_calibration']=calibrate(p,row['sweep'],frame,a.device)
        full=row['fullframe_chief_calibration']
        if full['state']=='calibrated':full['visibility_550']=bundle(p,550.,[full['object_angle_rad']],256,a.device)[0]
        # Explicitly retain all requested endpoints. Failed edge calibration is
        # not replaced with a smaller field that happens to pass.
        row['elapsed_seconds']=time.monotonic()-t
        atomic_json(a.output_dir/(name+'-field-terminal.json'),row,immutable=True)
        atomic_json(a.output_dir/'audit-checkpoint.json',report)
        print('FIELD_AUDIT_CASE_TERMINAL',name,flush=True)
    report.update(status='completed',finished_unix=time.time(),case_count=len(report['cases']))
    atomic_json(a.output_dir/'audit-terminal.json',report,immutable=True)
    return 0


def orchestrate(a):
    require_handover()
    if torch.cuda.device_count()<2:raise RuntimeError('two GPUs required; no CPU fallback')
    a.output_dir.mkdir(parents=True,exist_ok=True);names=[c['name'] for c in cases()];children=[]
    try:
        for i in (0,1):
            folder=a.output_dir/f'gpu{i}';log=(a.output_dir/f'gpu{i}.log').open('w')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--device',f'cuda:{i}','--output-dir',str(folder),'--names',*names[i::2]],stdout=log,stderr=subprocess.STDOUT)
            children.append((i,p,log))
        while any(p.poll() is None for _,p,_ in children):
            atomic_json(a.output_dir/'driver-progress.json',{'updated_unix':time.time(),'gpu_only_optics':True,'lanes':[{'device':f'cuda:{i}','returncode':p.poll()} for i,p,_ in children]});time.sleep(5)
        reports=[json.loads((a.output_dir/f'gpu{i}/audit-terminal.json').read_text()) for i,p,_ in children if (a.output_dir/f'gpu{i}/audit-terminal.json').exists()]
        data={'schema':'QfilmRWGPUFieldAuditCombined-v1','status':'completed' if all(p.returncode==0 for _,p,_ in children) else 'completed_with_errors','finished_unix':time.time(),
              'gpu_only_optics':True,'children':[{'device':f'cuda:{i}','returncode':p.returncode} for i,p,_ in children],'cases':[c for r in reports for c in r['cases']]}
        atomic_json(a.output_dir/'field-audit-terminal.json',data,immutable=True)
        return 0 if data['status']=='completed' and len(data['cases'])==11 else 2
    finally:
        for _,p,log in children:
            if p.poll() is None:p.terminate();p.wait(timeout=20)
            log.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--orchestrate',action='store_true');ap.add_argument('--device',default='cuda:0');ap.add_argument('--names',nargs='+',default=[c['name'] for c in cases()]);ap.add_argument('--output-dir',type=Path,default=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts/field-audit')))
    a=ap.parse_args();return orchestrate(a) if a.orchestrate else worker(a)
if __name__=='__main__':raise SystemExit(main())
