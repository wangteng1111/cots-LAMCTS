#!/usr/bin/env python3
"""Independent first-surface bracketing of unresolved chief coordinates on CUDA.

Every requested endpoint remains in the record. A bounded search failure is
never a proof of darkness. This is a coordinate/visibility audit, not PSF QA.
"""
from __future__ import annotations
import json,math,os,sys,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from evaluator.qfilm02 import _surfaces_at,_sag_and_grad
from evaluator.qfilmrw.field_audit import cuda_device,front_to_stop,propagate,bundle
from evaluator.qfilm04_io import atomic_json
from validation.qfilmrw_real11 import cases,require_handover

@torch.inference_mode()
def chiefs(p,angles,device):
    dev=cuda_device(device);surfs=_surfaces_at(p['surfaces'],550.)
    angle=torch.tensor(angles,dtype=torch.float64,device=dev).reshape(-1,1)
    stop=p['stop_z_mm'];rs=p['design_spec']['stop_diameter_mm']/2;zs=min(stop,surfs[0].z)-20
    radius=float(surfs[0].clear_aperture)/2
    xx=torch.linspace(-radius,radius,257,device=dev,dtype=torch.float64)
    sag,_,domain=_sag_and_grad(xx.abs(),surfs[0].R,surfs[0].conic,surfs[0].asphere)
    origin=xx[None,:]-(surfs[0].z+sag[None,:]-zs)*torch.tan(angle)
    sx,sy,valid=front_to_stop(surfs,angle,zs,stop,origin,torch.zeros_like(origin))
    valid &=domain[None,:]
    exact=valid&(sx.abs()<1e-8);has_exact=exact.any(-1)
    crossing=valid[:,:-1]&valid[:,1:]&(sx[:,:-1]*sx[:,1:]<0)
    count=crossing.sum(-1);idx=crossing.to(torch.int64).argmax(-1)
    ii=torch.arange(len(angles),device=dev)
    lo=origin[ii,idx].reshape(-1,1);hi=origin[ii,idx+1].reshape(-1,1)
    flo=sx[ii,idx].reshape(-1,1);ok=(count==1).reshape(-1,1)
    for _ in range(19):
        mid=(lo+hi)/2;fm,_,vm=front_to_stop(surfs,angle,zs,stop,mid,torch.zeros_like(mid))
        ok &=vm
        left=flo*fm<=0
        hi=torch.where(left,mid,hi);lo=torch.where(left,lo,mid);flo=torch.where(left,flo,fm)
    root=(lo+hi)/2
    exact_idx=exact.to(torch.int64).argmax(-1)
    root=torch.where(has_exact[:,None],origin[ii,exact_idx][:,None],root)
    ok |=has_exact[:,None]
    stopx,stopy,v=front_to_stop(surfs,angle,zs,stop,root,torch.zeros_like(root))
    residual=torch.hypot(stopx,stopy);ok &=v&(residual<=1e-4)
    virtual,_=propagate(surfs,angle,root,torch.zeros_like(root),zs,stop,rs,p['image_z_mm'],ok,clip=False)
    physical,_=propagate(surfs,angle,root,torch.zeros_like(root),zs,stop,rs,p['image_z_mm'],ok,clip=True)
    return [{'object_angle_rad':float(a),'geometric_chief_height_mm':float(virtual[4][i,0]) if bool(virtual[6][i,0]) else None,
      'stop_residual_mm':float(residual[i,0]),'physical_chief_valid':bool(physical[6][i,0]),
      'bracket_count':int(count[i]),'method':'first_surface_257_node_bracket_19_bisections_cuda'} for i,a in enumerate(angles)]

def main():
    require_handover();art=Path(os.environ['COTS_JOB_ARTIFACT_DIR']);art.mkdir(parents=True,exist_ok=True)
    job='000-000-020-qfilmrw-gpu-field-audit-20261031-02'
    base=Path('/var/lib/cots-lamcts/jobs')/job/'artifacts/field-audit-terminal.json'
    data=json.loads(base.read_text())
    if data['status']!='completed' or len(data['cases'])!=11:raise RuntimeError('complete source audit required')
    data=copy.deepcopy(data);lookup={c['name']:c for c in cases()};details=[]
    for i,row in enumerate(data['cases']):
        unresolved=[f for f in row['native_fields'] if f['state']!='calibrated']
        if row['fullframe_chief_calibration']['state']!='calibrated':unresolved.append(row['fullframe_chief_calibration'])
        if not unresolved:continue
        device=f'cuda:{i%2}';p=lookup[row['name']]['prescription']
        if row['source_sha256']!=lookup[row['name']]['source_sha256']:raise RuntimeError('source mismatch')
        angles=[math.radians(.25*j) for j in range(341)]
        scan=chiefs(p,angles,device);branch=[];last=None
        for v in scan:
            y=v['geometric_chief_height_mm']
            if y is None or (last is not None and y<last-1e-5):break
            branch.append(v);last=y
        detail={'name':row['name'],'device':device,'scan':scan,'refinements':[]};details.append(detail)
        for field in unresolved:
            previous=copy.deepcopy(field);target=field['target_image_height_mm']
            pair=next(((a,b) for a,b in zip(branch,branch[1:]) if a['geometric_chief_height_mm']<=target<=b['geometric_chief_height_mm']),None)
            update={'previous_calibration':previous,'independent_chief_search_max_height_mm':max((v['geometric_chief_height_mm'] for v in branch),default=None)}
            field.update(update)
            if not pair:continue
            low,high=pair;lo=low['object_angle_rad'];hi=high['object_angle_rad']
            ylo=low['geometric_chief_height_mm'];yhi=high['geometric_chief_height_mm']
            proposal=lo+(hi-lo)*(target-ylo)/(yhi-ylo);found=None
            for _ in range(12):
                v=chiefs(p,[proposal],device)[0];y=v['geometric_chief_height_mm']
                if y is None:break
                if abs(y-target)<=.001:found=v;break
                if y<target:lo=proposal
                else:hi=proposal
                proposal=(lo+hi)/2
            if found is None:continue
            field.update(state='calibrated',object_angle_rad=proposal,geometric_chief_height_mm=found['geometric_chief_height_mm'],
                residual_mm=abs(found['geometric_chief_height_mm']-target),coordinate_definition='aperture-free geometric chief at 550nm; independent first-surface bracketing',chief_search=found)
            if 'field_h' in field:
                field['wavelength_visibility']=[bundle(p,w,[proposal],128,device)[0] for w in (430.,470.,510.,550.,590.,630.,670.)]
                field['all_waves_have_transmitted_samples']=all(v['physical_transmitted']>0 for v in field['wavelength_visibility'])
            else:field['visibility_550']=bundle(p,550.,[proposal],256,device)[0]
            detail['refinements'].append({'target_image_height_mm':target,'angle_rad':proposal,'residual_mm':field['residual_mm']})
        print('INDEPENDENT_CHIEF_TERMINAL',row['name'],detail['refinements'],flush=True)
        atomic_json(art/'chief-search-checkpoint.json',{'details':details})
    data.update(refinement_job=os.environ.get('COTS_JOB_ID'),refinement_code=os.environ.get('COTS_GIT_COMMIT'),refined_unix=time.time())
    atomic_json(art/'independent-chief-search-terminal.json',{'status':'completed','details':details},immutable=True)
    atomic_json(art/'field-audit-refined-terminal.json',data,immutable=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
