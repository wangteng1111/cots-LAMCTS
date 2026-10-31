#!/usr/bin/env python3
"""Verify retired CPU routes and exercise audited fields on actual GPUs."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from evaluator.qfilm04_io import atomic_json
from evaluator.qfilmrw.rays import trace_lens
from evaluator.qfilmrw.fourier import reference_source
from validation.qfilmrw_real11 import cases,require_handover

def main():
    require_handover();art=Path(os.environ['COTS_JOB_ARTIFACT_DIR']);art.mkdir(parents=True,exist_ok=True)
    result=subprocess.run([sys.executable,str(ROOT/'validation/qfilmrw_tests.py')],capture_output=True,text=True,timeout=600)
    (art/'gpu-only-tests.log').write_text(result.stdout+result.stderr)
    report={'schema':'QfilmRWGPUOnlyPolicyProbe-v1','status':'running','test_returncode':result.returncode,'torch_version':str(torch.__version__),
        'execution_policy':'GPU_ONLY_NO_CPU_REFERENCE','code_commit':os.environ.get('COTS_GIT_COMMIT'),'cases':[]}
    if result.returncode:
        report.update(status='failed',finished_unix=time.time());atomic_json(art/'gpu-only-policy-terminal.json',report,immutable=True);return 2
    job='000-000-020-qfilmrw-gpu-field-refine-20261031-01'
    audit=Path('/var/lib/cots-lamcts/jobs')/job/'artifacts/field-audit-refined-terminal.json'
    data=json.loads(audit.read_text());domains={x['name']:x for x in data['cases']}
    for i,case in enumerate(cases()):
        device=f'cuda:{i%2}';domain=domains[case['name']]
        for field in domain['native_fields']:
            row={'name':case['name'],'field_h':field['field_h'],'device':device,'calibration_state':field['state']}
            report['cases'].append(row)
            if field['state']!='calibrated':row['state']='FIELD_CALIBRATION_UNRESOLVED';continue
            try:
                state=trace_lens(case['prescription'],550.,field['field_h'],64,device,object_angle_rad=field['object_angle_rad'])
                qx,qy,coeff,piston,guard=reference_source(state)
                if not bool(torch.isfinite(coeff[state.active]).all()):raise ValueError('nonfinite CUDA boundary field')
                row.update(state='CUDA_BOUNDARY_FIELD_OK',actual_device=str(state.x.device),wavefront=state.metadata,phase_neighbor_cycles=guard,
                    object_angle_rad=field['object_angle_rad'],target_image_height_mm=field['target_image_height_mm'])
            except Exception as exc:row.update(state='FIELD_NOT_AVAILABLE',error=repr(exc))
            atomic_json(art/'progress.json',{'updated_unix':time.time(),'row':row})
    report.update(status='completed',finished_unix=time.time(),optical_cpu_executions=0,condition_count=len(report['cases']),
        boundary_fields_ok=sum(r['state']=='CUDA_BOUNDARY_FIELD_OK' for r in report['cases']))
    atomic_json(art/'gpu-only-policy-terminal.json',report,immutable=True);return 0
if __name__=='__main__':raise SystemExit(main())
