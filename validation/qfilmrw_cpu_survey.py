#!/usr/bin/env python3
"""Parallel CPU-only survey of the same 231 conditions; never a GPU/full-pass claim."""
import argparse,json,os,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from validation.qfilmrw_real11 import CAMERA_SAMPLES
from evaluator.qfilm04_io import atomic_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--lanes',type=int,default=4);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True);children=[]
    try:
        for i in range(a.lanes):
            names=CAMERA_SAMPLES[i::a.lanes]
            if not names:continue
            label=f'lane{i}';out=a.output_dir/label;log=(a.output_dir/(label+'.log')).open('w')
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'validation/qfilmrw_real11.py'),'--mode','survey','--device','cpu','--output-dir',str(out),'--names',*names,'--threads',str(a.threads)],stdout=log,stderr=subprocess.STDOUT)
            children.append((out,p,log))
        while any(p.poll() is None for _,p,_ in children):
            rows=[]
            for out,p,_ in children:
                f=out/'matrix-checkpoint.json'
                if f.exists():rows.extend(json.loads(f.read_text())['conditions'])
            atomic_json(a.output_dir/'survey-progress.json',{'updated_unix':time.time(),'conditions_completed':len(rows),'completed_psf_grids':sum(r['completed_psf_grids'] for r in rows),'lanes':[p.poll() for _,p,_ in children]})
            time.sleep(5)
        reports=[json.loads((out/'matrix-terminal.json').read_text()) for out,_,_ in children if (out/'matrix-terminal.json').exists()]
        rows=[r for report in reports for r in report['conditions']]
        summary={'schema':'QfilmRWCPU231Survey-v1','status':'completed' if len(rows)==231 and all(p.returncode==0 for _,p,_ in children) else 'completed_with_errors',
            'finished_unix':time.time(),'condition_count':len(rows),'completed_psf_grids':sum(r['completed_psf_grids'] for r in rows),
            'numerically_verified':sum(r['numerically_verified'] for r in rows),'model_scope_validated':False,'gpu_executions':0,'training_eligible':False,
            'J_task':None,'conditions':rows,'code_commits':list({r.get('code_commit') for r in reports}),
            'code_sha256':[r['code_sha256'] for r in reports]}
        export=a.output_dir/'export';export.mkdir(exist_ok=True)
        for name in CAMERA_SAMPLES:
            native=[]
            for out,_,_ in children:
                for f in sorted(out.glob(name+'-h*/condition-terminal.json')):
                    d=json.loads(f.read_text())
                    keep={k:v for k,v in d.items() if k not in ('runs','selected')}
                    keep['runs']=[{k:v for k,v in r.items() if k not in ('raw_projection_x','raw_projection_y')} for r in d['runs']]
                    keep['native_terminal_sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
                    native.append(keep)
            atomic_json(export/(name+'.json'),{'name':name,'conditions':native,'omitted_raw_projections':'retained in native local condition files; this export is a compact diagnostic record, not full verification'})
        atomic_json(export/'summary.json',summary,immutable=True)
        return 0 if summary['status']=='completed' else 2
    finally:
        for _,p,log in children:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()
if __name__=='__main__':raise SystemExit(main())
