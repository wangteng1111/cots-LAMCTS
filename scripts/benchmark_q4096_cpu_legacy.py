#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures as cf, json, os, subprocess, sys, tempfile, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VENV=Path('/var/lib/cots-lamcts/venv/bin/python')
if VENV.exists() and sys.prefix != str(VENV.parent.parent):
    os.execv(str(VENV),[str(VENV),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
CACHE=Path('/var/lib/cots-lamcts/q4096_mandler_fixed_shared'); HASH='0dcde5089bd66deba991214558637fe906f6c2a52530619850e6c5bab0d684da'
def load_sample(n):
    out=[]
    for p in sorted(CACHE.glob('*.json')):
        try:
            r=json.loads(p.read_text());st=tuple(map(int,r.get('state',[])))
            if len(st)==8 and st[4:]==(2,2,2,2) and r.get('config_hash')==HASH and 'J' in r:
                out.append((st,float(r['J'])))
                if len(out)>=n:return out
        except Exception:pass
    raise RuntimeError(f'only {len(out)} cached states found')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=16);ap.add_argument('--workers',type=int,default=16);a=ap.parse_args();sample=load_sample(a.n)
    helper=ROOT/'scripts/eval_q4096_cpu_state.py';td=Path(tempfile.mkdtemp(prefix='q4096cpu-'))
    def one(item):
        st,expected=item;out=td/(str(abs(hash(st)))+'.json');cmd=[str(VENV),'-u',str(helper),'--state',json.dumps(st),'--out',str(out)]
        z=subprocess.run(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=900,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1'})
        if z.returncode!=0:raise RuntimeError(z.stdout[-3000:])
        r=json.loads(out.read_text());return abs(float(r['J'])-expected)
    t=time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:diffs=list(ex.map(one,sample))
    sec=time.perf_counter()-t
    res={'schema':'q4096-legacy-cpu-batch-v1','n':a.n,'workers':a.workers,'batch_sec':sec,'states_per_sec':a.n/sec,'effective_sec_per_state':sec/a.n,'max_abs_J_diff':max(diffs)}
    art=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'));art.mkdir(parents=True,exist_ok=True);(art/'q4096_cpu_legacy_benchmark.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()
