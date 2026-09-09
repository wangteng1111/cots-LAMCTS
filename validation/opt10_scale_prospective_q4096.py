#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
ART=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'));ART.mkdir(parents=True,exist_ok=True)
SOURCE=Path('/var/lib/cots-lamcts/jobs/opt10-pretrain-scale-meta-20260910-0061/artifacts/opt10_pretrain_scale_proposals.json')

def decode(st):
    design=[(int(x)//2,bool(int(x)%2)) for x in st[:4]];gr=[(.20,.29,.38,.47,.56),(7.50,8.32,9.14,9.96,10.78),(11.,12.18,13.36,14.54,15.72),(.20,.29,.38,.47,.56)]
    return tuple(design),tuple(gr[i][int(st[4+i])] for i in range(4))
def one(st,out):
    from evaluator import market_final_direct_q4096_v4 as E
    d,g=decode(st);r=E.evaluate_final_with_gaps(d,g,4096)
    q={'state':list(map(int,st)),'J':float(r['merit_J']),'metrics':{k:float(r[k]) for k in ['efl','fno','mtf_mean','mtf_geomean_reg','mtf_p10','onaxis_pupil_fill','min_illum','dist_max','lca']},'costs':r['costs'],'config_hash':r['config_hash']}
    Path(out).write_text(json.dumps(q));print(json.dumps(q),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state');ap.add_argument('--out');a=ap.parse_args()
    if a.state:return one(json.loads(a.state),a.out)
    if not SOURCE.exists():raise FileNotFoundError(SOURCE)
    proposals=json.loads(SOURCE.read_text());tmp=ART/'eval';tmp.mkdir(exist_ok=True)
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1'})
    def runone(group,i,st):
        p=tmp/f'{group}_{i}.json';cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--state',json.dumps(st),'--out',str(p)]
        z=subprocess.run(cmd,cwd=str(ROOT),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=900)
        if z.returncode!=0:return {'state':st,'error':z.stdout[-4000:]}
        return json.loads(p.read_text())
    jobs=[]
    for g,sts in proposals.items():
        for i,st in enumerate(sts):jobs.append((g,i,st))
    rows={g:[None]*len(sts) for g,sts in proposals.items()}
    with ThreadPoolExecutor(max_workers=10) as ex:
        fs={ex.submit(runone,g,i,st):(g,i) for g,i,st in jobs}
        for f in as_completed(fs):
            g,i=fs[f];rows[g][i]=f.result();print('eval',g,i,rows[g][i].get('J',rows[g][i].get('error')),flush=True)
    import numpy as np
    from scipy.stats import mannwhitneyu
    def stats(rs):
        j=np.asarray([x['J'] for x in rs if x and 'J' in x],float)
        return {'n':int(len(j)),'mean_J':float(j.mean()),'median_J':float(np.median(j)),'best_J':float(j.min()),'worst_J':float(j.max())}
    out={'schema':'OPT1.0-scale-prospective-q4096-v1','groups':{g:{'stats':stats(rs),'rows':rs} for g,rs in rows.items()},'tests':{}}
    sj=np.asarray([x['J'] for x in rows['scratch']],float);rj=np.asarray([x['J'] for x in rows['random']],float)
    for g in rows:
        if g in {'scratch','random'}:continue
        x=np.asarray([q['J'] for q in rows[g]],float)
        out['tests'][g]={'vs_scratch_mwu_less':float(mannwhitneyu(x,sj,alternative='less').pvalue),'vs_random_mwu_less':float(mannwhitneyu(x,rj,alternative='less').pvalue),'P_better_scratch':float((x[:,None]<sj[None,:]).mean()),'P_better_random':float((x[:,None]<rj[None,:]).mean())}
    out['evaluator_hash']='0dcde5089bd66deba991214558637fe906f6c2a52530619850e6c5bab0d684da'
    (ART/'opt10_scale_prospective_q4096.json').write_text(json.dumps(out,indent=2));print(json.dumps({g:v['stats'] for g,v in out['groups'].items()},indent=2),flush=True);print(json.dumps(out['tests'],indent=2),flush=True)
if __name__=='__main__':main()
