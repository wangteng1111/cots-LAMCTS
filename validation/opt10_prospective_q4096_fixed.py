#!/usr/bin/env python3
from __future__ import annotations
import os,sys,json,subprocess,time
from pathlib import Path
VENV_ROOT=Path('/var/lib/cots-lamcts/venv'); VENV=VENV_ROOT/'bin/python'
if Path(sys.prefix).resolve()!=VENV_ROOT.resolve() and VENV.exists():
    os.execv(str(VENV),[str(VENV),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
ART=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts')); ART.mkdir(parents=True,exist_ok=True)
PROPOSALS={
'scratch':[[28,29,16,6,2,0,4,3],[28,53,40,6,3,2,4,3],[44,21,15,41,1,4,4,4],[36,29,48,6,1,1,3,0],[52,69,62,46,1,3,4,3],[52,5,30,15,0,2,4,4],[52,37,71,22,4,3,3,1],[28,53,48,1,3,0,3,0]],
'pretrained':[[52,61,32,54,0,3,1,1],[4,61,33,39,0,2,1,2],[36,53,32,38,3,3,4,4],[28,26,62,38,0,1,1,4],[12,10,41,38,4,3,1,2],[20,61,63,33,0,1,0,0],[20,53,41,56,3,0,0,3],[60,26,30,62,4,2,1,0]],
'random':[[60,37,64,9,2,3,3,1],[38,4,40,27,4,0,3,1],[31,38,66,26,1,3,4,4],[14,59,1,49,4,4,1,4],[44,38,62,15,1,3,3,4],[71,69,27,7,0,0,3,1],[60,54,71,26,2,2,0,3],[54,59,57,30,3,2,0,2]]}
GRIDS=[(.20,.29,.38,.47,.56),(7.50,8.32,9.14,9.96,10.78),(11.0,12.18,13.36,14.54,15.72),(.20,.29,.38,.47,.56)]
def decode(st): return tuple((int(x)//2,bool(int(x)%2)) for x in st[:4]),tuple(GRIDS[i][int(st[4+i])] for i in range(4))
def child(st,out):
    from evaluator import market_final_direct_q4096_v4 as E
    d,g=decode(st); r=E.evaluate_final_with_gaps(d,g,4096)
    q={'state':st,'J':float(r['merit_J']),'score':float(r['score']),'config_hash':r['config_hash'],'efl':r['efl'],'fno':r['fno'],'mtf_mean':r['mtf_mean'],'mtf_geomean_reg':r['mtf_geomean_reg'],'mtf_p10':r['mtf_p10'],'min_illum':r['min_illum'],'dist_max':r['dist_max'],'lca':r['lca'],'costs':r['costs']}
    Path(out).write_text(json.dumps(q)); return 0
def main():
    import numpy as np
    from scipy.stats import mannwhitneyu
    from concurrent.futures import ThreadPoolExecutor,as_completed
    jobs=[(k,i,st) for k,arr in PROPOSALS.items() for i,st in enumerate(arr)]; tmp=ART/'eval';tmp.mkdir(exist_ok=True)
    def one(k,i,st):
        out=tmp/f'{k}_{i}.json'; cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--child',json.dumps(st),'--out',str(out)]; last=''
        for a in range(3):
            try:
                p=subprocess.run(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1'})
                last=p.stdout[-4000:]
                if p.returncode==0 and out.exists(): return k,i,json.loads(out.read_text())
            except subprocess.TimeoutExpired: last='timeout'
        return k,i,{'state':st,'error':last}
    res={k:[None]*8 for k in PROPOSALS}
    with ThreadPoolExecutor(max_workers=8) as ex:
        fs=[ex.submit(one,*j) for j in jobs]
        for f in as_completed(fs):
            k,i,r=f.result();res[k][i]=r;print(k,i,r.get('J',r.get('error')),flush=True)
    def stats(rows):
        js=[r['J'] for r in rows if r and 'J'in r]
        return {'n':len(js),'mean_J':float(np.mean(js)) if js else None,'median_J':float(np.median(js)) if js else None,'best_J':float(np.min(js)) if js else None,'worst_J':float(np.max(js)) if js else None}
    out={'schema':'OPT1.0-workstation-prospective-q4096-v1','evaluator_hash':'0dcde5089bd66deba991214558637fe906f6c2a52530619850e6c5bab0d684da','groups':{k:{'stats':stats(v),'rows':v} for k,v in res.items()}}
    if all(out['groups'][k]['stats']['n']==8 for k in res):
        s=[r['J'] for r in res['scratch']];p=[r['J'] for r in res['pretrained']];rr=[r['J'] for r in res['random']]
        out['tests']={'pretrained_vs_scratch_mwu_less':float(mannwhitneyu(p,s,alternative='less').pvalue),'pretrained_vs_random_mwu_less':float(mannwhitneyu(p,rr,alternative='less').pvalue),'scratch_vs_random_mwu_less':float(mannwhitneyu(s,rr,alternative='less').pvalue),'P_pretrained_better_scratch':float(np.mean([a<b for a in p for b in s])),'P_pretrained_better_random':float(np.mean([a<b for a in p for b in rr]))}
    (ART/'opt10_prospective_q4096_results.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v['stats'] for k,v in out['groups'].items()},indent=2));print(json.dumps(out.get('tests',{}),indent=2))
if __name__=='__main__':
    import argparse;ap=argparse.ArgumentParser();ap.add_argument('--child');ap.add_argument('--out');a=ap.parse_args()
    if a.child: raise SystemExit(child(json.loads(a.child),a.out))
    main()
