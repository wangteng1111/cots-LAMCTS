#!/usr/bin/env python3
"""Alpha Lense first workstation smoke: real Mandler prescriptions + authoritative Q4096.
This is an integration/meta-test, not the final real-vendor COTS experiment.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; VENV=Path('/var/lib/cots-lamcts/venv/bin/python')
if sys.prefix != str(VENV.parent.parent) and VENV.exists(): os.execv(str(VENV),[str(VENV),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ.setdefault(k,'1')
import numpy as np, torch
from alpha_lense.core import Candidate,DesignSpec,EvalResult,AlphaLenseNet,AlphaLenseMCTS,better
from validation import optv1_lamcts_mandler_meta as BASE
from validation import opt10_workstation_pretrain_meta as B
from evaluator.q4096_cuda_pool import MultiGPUQ4096Pool,EXPECTED_CONFIG_HASH

ART=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'));ART.mkdir(parents=True,exist_ok=True)

class MandlerCatalog:
    def __init__(self):
        from core import mandler_benchmark_core as M
        a=np.asarray(M.SIGMAT,np.float32);self.sig=(a-a.mean(0))/(a.std(0)+1e-6)
    def legal_actions(self,state,spec):
        # Complete one-slot substitutions. Action = slot*72 + oriented component.
        out=[]
        for slot in range(4):
            for comp in range(72):
                if comp!=state.parts[slot]:out.append(slot*72+comp)
        return out
    def apply(self,state,action):
        slot,comp=divmod(int(action),72);p=list(state.parts);p[slot]=comp;return Candidate(tuple(p))
    def action_features(self,state,actions,spec):
        rows=[]
        for a in actions:
            slot,comp=divmod(int(a),72);d=self.sig[comp]-self.sig[state.parts[slot]]
            x=np.zeros(32,np.float32);n=min(24,len(d));x[:n]=d[:n];x[24+slot]=1.;x[28]=float(slot)/3.;x[29]=float(comp%2);x[30]=float(np.linalg.norm(d));x[31]=1.
            rows.append(x)
        return torch.from_numpy(np.stack(rows))

def main():
    seed=17;np.random.seed(seed);torch.manual_seed(seed)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    _,mu,sd=BASE.norm_moments()
    def tokenize(c):
        X,VM=BASE.seq_batch([c.parts],mu,sd);return torch.from_numpy(X[0]).float(),torch.from_numpy(VM[0]).bool()
    pool=MultiGPUQ4096Pool(gpus=(0,1),workers_per_gpu=2)
    calls=0
    def physics(c):
        nonlocal calls
        r=pool.evaluate_states4([c.parts],fixed_gap_indices=BASE.FIXED_GAPS)[0];calls+=1
        if r.get('config_hash')!=EXPECTED_CONFIG_HASH:raise RuntimeError('Q4096 hash mismatch')
        m=dict(r.get('metrics') or {})
        # Current evaluator names.
        metrics={'efl':m.get('efl'),'fno':m.get('fno'),'min_illum':m.get('min_illum'),'distortion_pct':100.*abs(m.get('dist_max',0.))}
        return EvalResult(float(r['J']),metrics)
    # Broad but real constraints for integration. This is deliberately NOT the final T3/80 design brief.
    spec=DesignSpec(efl_target_mm=80.,efl_tol_mm=30.,max_f_number=6.,min_relative_illumination=0.05,max_distortion_pct=20.)
    net=AlphaLenseNet().to(device).eval();cat=MandlerCatalog();search=AlphaLenseMCTS(net,cat,tokenize,physics,spec,c_puct=1.5,device=device)
    root=Candidate((0,2,4,6));t0=time.time();res=search.search(root,simulations=48,physics_budget=12,temperature=1.)
    # Explicitly test feasibility-first comparator with synthetic values too.
    good=EvalResult(10.,{'efl':80.},True,{'efl':1.},());bad=EvalResult(.001,{'efl':20.},False,{'efl':-58.},('efl',));assert better(good,bad)
    out={'schema':1,'name':'alpha-lense-q4096-smoke','device':device,'root':list(root.parts),'next_state':list(res.next_state.parts),'best_state':None if res.best_state is None else list(res.best_state.parts),'best_J':None if res.best_eval is None else res.best_eval.J_quality,'best_feasible':None if res.best_eval is None else res.best_eval.feasible,'best_margins':None if res.best_eval is None else res.best_eval.margins,'physics_evals':res.physics_evals,'physics_calls':calls,'policy_sum':float(res.policy.sum()),'root_visits':float(res.root.N.sum()),'elapsed_sec':time.time()-t0,'constraints':'broad integration only; not final T3/80 brief','q4096_config_hash':EXPECTED_CONFIG_HASH}
    (ART/'alpha_lense_smoke.json').write_text(json.dumps(out,indent=2)+'\n');print('ALPHA_LENSE_SMOKE',json.dumps(out),flush=True);pool.close()
if __name__=='__main__':main()
