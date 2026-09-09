#!/usr/bin/env python3
from __future__ import annotations

import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k,'1')

import csv
import gzip
import importlib.util
import json
import math
import multiprocessing as mp
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr, kendalltau, mannwhitneyu

REPO = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get('COTS_JOB_ARTIFACT_DIR', '.'))
ART.mkdir(parents=True, exist_ok=True)
DATA_GZ = REPO / 'data' / 'meta' / 'opt10_compact_q4096_dataset.json.gz'
EVAL_PATH = REPO / 'evaluator' / 'market_final_direct_q4096_v4.py'
CORPUS_ROOT = Path('/var/lib/cots-lamcts/corpora/goptical')
GOPTICAL_URL = 'https://github.com/dibyendumajumdar/goptical.git'
EXPECTED_EVAL_HASH = '0dcde5089bd66deba991214558637fe906f6c2a52530619850e6c5bab0d684da'
GAP_GRID=((0.20,0.29,0.38,0.47,0.56),(7.50,8.32,9.14,9.96,10.78),(11.00,12.18,13.36,14.54,15.72),(0.20,0.29,0.38,0.47,0.56))
MAXLEN=64
FIELDS=8
SEEDS=(17,43,89,131,211)
POOL_N=200_000
TOPK=16
RANDOMK=16
EVAL_WORKERS=16


def load_module(path: Path, name: str):
    sp=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

E=load_module(EVAL_PATH,'opt10_eval_v4')
M=E.M
if E.CONFIG_HASH != EXPECTED_EVAL_HASH:
    raise RuntimeError(f'evaluator hash mismatch: {E.CONFIG_HASH}')
N_ORIENTED=2*len(M.CAT)


def run(cmd, cwd=None, timeout=1800, check=True):
    p=subprocess.run(cmd,cwd=str(cwd) if cwd else None,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout,check=False)
    print('+',' '.join(cmd),flush=True); print(p.stdout[-8000:],flush=True)
    if check and p.returncode!=0: raise RuntimeError(f'command failed {p.returncode}: {cmd}')
    return p


def ensure_corpus():
    CORPUS_ROOT.parent.mkdir(parents=True,exist_ok=True)
    if not (CORPUS_ROOT/'.git').exists():
        if CORPUS_ROOT.exists(): shutil.rmtree(CORPUS_ROOT)
        run(['git','clone','--depth','1',GOPTICAL_URL,str(CORPUS_ROOT)],timeout=1800)
    else:
        run(['git','fetch','--depth','1','origin','master'],cwd=CORPUS_ROOT,timeout=900,check=False)
        run(['git','reset','--hard','origin/master'],cwd=CORPUS_ROOT,timeout=120,check=False)


def fnum(x):
    try:
        if str(x).lower() in ('infinity','inf','+infinity'): return math.inf
        return float(str(x).replace(',',''))
    except Exception: return None


def parse_goptical_file(path: Path):
    text=path.read_text(errors='replace').splitlines(); sec=''; title=path.stem; focal=None; variables={}; aspheres=set(); raw_lens=[]
    for line in text:
        s=line.strip()
        if not s: continue
        if s.startswith('[') and s.endswith(']'):
            sec=s.lower(); continue
        cols=[c.strip() for c in line.split('\t')]
        if sec=='[descriptive data]' and len(cols)>=2 and cols[0].lower()=='title': title=cols[1]
        elif sec=='[constants]' and len(cols)>=2 and cols[0].lower()=='focal length':
            for c in cols[1:]:
                v=fnum(c)
                if v is not None and math.isfinite(v) and abs(v)>1e-6: focal=abs(v); break
        elif sec=='[variable distances]' and len(cols)>=2:
            for c in cols[1:]:
                v=fnum(c)
                if v is not None and math.isfinite(v): variables[cols[0]]=v; break
        elif sec=='[lens data]': raw_lens.append(cols)
        elif sec=='[aspherical data]' and cols:
            try: aspheres.add(int(cols[0]))
            except Exception: pass
    if focal is None or not (3.0 <= focal <= 2000.0): return None
    seq=[]; z=0.0; numeric_surfaces=0
    for cols in raw_lens:
        if len(cols)<3: continue
        try: no=int(cols[0])
        except Exception: continue
        rtok=cols[1]
        thtok=cols[2]
        th=fnum(thtok)
        if th is None: th=variables.get(thtok)
        if th is None or not math.isfinite(th) or abs(th)>5000: th=0.0
        stop=1.0 if rtok.upper() in ('AS','FS','STOP') else 0.0
        if stop:
            seq.append([0.0,th/focal,0.0,0.0,z/focal,1.0,0.0,0.0]); z+=th; continue
        if rtok.upper() in ('CG','PLANE'): R=math.inf
        else:
            R=fnum(rtok)
            if R is None: z+=th; continue
        nd=fnum(cols[3]) if len(cols)>3 and cols[3] else None
        vd=fnum(cols[4]) if len(cols)>4 and cols[4] else None
        curv=0.0 if not math.isfinite(R) or abs(R)<1e-12 else float(np.clip(focal/R,-8,8))
        nfeat=0.0 if nd is None else float(nd-1.0)
        vfeat=0.0 if vd is None or vd<=0 else float(np.clip(50.0/vd,0,4))
        seq.append([curv,th/focal,nfeat,vfeat,z/focal,0.0,float(nd is not None),float(no in aspheres)])
        z+=th; numeric_surfaces+=1
    if numeric_surfaces<4 or len(seq)<4: return None
    a=np.asarray(seq[:MAXLEN],np.float32)
    if not np.all(np.isfinite(a)): return None
    return {'name':title,'path':str(path.relative_to(CORPUS_ROOT)),'focal':focal,'seq':a}


def load_human_corpus():
    ensure_corpus(); out=[]
    for p in sorted((CORPUS_ROOT/'data').rglob('*.txt')):
        try:
            x=parse_goptical_file(p)
            if x is not None: out.append(x)
        except Exception as ex:
            print('parse skip',p,ex,flush=True)
    # exact duplicate sequence removal
    seen=set(); uniq=[]
    for x in out:
        key=x['seq'].tobytes()
        if key not in seen: seen.add(key); uniq.append(x)
    return uniq


def decode_state(state):
    state=tuple(map(int,state)); design=tuple((state[i]//2,bool(state[i]%2)) for i in range(4)); gi=state[4:8]
    gaps=tuple(float(GAP_GRID[j][gi[j]]) for j in range(4)); return design,gaps


def _curv(R,f):
    if R is None or not math.isfinite(float(R)) or abs(float(R))<1e-9:return 0.0
    return float(np.clip(f/float(R),-8,8))

def _vfeat(v): return 0.0 if v is None or float(v)<=0 else float(np.clip(50.0/float(v),0,4))


def cots_seq(state):
    design,gaps=decode_state(state); comps=[M.CAT[i].oriented(fl) for i,fl in design]; g12,g2s,gs3,g34=gaps; L=[sum(c.th) for c in comps]
    z2=-g2s-L[1]; z1=z2-g12-L[0]; z3=gs3; z4=z3+L[2]+g34; starts=[z1,z2,z3,z4]; f=100.0; entries=[]
    for c,z0 in zip(comps,starts):
        z=z0
        for j,R in enumerate(c.radii):
            if j < len(c.ne): n=float(c.ne[j]); v=float(c.ve[j])
            else: n=1.0; v=None
            entries.append({'z':z,'R':R,'n':n,'v':v,'stop':0.0,'asp':0.0})
            if j < len(c.th): z += float(c.th[j])
    entries.append({'z':0.0,'R':None,'n':1.0,'v':None,'stop':1.0,'asp':0.0}); entries.sort(key=lambda q:(q['z'],q['stop']))
    seq=[]
    for i,e in enumerate(entries):
        dz=(entries[i+1]['z']-e['z']) if i+1<len(entries) else 0.0
        seq.append([_curv(e['R'],f),dz/f,e['n']-1.0,_vfeat(e['v']),e['z']/f,e['stop'],float(e['n']>1.0001),e['asp']])
    return np.asarray(seq[:MAXLEN],np.float32)


def pad(seqs):
    x=np.zeros((len(seqs),MAXLEN,FIELDS),np.float32); valid=np.zeros((len(seqs),MAXLEN),bool)
    for i,s in enumerate(seqs): n=min(len(s),MAXLEN); x[i,:n]=s[:n]; valid[i,:n]=1
    return x,valid


class SurfaceTransformer(nn.Module):
    def __init__(self,d=64,heads=4,layers=3,ff=128):
        super().__init__(); self.d=d
        self.proj=nn.Sequential(nn.Linear(FIELDS,d),nn.LayerNorm(d),nn.GELU())
        self.mask_token=nn.Parameter(torch.randn(1,1,d)*.02); self.cls=nn.Parameter(torch.randn(1,1,d)*.02); self.pos=nn.Parameter(torch.randn(1,MAXLEN+1,d)*.02)
        el=nn.TransformerEncoderLayer(d,heads,ff,dropout=.06,activation='gelu',batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(el,layers,norm=nn.LayerNorm(d)); self.recon=nn.Linear(d,FIELDS)
        self.reg=nn.Sequential(nn.Linear(d,48),nn.GELU(),nn.Linear(48,1))
    def encode(self,x,valid,maskpos=None):
        h=self.proj(x)
        if maskpos is not None: h=torch.where(maskpos.unsqueeze(-1),self.mask_token.expand(h.shape[0],h.shape[1],-1),h)
        cls=self.cls.expand(x.shape[0],-1,-1); h=torch.cat([cls,h],1)+self.pos[:,:x.shape[1]+1]
        padmask=torch.cat([torch.zeros((x.shape[0],1),dtype=torch.bool,device=x.device),~valid],1)
        return self.enc(h,src_key_padding_mask=padmask)
    def forward_reg(self,x,valid): return self.reg(self.encode(x,valid)[:,0]).squeeze(-1)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pretrain(seed,hx,hm,device,epochs=250):
    seed_all(seed); model=SurfaceTransformer().to(device); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-3)
    bx=torch.from_numpy(hx).to(device); bm=torch.from_numpy(hm).to(device); rng=np.random.default_rng(seed+500); best=1e99; bestsd=None; views=32
    for ep in range(epochs):
        idx=rng.integers(0,len(hx),size=max(256,views*len(hx))); tx=bx[idx].clone(); tm=bm[idx].clone()
        noise=torch.from_numpy(rng.normal(0,.012,size=tx[:,:,:5].shape).astype(np.float32)).to(device); tx[:,:,:5]+=noise*tm.unsqueeze(-1)
        mp=torch.from_numpy(((rng.random(tm.shape)<.25)&tm.detach().cpu().numpy())).to(device)
        # guarantee at least one masked valid token
        empty=(mp.sum(1)==0).nonzero(as_tuple=False).flatten()
        if len(empty):
            for ii in empty.tolist():
                vv=tm[ii].nonzero(as_tuple=False).flatten(); mp[ii,vv[0]]=True
        model.train(); z=model.encode(tx,tm,mp)[:,1:]; pred=model.recon(z); loss=((pred-bx[idx])**2)[mp].mean()
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); lv=float(loss.detach().cpu())
        if lv<best: best=lv; bestsd={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep%50==0: print('pretrain',seed,ep,lv,flush=True)
    model.load_state_dict(bestsd); return model,best


def finetune(model,seed,Xtr,Mtr,ytr,Xte,Mte,device,epochs=500):
    seed_all(seed); d=model.d; model.reg=nn.Sequential(nn.Linear(d,48),nn.GELU(),nn.Linear(48,1)).to(device)
    ym=float(ytr.mean()); ys=float(ytr.std()+1e-6); z=(ytr-ym)/ys; rng=np.random.default_rng(seed+123); idx=np.arange(len(ytr)); rng.shuffle(idx); nv=max(28,int(.18*len(idx))); va=idx[:nv]; tr=idx[nv:]
    a=torch.from_numpy(Xtr).to(device); m=torch.from_numpy(Mtr).to(device); zz=torch.from_numpy(z).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=2e-3); lf=nn.SmoothL1Loss(beta=.45); best=1e99; bestsd=None; stale=0
    for ep in range(epochs):
        model.train(); perm=rng.permutation(tr)
        for p in range(0,len(perm),64):
            ix=torch.from_numpy(perm[p:p+64]).to(device); pred=model.forward_reg(a[ix],m[ix]); loss=lf(pred,zz[ix]); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(lf(model.forward_reg(a[va],m[va]),zz[va]).cpu())
        if vl<best-1e-4: best=vl; bestsd={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else:
            stale+=1
            if stale>=70: break
    model.load_state_dict(bestsd); model.to(device).eval();
    with torch.no_grad(): pred=model.forward_reg(torch.from_numpy(Xte).to(device),torch.from_numpy(Mte).to(device)).cpu().numpy()*ys+ym
    return model,pred,{'epochs':ep+1,'val':best,'ym':ym,'ys':ys}


def metrics(y,p,J):
    n=len(y); k=max(1,n//4); tt=set(np.argsort(y)[-k:]); pp=set(np.argsort(p)[-k:]); top=np.argsort(p)[-min(8,n):]
    return {'spearman':float(spearmanr(y,p).statistic),'kendall':float(kendalltau(y,p).statistic),'top_quartile_precision':len(tt&pp)/k,'pred_top8_mean_J':float(J[top].mean()),'pred_top8_best_J':float(J[top].min())}


def infer_states(models,states,mu,sd,device,batch=4096):
    vals=[]
    for p in range(0,len(states),batch):
        ss=states[p:p+batch]; seqs=[]
        for st in ss:
            q=cots_seq(st); q=q.copy(); q[:,:5]=(q[:,:5]-mu)/sd; seqs.append(q)
        x,m=pad(seqs); tx=torch.from_numpy(x).to(device); tm=torch.from_numpy(m).to(device); ps=[]
        with torch.no_grad():
            for model,ym,ys in models: ps.append((model.forward_reg(tx,tm).cpu().numpy()*ys+ym))
        vals.append(np.mean(ps,axis=0))
    return np.concatenate(vals)


def random_states(n,seed,exclude):
    rng=np.random.default_rng(seed); out=[]; seen=set(exclude)
    while len(out)<n:
        chunk=np.column_stack([rng.integers(0,N_ORIENTED,size=8192) for _ in range(4)]+[rng.integers(0,5,size=8192) for _ in range(4)])
        for r in chunk:
            t=tuple(map(int,r))
            if t not in seen: seen.add(t); out.append(t)
            if len(out)>=n: break
    return np.asarray(out,dtype=np.int16)


def eval_state_worker(state):
    # spawned worker: import authoritative evaluator in a clean process
    repo=Path(__file__).resolve().parents[1]; ev=load_module(repo/'evaluator'/'market_final_direct_q4096_v4.py',f'worker_eval_{os.getpid()}')
    st=tuple(map(int,state)); design=tuple((st[i]//2,bool(st[i]%2)) for i in range(4)); gi=st[4:8]; gaps=tuple(float(GAP_GRID[j][gi[j]]) for j in range(4))
    r=ev.evaluate_final_with_gaps(design,gaps,qmc_samples=ev.QMC_SAMPLES)
    return {'state':list(st),'J':float(r['merit_J']),'score':float(r['score']),'quality_score_100':float(r['quality_score_100']),'mtf_geomean_reg':float(r['mtf_geomean_reg']),'mtf_p10':float(r['mtf_p10']),'min_illum':float(r['min_illum']),'dist_max':float(r['dist_max']),'efl':float(r['efl']),'fno':float(r['fno']),'config_hash':r['config_hash']}


def main():
    t0=time.time(); print('torch',torch.__version__,'cuda',torch.version.cuda,'devices',torch.cuda.device_count(),flush=True)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    device=torch.device('cuda:0')
    human=load_human_corpus(); print('parsed human prescriptions',len(human),flush=True)
    if len(human)<10: raise RuntimeError('too few parsed human prescriptions')
    with gzip.open(DATA_GZ,'rt') as f: ds=json.load(f)
    if ds['evaluator_config_hash']!=EXPECTED_EVAL_HASH: raise RuntimeError('dataset evaluator hash mismatch')
    train=ds['train_192']; ext=ds['heldout_24']+ds['prospective_32']
    # dedupe external
    seen=set(); ext2=[]
    for r in ext:
        k=tuple(r['state'])
        if k not in seen: seen.add(k); ext2.append(r)
    ext=ext2
    hseq=[x['seq'] for x in human]; trseq=[cots_seq(r['state']) for r in train]; teseq=[cots_seq(r['state']) for r in ext]
    stats=np.concatenate(hseq+trseq,axis=0); mu=stats[:,:5].mean(0); sd=stats[:,:5].std(0)+1e-5
    def norm(s): q=s.copy(); q[:,:5]=(q[:,:5]-mu)/sd; return q
    hx,hm=pad([norm(s) for s in hseq]); Xtr,Mtr=pad([norm(s) for s in trseq]); Xte,Mte=pad([norm(s) for s in teseq])
    ytr=np.asarray([-math.log(float(r['J'])+1e-9) for r in train],np.float32); yte=np.asarray([-math.log(float(r['J'])+1e-9) for r in ext],np.float32); Jte=np.asarray([float(r['J']) for r in ext])
    models={'scratch':[],'pretrained':[]}; offline={}
    for kind in ('scratch','pretrained'):
        preds=[]; infos=[]
        for si,seed in enumerate(SEEDS):
            if kind=='pretrained': model,pl=pretrain(seed,hx,hm,device)
            else: seed_all(seed); model=SurfaceTransformer().to(device); pl=None
            model,p,fi=finetune(model,seed,Xtr,Mtr,ytr,Xte,Mte,device); preds.append(p); infos.append({'seed':seed,'pretrain_loss':pl,'finetune':fi}); models[kind].append((model,fi['ym'],fi['ys']))
            print(kind,seed,metrics(yte,p,Jte),flush=True)
        ens=np.mean(preds,axis=0); offline[kind]={'metrics':metrics(yte,ens,Jte),'individual':[metrics(yte,p,Jte) for p in preds],'training':infos,'predictions':ens.tolist()}
    # paired bootstrap on fixed external test
    rng=np.random.default_rng(20260910); dif=[]; ps=np.asarray(offline['pretrained']['predictions']); ss=np.asarray(offline['scratch']['predictions'])
    for _ in range(10000):
        ix=rng.integers(0,len(yte),size=len(yte)); a=spearmanr(yte[ix],ps[ix]).statistic; b=spearmanr(yte[ix],ss[ix]).statistic
        if np.isfinite(a) and np.isfinite(b): dif.append(a-b)
    bootstrap={'mean':float(np.mean(dif)),'p_delta_le_0':float(np.mean(np.asarray(dif)<=0)),'q025':float(np.quantile(dif,.025)),'q975':float(np.quantile(dif,.975))}
    # prospective global pool
    exclude={tuple(r['state']) for r in train+ext}; pool=random_states(POOL_N,2026091001,exclude)
    print('ranking pool',len(pool),flush=True)
    pred_s=infer_states(models['scratch'],pool,mu,sd,device); pred_p=infer_states(models['pretrained'],pool,mu,sd,device)
    top_s=np.argsort(pred_s)[-TOPK:][::-1]; top_p=np.argsort(pred_p)[-TOPK:][::-1]
    rng=np.random.default_rng(2026091002); ridx=rng.choice(len(pool),size=RANDOMK,replace=False)
    groups={'scratch':[tuple(map(int,pool[i])) for i in top_s],'opt10':[tuple(map(int,pool[i])) for i in top_p],'random':[tuple(map(int,pool[i])) for i in ridx]}
    union=[]; useen=set()
    for g in groups.values():
        for st in g:
            if st not in useen: useen.add(st); union.append(st)
    print('Q4096 prospective unique',len(union),'overlap scratch/opt',len(set(groups['scratch'])&set(groups['opt10'])),flush=True)
    results={}; ctx=mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=EVAL_WORKERS,mp_context=ctx) as ex:
        futs={ex.submit(eval_state_worker,st):st for st in union}
        for fut in as_completed(futs):
            r=fut.result(); results[tuple(r['state'])]=r; print('Q4096',r['state'],'J',r['J'],flush=True)
    prospective={}
    for name,sts in groups.items():
        rows=[results[st] for st in sts]; arr=np.asarray([r['J'] for r in rows]); prospective[name]={'states':[list(s) for s in sts],'rows':rows,'mean_J':float(arr.mean()),'median_J':float(np.median(arr)),'best_J':float(arr.min()),'worst_J':float(arr.max())}
    a=np.asarray([r['J'] for r in prospective['opt10']['rows']]); b=np.asarray([r['J'] for r in prospective['scratch']['rows']]); c=np.asarray([r['J'] for r in prospective['random']['rows']])
    prospective['stats']={'opt10_vs_scratch_mwu_less_p':float(mannwhitneyu(a,b,alternative='less').pvalue),'opt10_vs_random_mwu_less_p':float(mannwhitneyu(a,c,alternative='less').pvalue),'scratch_vs_random_mwu_less_p':float(mannwhitneyu(b,c,alternative='less').pvalue)}
    summary={'schema':'OPT1.0-workstation-meta-v1','git_commit':os.environ.get('COTS_GIT_COMMIT'),'evaluator_hash':E.CONFIG_HASH,'human_corpus':{'source':GOPTICAL_URL,'parsed_prescriptions':len(human),'names':[x['name'] for x in human]},'train_q4096':len(train),'external_test':len(ext),'model':{'d':64,'heads':4,'layers':3,'ff':128,'ensemble_seeds':list(SEEDS)},'offline':offline,'paired_bootstrap_delta_spearman':bootstrap,'prospective':prospective,'pool_n':POOL_N,'elapsed_sec':time.time()-t0,'cuda':{'torch':torch.__version__,'cuda':torch.version.cuda,'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}}
    (ART/'opt10_workstation_meta_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (ART/'opt10_workstation_prospective.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['group','state','J','mtf_geomean_reg','mtf_p10','min_illum','dist_max','efl','fno'])
        for name in ('scratch','opt10','random'):
            for r in prospective[name]['rows']: w.writerow([name,' '.join(map(str,r['state'])),r['J'],r['mtf_geomean_reg'],r['mtf_p10'],r['min_illum'],r['dist_max'],r['efl'],r['fno']])
    print(json.dumps({'human':len(human),'offline':{k:v['metrics'] for k,v in offline.items()},'bootstrap':bootstrap,'prospective':{k:{kk:vv for kk,vv in v.items() if kk in ('mean_J','median_J','best_J','worst_J')} if isinstance(v,dict) else v for k,v in prospective.items()},'elapsed_sec':summary['elapsed_sec']},indent=2),flush=True)

if __name__=='__main__': main()
