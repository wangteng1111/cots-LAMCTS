#!/usr/bin/env python3
from __future__ import annotations
import gzip, hashlib, json, math, os, random, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
VENV=Path('/var/lib/cots-lamcts/venv/bin/python')
if sys.prefix != str(VENV.parent.parent) and VENV.exists():
    os.execv(str(VENV),[str(VENV),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])

os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from scipy.stats import spearmanr
from validation import opt10_workstation_pretrain_meta as B

ART=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'));ART.mkdir(parents=True,exist_ok=True)
OB=Path('/var/lib/cots-lamcts/corpora/opticalbench_hub/parsed_lenses.json.gz')
CK=Path('/var/lib/cots-lamcts/checkpoints/opt10_scale_masked_v1');CK.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data/meta/opt10_compact_q4096_dataset.json.gz'
SEEDS=[17,43,89,131,173];BUDGETS=[48,96,192];TARGET_SCALES=[21,256,1024]
PRE_EPOCHS=300;VIEWS=24;PRE_BATCH=512
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');B.DEVICE=DEVICE


def ob_seq(L):
    f=float(L['f']);z=0.;seq=[];asph=set(map(str,L.get('asph',[])))
    for typ,R,thi,nd,vd,sid in L['rows']:
        thi=float(thi or 0)
        if typ=='FIELDSTOP': z+=thi;continue
        if typ=='STOP': seq.append([0.,thi/f,0.,0.,z/f,1.,0.,0.]);z+=thi;continue
        n=1. if nd is None else float(nd)
        seq.append([B._curv(R,f),thi/f,n-1.,B._vfeat(vd),z/f,0.,float(nd is not None),float(str(sid) in asph)])
        z+=thi
    return np.asarray(seq[:B.MAXLEN],np.float32)

def shash(s): return hashlib.sha1(np.round(s,5).tobytes()).hexdigest()

def load_corpora():
    if not OB.exists(): raise FileNotFoundError(f'bulk OpticalBench corpus missing: {OB}')
    B.ensure_corpus();small=[];hs=set()
    for p in sorted((B.CORPUS_ROOT/'data').rglob('*.txt')):
        try:
            L=B.parse_goptical_file(p)
            if not L:continue
            s=B.human_seq(L);h=shash(s)
            if h not in hs:hs.add(h);small.append({'name':'goptical:'+L['name'],'seq':s})
        except Exception:pass
    with gzip.open(OB,'rt',encoding='utf-8') as f: raw=json.load(f)
    extra=[]
    for L in raw:
        try:s=ob_seq(L)
        except Exception:continue
        if len(s)<4:continue
        h=shash(s)
        if h not in hs:hs.add(h);extra.append({'name':'optbench:'+L['filename'],'seq':s})
    rng=np.random.default_rng(20260910);rng.shuffle(extra)
    base=small[:21]
    full=base+extra
    scales=[]
    for n in TARGET_SCALES:
        nn=min(n,len(full)); scales.append((nn,full[:nn]))
    # de-duplicate scale labels if corpus smaller than target
    out=[];seen=set()
    for n,x in scales:
        if n not in seen:seen.add(n);out.append((n,x))
    return small,full,out

def pretrain_batched(seed,hx,hm,scale):
    ck=CK/f'pretrain_scale{scale}_seed{seed}.pt'
    if ck.exists():
        B.seedall(seed);m=B.SurfaceTransformer().to(DEVICE);m.load_state_dict(torch.load(ck,map_location=DEVICE,weights_only=True));return m,{'cached':True,'best_loss':None}
    B.seedall(seed);m=B.SurfaceTransformer().to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=2e-3)
    bx=torch.from_numpy(hx);bm=torch.from_numpy(hm);rng=np.random.default_rng(seed+500+scale);best=1e30;bestsd=None
    ns=max(128,len(hx)*VIEWS);steps=math.ceil(ns/PRE_BATCH);hist=[]
    for ep in range(PRE_EPOCHS):
        el=[]
        for step in range(steps):
            bsz=min(PRE_BATCH,ns-step*PRE_BATCH);sel=rng.integers(0,len(hx),size=bsz)
            tx=bx[sel].to(DEVICE,non_blocking=True);tm=bm[sel].to(DEVICE,non_blocking=True);xx=tx.clone()
            noise=torch.from_numpy(rng.normal(0,.018,size=xx[:,:,:5].shape).astype(np.float32)).to(DEVICE);xx[:,:,:5]+=noise*tm.unsqueeze(-1)
            mp_np=(rng.random(tm.shape)<.30)&tm.cpu().numpy()
            for i in range(len(mp_np)):
                if not mp_np[i].any():
                    ids=np.flatnonzero(tm[i].cpu().numpy())
                    if len(ids):mp_np[i,ids[0]]=True
            mp=torch.from_numpy(mp_np).to(DEVICE)
            m.train();z=m.encode(xx,tm,mp)[:,1:];pred=m.recon(z);loss=((pred-tx)**2)[mp].mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();el.append(float(loss.detach()))
        lv=float(np.mean(el));hist.append(lv)
        if lv<best:best=lv;bestsd={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if ep%50==0 or ep==PRE_EPOCHS-1:print('pretrain',scale,seed,'epoch',ep,'loss',lv,flush=True)
    m.load_state_dict(bestsd);torch.save({k:v.cpu() for k,v in m.state_dict().items()},ck)
    return m,{'cached':False,'best_loss':best,'final_loss':hist[-1],'epochs':PRE_EPOCHS,'views_per_lens':VIEWS,'batch':PRE_BATCH}

def clone_from_state(sd):
    m=B.SurfaceTransformer().to(DEVICE);m.load_state_dict(sd);return m

def bootstrap_delta(yt,a,b):
    rng=np.random.default_rng(20260910);ds=[]
    for _ in range(10000):
        ix=rng.integers(0,len(yt),len(yt));ra=spearmanr(yt[ix],a[ix]).statistic;rb=spearmanr(yt[ix],b[ix]).statistic
        if np.isfinite(ra) and np.isfinite(rb):ds.append(ra-rb)
    z=np.asarray(ds);return {'mean_delta_spearman':float(z.mean()),'q025':float(np.quantile(z,.025)),'q975':float(np.quantile(z,.975)),'p_delta_le_0':float(np.mean(z<=0))}

def main():
    t0=time.time();small,full,scales=load_corpora();print('corpus small/full/scales',len(small),len(full),[x[0] for x in scales],flush=True)
    with gzip.open(DATA,'rt') as f:ds=json.load(f)
    train=ds['train_192'];test=ds['heldout_24']+ds['prospective_32']
    trraw=[B.cots_seq(r['state']) for r in train];teraw=[B.cots_seq(r['state']) for r in test]
    # One common normalization for every arm. Human corpus moments are included once, as in OPT1.0 v1,
    # but are identical for scratch and all pretraining scales, so scale is the only changing factor.
    fullseq=[x['seq'] for x in full];stat=np.concatenate(fullseq+trraw,axis=0);mu=stat[:,:5].mean(0);sd=stat[:,:5].std(0)+1e-5
    trseq=[B.normseq(s,mu,sd) for s in trraw];teseq=[B.normseq(s,mu,sd) for s in teraw]
    X,M=B.pad(trseq);XT,MT=B.pad(teseq);y=np.asarray([-math.log(float(r['J'])+1e-9) for r in train],np.float32);yt=np.asarray([-math.log(float(r['J'])+1e-9) for r in test],np.float32)
    result={'schema':'OPT1.0-pretrain-scale-meta-v1','device':str(DEVICE),'torch':torch.__version__,'available_unique_corpus':len(full),'scales':[n for n,_ in scales],'seeds':SEEDS,'budgets':BUDGETS,'pretrain_epochs':PRE_EPOCHS,'views_per_lens':VIEWS,'models':{},'normalization':'common full-human+q4096-train moments for every arm'}
    final_models={};scratch_preds={}
    # scratch baseline once
    result['models']['scratch']={};final_models['scratch']=[]
    for budget in BUDGETS:
        pp=[];infos=[]
        for seed in SEEDS:
            B.seedall(seed);m=B.SurfaceTransformer().to(DEVICE);m,p,fi=B.finetune(m,seed,X,M,y,budget,XT,MT);pp.append(p);infos.append({'seed':seed,'finetune':fi})
            if budget==192:final_models['scratch'].append(m)
        ens=np.mean(pp,0);scratch_preds[budget]=ens;result['models']['scratch'][str(budget)]={'ensemble':B.met(yt,ens),'individual':[B.met(yt,p) for p in pp],'training':infos,'predictions':ens.tolist()}
        print('scratch',budget,result['models']['scratch'][str(budget)]['ensemble'],flush=True)
    # pretraining scales
    for scale,items in scales:
        key=f'pretrain_{scale}';result['models'][key]={};final_models[key]=[]
        hseq=[B.normseq(x['seq'],mu,sd) for x in items];HX,HM=B.pad(hseq)
        pstates={};pinfo={}
        for seed in SEEDS:
            pm,pi=pretrain_batched(seed,HX,HM,scale);pstates[seed]={k:v.detach().cpu().clone() for k,v in pm.state_dict().items()};pinfo[seed]=pi;del pm;torch.cuda.empty_cache()
        for budget in BUDGETS:
            pp=[];infos=[]
            for seed in SEEDS:
                m=clone_from_state(pstates[seed]);m,p,fi=B.finetune(m,seed,X,M,y,budget,XT,MT);pp.append(p);infos.append({'seed':seed,'pretrain':pinfo[seed],'finetune':fi})
                if budget==192:final_models[key].append(m)
            ens=np.mean(pp,0);result['models'][key][str(budget)]={'ensemble':B.met(yt,ens),'individual':[B.met(yt,p) for p in pp],'training':infos,'predictions':ens.tolist(),'bootstrap_vs_scratch':bootstrap_delta(yt,ens,scratch_preds[budget])}
            print(key,budget,result['models'][key][str(budget)]['ensemble'],result['models'][key][str(budget)]['bootstrap_vs_scratch'],flush=True)
    # same fresh 100k global pool for every arm
    known={tuple(map(int,r['state'])) for r in train+test};pool=B.pool_states(100000,known,seed=202609101);pseq=[B.cots_seq(st) for st in pool]
    selected={}
    for key,mods in final_models.items():
        pr=B.predict_pool(mods,pseq,mu,sd,batch=4096);idx=list(np.argsort(pr)[-8:][::-1]);selected[key]=[pool[i] for i in idx]
    rr=np.random.default_rng(202609102);idx=list(rr.choice(len(pool),8,replace=False));selected['random']=[pool[i] for i in idx]
    result['proposals']={k:[list(x) for x in v] for k,v in selected.items()}
    result['elapsed_sec']=time.time()-t0;result['evaluator_hash']='0dcde5089bd66deba991214558637fe906f6c2a52530619850e6c5bab0d684da'
    (ART/'opt10_pretrain_scale_meta.json').write_text(json.dumps(result,indent=2))
    (ART/'opt10_pretrain_scale_proposals.json').write_text(json.dumps(result['proposals'],indent=2))
    print(json.dumps({'available_unique_corpus':len(full),'scales':[n for n,_ in scales],'external192':{k:v['192']['ensemble'] for k,v in result['models'].items()},'elapsed_sec':result['elapsed_sec']},indent=2),flush=True)
if __name__=='__main__':main()
