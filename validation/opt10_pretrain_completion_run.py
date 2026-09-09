#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures as cf, gzip, hashlib, json, math, os, random, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
VENV=Path('/var/lib/cots-lamcts/venv/bin/python')
if sys.prefix != str(VENV.parent.parent) and VENV.exists():
    os.execv(str(VENV),[str(VENV),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('NUMBA_NUM_THREADS','1')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import spearmanr, mannwhitneyu
from validation import opt10_workstation_pretrain_meta as B
from validation import opt10_pretrain_scale_meta as S
ART=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'));ART.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data/meta/opt10_compact_q4096_dataset.json.gz'
FIRST=Path('/var/lib/cots-lamcts/corpora/opticalbench_hub/parsed_lenses.json.gz')
MULTI=Path('/var/lib/cots-lamcts/corpora/opticalbench_hub/parsed_multistate.json.gz')
CK=Path('/var/lib/cots-lamcts/checkpoints/opt10_completion_v1');CK.mkdir(parents=True,exist_ok=True)
SEEDS=[17,43,89,131,173]; BUDGETS=[48,96,192]; EPOCHS=300; VIEWS=24; PRE_BATCH=384
BASE=dict(d=64,heads=4,layers=3,ff=128); MED=dict(d=96,heads=4,layers=4,ff=192)
VARIANTS={
 'scratch_base':dict(pretrain=False,arch=BASE),
 'scratch_medium':dict(pretrain=False,arch=MED),
 'mlm_first_base':dict(pretrain=True,arch=BASE,corpus='first',order=False,contrast=False,family=False),
 'mlm_multi_base':dict(pretrain=True,arch=BASE,corpus='multi',order=False,contrast=False,family=False),
 'mlm_order_base':dict(pretrain=True,arch=BASE,corpus='multi',order=True,contrast=False,family=False),
 'mlm_contrast_base':dict(pretrain=True,arch=BASE,corpus='multi',order=False,contrast=True,family=False),
 'mlm_all_base':dict(pretrain=True,arch=BASE,corpus='multi',order=True,contrast=True,family=True),
 'mlm_all_medium':dict(pretrain=True,arch=MED,corpus='multi',order=True,contrast=True,family=True),
}

class PT(nn.Module):
    def __init__(self,d=64,heads=4,layers=3,ff=128):
        super().__init__();self.d=d
        self.proj=nn.Sequential(nn.Linear(B.FIELDS,d),nn.LayerNorm(d),nn.GELU())
        self.mask_token=nn.Parameter(torch.randn(1,1,d)*.02);self.cls=nn.Parameter(torch.randn(1,1,d)*.02);self.pos=nn.Parameter(torch.randn(1,B.MAXLEN+1,d)*.02)
        el=nn.TransformerEncoderLayer(d,heads,ff,dropout=.08,activation='gelu',batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(el,layers,norm=nn.LayerNorm(d));self.recon=nn.Linear(d,B.FIELDS)
        self.reg=nn.Sequential(nn.Linear(d,48),nn.GELU(),nn.Linear(48,1));self.order_head=nn.Linear(d,1);self.ctr=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,32))
    def encode(self,x,valid,maskpos=None):
        h=self.proj(x)
        if maskpos is not None:h=torch.where(maskpos.unsqueeze(-1),self.mask_token.expand(h.shape[0],h.shape[1],-1),h)
        h=torch.cat([self.cls.expand(x.shape[0],-1,-1),h],1)+self.pos[:,:x.shape[1]+1]
        pm=torch.cat([torch.zeros((x.shape[0],1),dtype=torch.bool,device=x.device),~valid],1)
        return self.enc(h,src_key_padding_mask=pm)
    def forward_reg(self,x,valid):return self.reg(self.encode(x,valid)[:,0]).squeeze(-1)

def model_for(v):return PT(**VARIANTS[v]['arch'])
def shash(s):return hashlib.sha1(np.round(s,6).tobytes()).hexdigest()
def load_lens_file(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
def seq_items(path):
    items=[];seen=set()
    for L in load_lens_file(path):
        try:s=S.ob_seq(L)
        except:continue
        if len(s)<4:continue
        h=shash(s)
        if h in seen:continue
        seen.add(h);items.append({'seq':s,'family':L.get('filename',L.get('name','?')),'state':int(L.get('state_idx',0))})
    return items

def common_data():
    first=seq_items(FIRST);multi=seq_items(MULTI)
    with gzip.open(DATA,'rt') as f:ds=json.load(f)
    train=ds['train_192'];test=ds['heldout_24']+ds['prospective_32']
    trraw=[B.cots_seq(r['state']) for r in train];teraw=[B.cots_seq(r['state']) for r in test]
    stat=np.concatenate([x['seq'] for x in multi]+trraw,axis=0);mu=stat[:,:5].mean(0);sd=stat[:,:5].std(0)+1e-5
    def norm_items(xs):return [{'seq':B.normseq(x['seq'],mu,sd),'family':x['family'],'state':x['state']} for x in xs]
    first=norm_items(first);multi=norm_items(multi);tr=[B.normseq(s,mu,sd) for s in trraw];te=[B.normseq(s,mu,sd) for s in teraw]
    X,M=B.pad(tr);XT,MT=B.pad(te);y=np.asarray([-math.log(float(r['J'])+1e-9) for r in train],np.float32);yt=np.asarray([-math.log(float(r['J'])+1e-9) for r in test],np.float32)
    return first,multi,train,test,mu,sd,X,M,XT,MT,y,yt

def pad_items(items):return B.pad([x['seq'] for x in items])
def seedall(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def make_mask(tm,rng,p=.30):
    a=(rng.random(tm.shape)<p)&tm.cpu().numpy()
    for i in range(len(a)):
        if not a[i].any():
            ids=np.flatnonzero(tm[i].cpu().numpy())
            if len(ids):a[i,ids[0]]=True
    return torch.from_numpy(a)
def noisy_view(tx,tm,rng):
    x=tx.clone();n=torch.from_numpy(rng.normal(0,.018,size=x[:,:,:5].shape).astype(np.float32));x[:,:,:5]+=n*tm.unsqueeze(-1);return x

def order_corrupt(tx,tm,rng):
    x=tx.clone();lab=np.ones(len(x),np.float32)
    for i in range(len(x)):
        ids=np.flatnonzero(tm[i].numpy())
        if len(ids)>2 and rng.random()<.5:
            j=int(rng.integers(0,len(ids)-1));a,b=int(ids[j]),int(ids[j+1]);tmp=x[i,a].clone();x[i,a]=x[i,b];x[i,b]=tmp;lab[i]=0.
    return x,torch.from_numpy(lab)

def family_maps(items):
    d={}
    for i,x in enumerate(items):d.setdefault(x['family'],[]).append(i)
    return d

def pretrain_one(variant,seed,gpu):
    cfg=VARIANTS[variant];assert cfg['pretrain'];device=torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu');seedall(seed)
    first,multi,*_=common_data();items=first if cfg['corpus']=='first' else multi;HX,HM=pad_items(items);fmap=family_maps(items)
    m=model_for(variant).to(device);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=2e-3);rng=np.random.default_rng(seed+1009)
    bx=torch.from_numpy(HX);bm=torch.from_numpy(HM);ns=max(256,len(items)*VIEWS);steps=math.ceil(ns/PRE_BATCH);best=1e30;bestsd=None;hist=[]
    for ep in range(EPOCHS):
        losses=[]
        for st in range(steps):
            bs=min(PRE_BATCH,ns-st*PRE_BATCH);sel=rng.integers(0,len(items),size=bs);tx=bx[sel];tm=bm[sel]
            v1=noisy_view(tx,tm,rng);mp1=make_mask(tm,rng);h1=m.encode(v1.to(device),tm.to(device),mp1.to(device));pred=m.recon(h1[:,1:]);loss=((pred-tx.to(device))**2)[mp1.to(device)].mean()
            if cfg.get('order'):
                ox,olab=order_corrupt(tx,tm,rng);ho=m.encode(ox.to(device),tm.to(device));olog=m.order_head(ho[:,0]).squeeze(-1);loss=loss+.20*F.binary_cross_entropy_with_logits(olog,olab.to(device))
            if cfg.get('contrast'):
                if cfg.get('family'):
                    sel2=[]
                    for idx in sel:
                        fam=items[int(idx)]['family'];cand=fmap[fam]
                        sel2.append(int(cand[int(rng.integers(0,len(cand)))]) if len(cand)>1 else int(idx))
                    t2=bx[np.asarray(sel2)];m2=bm[np.asarray(sel2)]
                else:t2=tx;m2=tm
                v2=noisy_view(t2,m2,rng);mp2=make_mask(m2,rng);h2=m.encode(v2.to(device),m2.to(device),mp2.to(device));z1=F.normalize(m.ctr(h1[:,0]),dim=-1);z2=F.normalize(m.ctr(h2[:,0]),dim=-1);log=z1@z2.T/.12;lab=torch.arange(len(z1),device=device);cl=.5*(F.cross_entropy(log,lab)+F.cross_entropy(log.T,lab));loss=loss+.12*cl
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        lv=float(np.mean(losses));hist.append(lv)
        if lv<best:best=lv;bestsd={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if ep%50==0 or ep==EPOCHS-1:print('PRE',variant,seed,'gpu',gpu,'ep',ep,'loss',lv,flush=True)
    m.load_state_dict(bestsd);ck=CK/f'{variant}_seed{seed}.pt';torch.save({k:v.cpu() for k,v in m.state_dict().items()},ck);info={'variant':variant,'seed':seed,'gpu':gpu,'corpus':cfg['corpus'],'corpus_n':len(items),'best_loss':best,'final_loss':hist[-1],'epochs':EPOCHS,'views':VIEWS};(CK/f'{variant}_seed{seed}.json').write_text(json.dumps(info));print(json.dumps(info),flush=True)

def launch_pretrains():
    jobs=[]
    for v,c in VARIANTS.items():
        if not c['pretrain']:continue
        for s in SEEDS:
            if (CK/f'{v}_seed{s}.pt').exists():continue
            jobs.append((v,s))
    def run(j):
        v,s=j;gpu=(abs(hash(v))+SEEDS.index(s))%2;cmd=[str(VENV),'-u',str(Path(__file__).resolve()),'--pretrain-one',v,'--seed',str(s),'--gpu',str(gpu)];z=subprocess.run(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=43200);print(z.stdout[-4000:],flush=True);return v,s,z.returncode
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        for r in ex.map(run,jobs):
            if r[2]!=0:raise RuntimeError(f'pretrain failed {r}')

def load_ck(v,s,device):
    m=model_for(v).to(device);m.load_state_dict(torch.load(CK/f'{v}_seed{s}.pt',map_location=device,weights_only=True));return m

def bootstrap(yt,a,b,n=10000):
    rng=np.random.default_rng(20260910);ds=[]
    for _ in range(n):
        ix=rng.integers(0,len(yt),len(yt));ra=spearmanr(yt[ix],a[ix]).statistic;rb=spearmanr(yt[ix],b[ix]).statistic
        if np.isfinite(ra) and np.isfinite(rb):ds.append(ra-rb)
    z=np.asarray(ds);return {'mean_delta_spearman':float(z.mean()),'q025':float(np.quantile(z,.025)),'q975':float(np.quantile(z,.975)),'p_delta_le_0':float((z<=0).mean())}

def predict_models(mods,x,m,device,batch=8192):
    ps=[];tx=torch.from_numpy(x);tm=torch.from_numpy(m)
    for mod in mods:
        q=[];mod.eval()
        with torch.no_grad():
            for i in range(0,len(x),batch):q.append(mod.forward_reg(tx[i:i+batch].to(device),tm[i:i+batch].to(device)).cpu().numpy())
        ps.append(np.concatenate(q))
    return np.mean(ps,0)

def decode_state(st):
    design=[(int(x)//2,bool(int(x)%2)) for x in st[:4]];gr=[(.20,.29,.38,.47,.56),(7.50,8.32,9.14,9.96,10.78),(11.,12.18,13.36,14.54,15.72),(.20,.29,.38,.47,.56)];return tuple(design),tuple(gr[i][int(st[4+i])] for i in range(4))
def eval_child(st,out):
    from evaluator import market_final_direct_q4096_v4 as E
    d,g=decode_state(st);r=E.evaluate_final_with_gaps(d,g,4096);q={'state':list(map(int,st)),'J':float(r['merit_J']),'metrics':{k:float(r[k]) for k in ['efl','fno','mtf_mean','mtf_geomean_reg','mtf_p10','onaxis_pupil_fill','min_illum','dist_max','lca']},'costs':r['costs'],'config_hash':r['config_hash']};Path(out).write_text(json.dumps(q));return 0

def eval_states(groups):
    tmp=ART/'q4096';tmp.mkdir(exist_ok=True);uniq={tuple(st) for rows in groups.values() for st in rows};cache={}
    def one(st):
        h=hashlib.sha1(json.dumps(st).encode()).hexdigest()[:16];p=tmp/f'{h}.json'
        if p.exists():return st,json.loads(p.read_text())
        cmd=[str(VENV),'-u',str(Path(__file__).resolve()),'--eval-state',json.dumps(st),'--eval-out',str(p)];z=subprocess.run(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=900)
        if z.returncode!=0:return st,{'state':list(st),'error':z.stdout[-3000:]}
        return st,json.loads(p.read_text())
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        fut=[ex.submit(one,st) for st in uniq]
        for f in cf.as_completed(fut):st,r=f.result();cache[tuple(st)]=r;print('Q',st,r.get('J',r.get('error')),flush=True)
    return {g:[cache[tuple(st)] for st in rows] for g,rows in groups.items()}
def statrows(rs):
    j=np.asarray([r['J'] for r in rs if 'J' in r],float);return {'n':int(len(j)),'mean_J':float(j.mean()),'median_J':float(np.median(j)),'best_J':float(j.min()),'worst_J':float(j.max())}

def main():
    t0=time.time();assert FIRST.exists() and MULTI.exists();launch_pretrains();first,multi,train,test,mu,sd,X,M,XT,MT,y,yt=common_data();device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');B.DEVICE=device
    result={'schema':'OPT1.0-pretrain-completion-v1','first_n':len(first),'multistate_n':len(multi),'seeds':SEEDS,'budgets':BUDGETS,'variants':VARIANTS,'external':{},'normalization':'common multistate-human+q4096-train moments'};models={};preds={}
    for v,cfg in VARIANTS.items():
        result['external'][v]={};models[v]={}
        for budget in BUDGETS:
            pp=[];mods=[];infos=[]
            for s in SEEDS:
                if cfg['pretrain']:m=load_ck(v,s,device)
                else:seedall(s);m=model_for(v).to(device)
                m,p,fi=B.finetune(m,s,X,M,y,budget,XT,MT);pp.append(p);mods.append(m);infos.append({'seed':s,'finetune':fi})
            ens=np.mean(pp,0);base='scratch_medium' if cfg['arch']['d']==96 else 'scratch_base';entry={'ensemble':B.met(yt,ens),'individual':[B.met(yt,p) for p in pp],'training':infos,'predictions':ens.tolist()};result['external'][v][str(budget)]=entry;preds[(v,budget)]=ens
            if budget in {96,192}:models[v][budget]=mods
            print('FT',v,budget,entry['ensemble'],flush=True)
    for v,cfg in VARIANTS.items():
        if not cfg['pretrain']:continue
        base='scratch_medium' if cfg['arch']['d']==96 else 'scratch_base'
        for budget in BUDGETS:result['external'][v][str(budget)]['bootstrap_vs_matching_scratch']=bootstrap(yt,preds[(v,budget)],preds[(base,budget)])
    # prospective pools: 96-label -> three independent 100k pools, top8 each; 192-label -> one new 100k pool, top8.
    known={tuple(map(int,r['state'])) for r in train+test};groups={};pool_meta=[]
    for budget,pseeds in [(96,[202609201,202609202,202609203]),(192,[202609204])]:
        for pi,ps in enumerate(pseeds):
            pool=B.pool_states(100000,known,seed=ps);raw=[B.cots_seq(st) for st in pool];norm=[B.normseq(s,mu,sd) for s in raw];px,pm=B.pad(norm);pool_meta.append({'budget':budget,'pool_index':pi,'seed':ps,'n':len(pool)})
            for v in VARIANTS:
                pr=predict_models(models[v][budget],px,pm,device);idx=np.argsort(pr)[-8:][::-1];groups[f'{budget}:{pi}:{v}']=[pool[int(i)] for i in idx]
            rr=np.random.default_rng(ps+9000);idx=rr.choice(len(pool),8,replace=False);groups[f'{budget}:{pi}:random']=[pool[int(i)] for i in idx]
            print('POOL',budget,pi,'done',flush=True)
    (ART/'opt10_pretrain_completion_proposals.json').write_text(json.dumps({g:[list(s) for s in rows] for g,rows in groups.items()},indent=2));result['prospective_pools']=pool_meta
    ev=eval_states(groups);result['prospective']={'groups':{g:{'stats':statrows(rs),'rows':rs} for g,rs in ev.items()},'pooled':{},'tests':{}}
    for budget in [96,192]:
        pis=[0,1,2] if budget==96 else [0]
        arms=list(VARIANTS)+['random']
        for a in arms:
            rows=sum([ev[f'{budget}:{pi}:{a}'] for pi in pis],[]);result['prospective']['pooled'][f'{budget}:{a}']={'stats':statrows(rows),'rows':rows}
        for v,cfg in VARIANTS.items():
            if not cfg['pretrain']:continue
            base='scratch_medium' if cfg['arch']['d']==96 else 'scratch_base';x=np.asarray([r['J'] for r in result['prospective']['pooled'][f'{budget}:{v}']['rows']],float);s=np.asarray([r['J'] for r in result['prospective']['pooled'][f'{budget}:{base}']['rows']],float);rr=np.asarray([r['J'] for r in result['prospective']['pooled'][f'{budget}:random']['rows']],float);result['prospective']['tests'][f'{budget}:{v}']={'matching_scratch':base,'vs_scratch_mwu_less':float(mannwhitneyu(x,s,alternative='less').pvalue),'P_better_scratch':float((x[:,None]<s[None,:]).mean()),'vs_random_mwu_less':float(mannwhitneyu(x,rr,alternative='less').pvalue),'P_better_random':float((x[:,None]<rr[None,:]).mean())}
    result['evaluator_hash']='0dcde5089bd66deba991214558637fe906f6c2a52530619850e6c5bab0d684da';result['elapsed_sec']=time.time()-t0;(ART/'opt10_pretrain_completion_results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'first_n':len(first),'multistate_n':len(multi),'external96':{v:result['external'][v]['96']['ensemble'] for v in VARIANTS},'prospective96':{a:result['prospective']['pooled'][f'96:{a}']['stats'] for a in list(VARIANTS)+['random']},'elapsed_sec':result['elapsed_sec']},indent=2),flush=True)

def cli():
    ap=argparse.ArgumentParser();ap.add_argument('--pretrain-one');ap.add_argument('--seed',type=int);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--eval-state');ap.add_argument('--eval-out');a=ap.parse_args()
    if a.pretrain_one:return pretrain_one(a.pretrain_one,a.seed,a.gpu)
    if a.eval_state:return eval_child(json.loads(a.eval_state),a.eval_out)
    return main()
if __name__=='__main__':cli()
