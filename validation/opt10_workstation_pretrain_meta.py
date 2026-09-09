#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, math, random, gzip, subprocess, shutil, time, re, argparse, hashlib
from pathlib import Path

VENV=Path('/var/lib/cots-lamcts/venv/bin/python')
if Path(sys.executable).resolve()!=VENV.resolve() and VENV.exists():
    os.execv(str(VENV), [str(VENV), '-u', __file__, *sys.argv[1:]])

os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr, kendalltau, mannwhitneyu

ROOT=Path(__file__).resolve().parents[1]
ART=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts')); ART.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data/meta/opt10_compact_q4096_dataset.json.gz'
CORPUS_ROOT=Path('/var/lib/cots-lamcts/corpora/goptical')
MAXLEN=64; FIELDS=8
SEEDS=[17,43,89,131,173]
BUDGETS=[48,96,192]
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(8)

# ---- exact evaluator child mode ------------------------------------------------
def decode_state(state):
    st=list(map(int,state)); design=[(x//2,bool(x%2)) for x in st[:4]]
    grids=[(.20,.29,.38,.47,.56),(7.50,8.32,9.14,9.96,10.78),(11.0,12.18,13.36,14.54,15.72),(.20,.29,.38,.47,.56)]
    gaps=tuple(grids[i][st[4+i]] for i in range(4))
    return tuple(design), gaps

def eval_state_one(state):
    from evaluator import market_final_direct_q4096_v4 as E
    design,gaps=decode_state(state)
    r=E.evaluate_final_with_gaps(design,gaps,4096)
    return {'state':list(map(int,state)),'J':float(r['merit_J']),'metrics':{k:r[k] for k in ['efl','fno','mtf_mean','mtf_geomean_reg','mtf_p10','onaxis_pupil_fill','min_illum','dist_max','lca']},'costs':r['costs'],'config_hash':r['config_hash']}

def child_eval_mode(args):
    state=json.loads(args.eval_state)
    out=eval_state_one(state)
    Path(args.eval_out).write_text(json.dumps(out))
    return 0

# ---- corpus acquisition + parsing ---------------------------------------------
def ensure_corpus():
    if (CORPUS_ROOT/'.git').exists():
        subprocess.run(['git','-C',str(CORPUS_ROOT),'fetch','--depth','1','origin','master'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        subprocess.run(['git','-C',str(CORPUS_ROOT),'reset','--hard','origin/master'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    else:
        CORPUS_ROOT.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--depth','1','https://github.com/dibyendumajumdar/goptical.git',str(CORPUS_ROOT)],check=True)

def fnum(s):
    try:
        v=float(str(s).strip()); return v if math.isfinite(v) else None
    except: return None

def parse_goptical_file(p:Path):
    lines=p.read_text(errors='ignore').splitlines(); sec=''; variables={}; f=None; rows=[]; asph=set()
    for ln in lines:
        t=ln.strip()
        if not t: continue
        if t.startswith('[') and t.endswith(']'): sec=t.lower(); continue
        parts=[x.strip() for x in re.split(r'\t+',t)]
        if sec=='[variable distances]' and len(parts)>=2:
            v=fnum(parts[1]);
            if v is not None: variables[parts[0]]=v
        elif sec=='[constants]' and len(parts)>=2 and parts[0].lower()=='focal length':
            f=fnum(parts[1])
        elif sec=='[lens data]' and len(parts)>=2:
            key=parts[0]
            if key.upper() in {'AS','FS','STOP'}:
                th=fnum(parts[1]) if len(parts)>1 else 0.0
                if th is None: th=variables.get(parts[1],0.0)
                rows.append(('STOP',None,float(th or 0),None,None))
                continue
            if key.upper()=='CG':
                continue
            try: no=int(key)
            except: continue
            R=None if parts[1].lower() in {'infinity','inf'} else fnum(parts[1])
            if R is None and parts[1].lower() not in {'infinity','inf'}: continue
            th=0.0
            if len(parts)>2:
                th=fnum(parts[2])
                if th is None: th=variables.get(parts[2],0.0)
            nd=fnum(parts[3]) if len(parts)>3 and parts[3] else None
            vd=fnum(parts[5]) if len(parts)>5 and parts[5] else None
            rows.append((no,R,float(th or 0),nd,vd))
        elif sec=='[aspherical data]' and parts:
            try: asph.add(int(parts[0]))
            except: pass
    if f is None or f<=1 or len([r for r in rows if r[0]!='STOP'])<4: return None
    return {'name':p.parent.name+'/'+p.name,'f':float(f),'rows':rows,'asph':asph}

def _curv(R,f):
    if R is None or not math.isfinite(float(R)) or abs(float(R))<1e-9:return 0.0
    return float(np.clip(f/float(R),-8,8))
def _vfeat(v): return 0.0 if v is None or float(v)<=0 else float(np.clip(50.0/float(v),0,4))

def human_seq(L):
    f=L['f']; z=0.; seq=[]
    for no,R,thi,nd,vd in L['rows']:
        if no=='STOP': seq.append([0,float(thi)/f,0,0,z/f,1,0,0]); z+=float(thi); continue
        n=1.0 if nd is None else float(nd)
        seq.append([_curv(R,f),float(thi)/f,n-1.,_vfeat(vd),z/f,0,float(nd is not None),float(no in L['asph'])]); z+=float(thi)
    return np.asarray(seq[:MAXLEN],np.float32)

# COTS sequence from exact physical catalog
from core import mandler_benchmark_core as M
GAP_GRIDS=[(.20,.29,.38,.47,.56),(7.50,8.32,9.14,9.96,10.78),(11.0,12.18,13.36,14.54,15.72),(.20,.29,.38,.47,.56)]
def cots_seq(state):
    st=tuple(map(int,state)); comps=[M.CAT[x//2].oriented(bool(x%2)) for x in st[:4]]; gaps=[GAP_GRIDS[i][st[4+i]] for i in range(4)]
    g12,g2s,gs3,g34=gaps; L=[sum(c.th) for c in comps]
    z2=-g2s-L[1]; z1=z2-g12-L[0]; z3=gs3; z4=z3+L[2]+g34; starts=[z1,z2,z3,z4]
    entries=[]; f=100.0
    for c,z0 in zip(comps,starts):
        z=z0
        for j,R in enumerate(c.radii):
            if j<len(c.ne): n=float(c.ne[j]); v=float(c.ve[j])
            else: n=1.; v=None
            entries.append({'z':z,'R':R,'n':n,'v':v,'stop':0,'asp':0})
            if j<len(c.th): z+=float(c.th[j])
    entries.append({'z':0.,'R':None,'n':1.,'v':None,'stop':1,'asp':0}); entries.sort(key=lambda q:(q['z'],q['stop']))
    seq=[]
    for i,e in enumerate(entries):
        dz=(entries[i+1]['z']-e['z']) if i+1<len(entries) else 0.
        seq.append([_curv(e['R'],f),dz/f,e['n']-1.,_vfeat(e['v']),e['z']/f,e['stop'],float(e['n']>1.0001),e['asp']])
    return np.asarray(seq[:MAXLEN],np.float32)

def pad(seqs):
    x=np.zeros((len(seqs),MAXLEN,FIELDS),np.float32); m=np.zeros((len(seqs),MAXLEN),bool)
    for i,s in enumerate(seqs): n=min(len(s),MAXLEN); x[i,:n]=s[:n];m[i,:n]=1
    return x,m

# ---- model --------------------------------------------------------------------
class SurfaceTransformer(nn.Module):
    def __init__(self,d=64,heads=4,layers=3,ff=128):
        super().__init__(); self.d=d
        self.proj=nn.Sequential(nn.Linear(FIELDS,d),nn.LayerNorm(d),nn.GELU()); self.mask_token=nn.Parameter(torch.randn(1,1,d)*.02); self.cls=nn.Parameter(torch.randn(1,1,d)*.02); self.pos=nn.Parameter(torch.randn(1,MAXLEN+1,d)*.02)
        el=nn.TransformerEncoderLayer(d,heads,ff,dropout=.08,activation='gelu',batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(el,layers,norm=nn.LayerNorm(d)); self.recon=nn.Linear(d,FIELDS); self.reg=nn.Sequential(nn.Linear(d,48),nn.GELU(),nn.Linear(48,1))
    def encode(self,x,valid,maskpos=None):
        h=self.proj(x)
        if maskpos is not None: h=torch.where(maskpos.unsqueeze(-1),self.mask_token.expand(h.shape[0],h.shape[1],-1),h)
        h=torch.cat([self.cls.expand(x.shape[0],-1,-1),h],1)+self.pos[:,:x.shape[1]+1]
        pm=torch.cat([torch.zeros((x.shape[0],1),dtype=torch.bool,device=x.device),~valid],1)
        return self.enc(h,src_key_padding_mask=pm)
    def forward_reg(self,x,valid): return self.reg(self.encode(x,valid)[:,0]).squeeze(-1)

def seedall(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def pretrain(seed,hx,hm,epochs=300):
    seedall(seed); model=SurfaceTransformer().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=2e-3)
    bx=torch.from_numpy(hx).to(DEVICE); bm=torch.from_numpy(hm).to(DEVICE); rng=np.random.default_rng(seed+500); best=1e9; bestsd=None; views=24
    for ep in range(epochs):
        sel=rng.integers(0,len(hx),size=max(128,len(hx)*views)); tx=bx[sel]; tm=bm[sel]; xx=tx.clone()
        noise=torch.from_numpy(rng.normal(0,.018,size=xx[:,:,:5].shape).astype(np.float32)).to(DEVICE); xx[:,:,:5]+=noise*tm.unsqueeze(-1)
        mp_np=(rng.random(tm.shape)<.30)&tm.detach().cpu().numpy()
        for i in range(len(mp_np)):
            if not mp_np[i].any():
                ids=np.flatnonzero(tm[i].detach().cpu().numpy());
                if len(ids): mp_np[i,ids[0]]=True
        mp=torch.from_numpy(mp_np).to(DEVICE)
        model.train(); z=model.encode(xx,tm,mp)[:,1:]; pred=model.recon(z); loss=((pred-tx)**2)[mp].mean()
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); lv=float(loss.detach())
        if lv<best: best=lv; bestsd={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(bestsd); return model, best

def finetune(model,seed,X,MASK,y,budget,testX,testM,epochs=400):
    seedall(seed); model.reg=nn.Sequential(nn.Linear(model.d,48),nn.GELU(),nn.Linear(48,1)).to(DEVICE)
    rng=np.random.default_rng(20260910+budget); order=np.arange(len(y)); rng.shuffle(order); use=order[:budget]
    nv=max(8,min(32,int(.18*budget))); va=use[:nv]; tr=use[nv:]
    ym=float(y[use].mean()); ys=float(y[use].std()+1e-6); z=(y-ym)/ys
    a=torch.from_numpy(X).to(DEVICE); mm=torch.from_numpy(MASK).to(DEVICE); zz=torch.from_numpy(z).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=2e-3); lf=nn.SmoothL1Loss(beta=.5); best=1e9; bestsd=None; stale=0
    for ep in range(epochs):
        model.train(); perm=torch.tensor(rng.permutation(tr),device=DEVICE)
        for p in range(0,len(tr),32):
            ix=perm[p:p+32]; pr=model.forward_reg(a[ix],mm[ix]); loss=lf(pr,zz[ix]); opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        model.eval();
        with torch.no_grad(): vl=float(lf(model.forward_reg(a[va],mm[va]),zz[va]).item())
        if vl<best-1e-4: best=vl;bestsd={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=45: break
    model.load_state_dict(bestsd); model.eval();
    with torch.no_grad(): pred=model.forward_reg(torch.from_numpy(testX).to(DEVICE),torch.from_numpy(testM).to(DEVICE)).cpu().numpy()*ys+ym
    return model,pred,{'val':best,'epochs':ep+1,'budget':budget}

def met(ytrue,pred):
    J=np.exp(-ytrue); n=len(ytrue); k=max(1,n//4); top=np.argsort(pred)[-k:]
    return {'spearman':float(spearmanr(ytrue,pred).statistic),'kendall':float(kendalltau(ytrue,pred).statistic),'top_quartile_precision':float(len(set(top)&set(np.argsort(ytrue)[-k:]))/k),'pred_top8_mean_J':float(J[np.argsort(pred)[-min(8,n):]].mean()),'pred_top8_best_J':float(J[np.argsort(pred)[-min(8,n):]].min())}

# ---- fresh pool + exact prospective -------------------------------------------
def pool_states(n,seen,seed=202609101):
    rng=np.random.default_rng(seed); out=[]; sset=set(seen)
    while len(out)<n:
        m=min(200000,n-len(out)+1000); a=np.c_[rng.integers(0,72,size=(m,4)),rng.integers(0,5,size=(m,4))]
        for row in a:
            t=tuple(map(int,row))
            if t not in sset: sset.add(t); out.append(t)
            if len(out)>=n: break
    return out

def predict_pool(models,pool_seq,mu,sd,batch=2048):
    X=[normseq(s,mu,sd) for s in pool_seq]; x,m=pad(X); preds=[]
    for model in models:
        model.eval(); chunks=[]
        with torch.no_grad():
            for i in range(0,len(x),batch): chunks.append(model.forward_reg(torch.from_numpy(x[i:i+batch]).to(DEVICE),torch.from_numpy(m[i:i+batch]).to(DEVICE)).cpu().numpy())
        preds.append(np.concatenate(chunks))
    return np.mean(preds,0)

def normseq(s,mu,sd): q=s.copy();q[:,:5]=(q[:,:5]-mu)/sd;return q

def evaluate_fixed(states,label):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    tmp=ART/f'_eval_{label}'; tmp.mkdir(exist_ok=True); script=Path(__file__).resolve()
    def one(i,st):
        out=tmp/f'{i}.json'; cmd=[sys.executable,'-u',str(script),'--eval-state',json.dumps(list(st)),'--eval-out',str(out)]; last=''
        for attempt in range(3):
            try:
                p=subprocess.run(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=240,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1'})
                last=p.stdout[-4000:]
                if p.returncode==0 and out.exists(): return json.loads(out.read_text())
            except subprocess.TimeoutExpired: last='timeout'
        return {'state':list(st),'error':last}
    res=[None]*len(states)
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(one,i,st):i for i,st in enumerate(states)}
        for f in as_completed(fut): res[fut[f]]=f.result(); print('eval',label,fut[f],res[fut[f]].get('J',res[fut[f]].get('error')),flush=True)
    return res

# ---- main ----------------------------------------------------------------------
def main():
    t0=time.time(); ensure_corpus()
    lenses=[]
    for p in sorted((CORPUS_ROOT/'data').rglob('*.txt')):
        try:
            q=parse_goptical_file(p)
            if q: lenses.append(q)
        except Exception: pass
    uniq=[]; seenh=set()
    for L in lenses:
        s=human_seq(L); h=hashlib.sha1(np.round(s,5).tobytes()).hexdigest()
        if h not in seenh: seenh.add(h); uniq.append(L)
    lenses=uniq; hseq=[human_seq(L) for L in lenses]
    with gzip.open(DATA,'rt') as f: ds=json.load(f)
    train=ds['train_192']; test=ds['heldout_24']+ds['prospective_32']
    trseq=[cots_seq(r['state']) for r in train]; teseq=[cots_seq(r['state']) for r in test]
    stat=np.concatenate(hseq+trseq,axis=0); mu=stat[:,:5].mean(0); sd=stat[:,:5].std(0)+1e-5
    hseq=[normseq(s,mu,sd) for s in hseq]; trseq=[normseq(s,mu,sd) for s in trseq]; teseq=[normseq(s,mu,sd) for s in teseq]
    HX,HM=pad(hseq); X,MASK=pad(trseq); XT,MT=pad(teseq); y=np.asarray([-math.log(float(r['J'])+1e-9) for r in train],np.float32); yt=np.asarray([-math.log(float(r['J'])+1e-9) for r in test],np.float32)
    result={'schema':'OPT1.0-workstation-pretrain-meta-v1','device':str(DEVICE),'torch':torch.__version__,'human_corpus_count':len(lenses),'human_names':[L['name'] for L in lenses],'q4096_train':len(train),'external_test':len(test),'seeds':SEEDS,'budgets':BUDGETS,'models':{}}
    final_models={'scratch':[],'pretrained':[]}
    for budget in BUDGETS:
        result['models'][str(budget)]={}
        for kind in ['scratch','pretrained']:
            pp=[]; infos=[]
            for seed in SEEDS:
                if kind=='pretrained': model,pl=pretrain(seed,HX,HM)
                else: seedall(seed); model=SurfaceTransformer().to(DEVICE); pl=None
                model,p,fi=finetune(model,seed,X,MASK,y,budget,XT,MT); pp.append(p); infos.append({'seed':seed,'pretrain_loss':pl,'finetune':fi})
                if budget==192: final_models[kind].append(model)
                print('fit',budget,kind,seed,met(yt,p),flush=True)
            ens=np.mean(pp,0); result['models'][str(budget)][kind]={'ensemble':met(yt,ens),'individual':[met(yt,p) for p in pp],'training':infos,'predictions':ens.tolist()}
    ps=np.asarray(result['models']['192']['pretrained']['predictions']); ss=np.asarray(result['models']['192']['scratch']['predictions']); rng=np.random.default_rng(20260910); dif=[]
    for _ in range(10000):
        ix=rng.integers(0,len(yt),size=len(yt)); a=spearmanr(yt[ix],ps[ix]).statistic; b=spearmanr(yt[ix],ss[ix]).statistic
        if np.isfinite(a) and np.isfinite(b): dif.append(a-b)
    result['paired_bootstrap_192']={'mean_delta_spearman':float(np.mean(dif)),'q025':float(np.quantile(dif,.025)),'q975':float(np.quantile(dif,.975)),'p_delta_le_0':float(np.mean(np.asarray(dif)<=0))}
    allknown={tuple(map(int,r['state'])) for r in train+test}; pool=pool_states(100000,allknown); print('pool generated',len(pool),flush=True)
    pseq=[cots_seq(st) for st in pool]; spr=predict_pool(final_models['scratch'],pseq,mu,sd); ppr=predict_pool(final_models['pretrained'],pseq,mu,sd)
    sidx=list(np.argsort(spr)[-8:][::-1]); pidx=list(np.argsort(ppr)[-8:][::-1]); rr=np.random.default_rng(202609102); ridx=list(rr.choice(len(pool),8,replace=False))
    selected={'scratch':[pool[i] for i in sidx],'pretrained':[pool[i] for i in pidx],'random':[pool[i] for i in ridx]}
    (ART/'opt10_workstation_proposals.json').write_text(json.dumps({k:[list(x) for x in v] for k,v in selected.items()},indent=2))
    evals={k:evaluate_fixed(v,k) for k,v in selected.items()}
    def groupstats(rows):
        js=[float(r['J']) for r in rows if 'J' in r]
        return {'n':len(js),'mean_J':float(np.mean(js)) if js else None,'median_J':float(np.median(js)) if js else None,'best_J':float(np.min(js)) if js else None,'worst_J':float(np.max(js)) if js else None}
    result['prospective']={k:{'stats':groupstats(v),'rows':v} for k,v in evals.items()}
    if all(len([r for r in evals[k] if 'J' in r])==8 for k in evals):
        sj=[r['J'] for r in evals['scratch']]; pj=[r['J'] for r in evals['pretrained']]; rj=[r['J'] for r in evals['random']]
        result['prospective']['tests']={'pretrained_vs_scratch_mwu_one_sided':float(mannwhitneyu(pj,sj,alternative='less').pvalue),'pretrained_vs_random_mwu_one_sided':float(mannwhitneyu(pj,rj,alternative='less').pvalue)}
    result['elapsed_sec']=time.time()-t0; result['evaluator_hash']='0dcde5089bd66deba991214558637fe906f6c2a52530619850e6c5bab0d684da'
    (ART/'opt10_workstation_pretrain_meta_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'human_corpus_count':len(lenses),'external_192':{k:result['models']['192'][k]['ensemble'] for k in ['scratch','pretrained']},'bootstrap':result['paired_bootstrap_192'],'prospective':{k:result['prospective'][k]['stats'] for k in ['scratch','pretrained','random']},'elapsed_sec':result['elapsed_sec']},indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--eval-state');ap.add_argument('--eval-out');args=ap.parse_args()
    if args.eval_state: raise SystemExit(child_eval_mode(args))
    main()
