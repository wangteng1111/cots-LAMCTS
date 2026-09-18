#!/usr/bin/env python3
"""Train Alpha Lense from Stage2 physics-grounded targets, initialized by OPTv1."""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from alpha_lense.optv1_alpha_model import AlphaLensePretrainModel,load_optv1_encoder,MAXLEN

def seedall(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def file_sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def optv1_norm():
 from validation import opt10_pretrain_completion_run as C
 from validation import opt10_workstation_pretrain_meta as B
 multi=C.seq_items(C.MULTI)
 with gzip.open(C.DATA,'rt') as f:ds=json.load(f)
 geom=[B.cots_seq(r['state']) for r in ds['train_192']]
 stat=np.concatenate([x['seq'] for x in multi]+geom,axis=0)
 return stat[:,:5].mean(0).astype(np.float32), (stat[:,:5].std(0)+1e-5).astype(np.float32)

def fin(x):
 try:return math.isfinite(float(x))
 except:return False

def vfeat(v):
 return 0.0 if not fin(v) or float(v)<=0 else float(np.clip(50.0/float(v),0,4))
def curv(r,f):
 return 0.0 if not fin(r) or abs(float(r))<1e-9 else float(np.clip(f/float(r),-8,8))

def tokenize(row,mu,sd):
 st=row['state'];spec=row.get('design_spec') or {};surfs=st.get('surfaces') or []
 f=abs(float(spec.get('efl_target_mm'))) if fin(spec.get('efl_target_mm')) and abs(float(spec.get('efl_target_mm')))>1e-6 else 100.0
 ent=[];z=0.0
 for i,s in enumerate(surfs):
  ent.append({'z':z,'surface':s,'stop':0,'order':i});z+=float(s.get('thickness',0.0))
 sz=st.get('stop_z_mm')
 if fin(sz):ent.append({'z':float(sz),'surface':None,'stop':1,'order':-1})
 ent.sort(key=lambda e:(e['z'],e['stop']))
 seq=[]
 for i,e in enumerate(ent[:MAXLEN]):
  nz=ent[i+1]['z'] if i+1<len(ent) else e['z'];dz=float(nz-e['z'])
  if e['stop']:seq.append([0.0,dz/f,0.0,0.0,e['z']/f,1.0,0.0,0.0])
  else:
   s=e['surface'];n=float(s.get('n_after',1.0));asp=bool(s.get('asphere')) or abs(float(s.get('conic',0.0)))>1e-14
   seq.append([curv(s.get('radius'),f),dz/f,n-1.0,vfeat(s.get('v_after')),e['z']/f,0.0,float(n>1.0001),float(asp)])
 arr=np.zeros((MAXLEN,8),np.float32);mask=np.zeros(MAXLEN,bool)
 n=min(len(seq),MAXLEN)
 if n:arr[:n]=np.asarray(seq[:n],np.float32);mask[:n]=True
 arr[:,:5]=(arr[:,:5]-mu)/sd
 return arr,mask,f

def specvec(spec):
 efl=abs(float(spec['efl_target_mm'])) if fin(spec.get('efl_target_mm')) else None
 vals=[
  math.log1p(efl)/5.0 if efl else 0.0,
  float(spec['max_f_number'])/8.0 if fin(spec.get('max_f_number')) else 0.0,
  float(spec['max_field_deg'])/90.0 if fin(spec.get('max_field_deg')) else 0.0,
  float(spec['image_circle_mm'])/100.0 if fin(spec.get('image_circle_mm')) else 0.0,
  float(spec['bfd_mm'])/max(efl or 100.0,1e-6) if fin(spec.get('bfd_mm')) else 0.0,
  float(spec['total_length_mm'])/max(efl or 100.0,1e-6) if fin(spec.get('total_length_mm')) else 0.0,
  float(spec['stop_diameter_mm'])/max(efl or 100.0,1e-6) if fin(spec.get('stop_diameter_mm')) else 0.0,
  math.log1p(float(spec.get('source_configuration_count',1)))/4.0,
 ]
 keys=('efl_target_mm','max_f_number','max_field_deg','image_circle_mm','bfd_mm','total_length_mm','stop_diameter_mm','source_configuration_count')
 masks=[1.0 if (k=='source_configuration_count' or fin(spec.get(k))) else 0.0 for k in keys]
 return np.asarray(vals+masks,np.float32)

def delta_targets(row,scale):
 y=np.zeros((MAXLEN,5),np.float32);m=np.zeros((MAXLEN,5),bool)
 d=row.get('policy_goal_delta') or {}
 for op in d.get('operations',[]):
  if op.get('op')!='match':continue
  i=op.get('current_index')
  if i is None or i<0 or i>=MAXLEN:continue
  vals=[op.get('radius_delta'),op.get('thickness_delta'),op.get('n_after_delta'),op.get('v_after_delta'),op.get('aperture_delta')]
  scales=[scale,scale,1.0,50.0,scale]
  for j,(v,s) in enumerate(zip(vals,scales)):
   if fin(v):y[i,j]=float(v)/max(float(s),1e-6);m[i,j]=True
 topo=float(d.get('surface_count_delta',0.0))
 return y,m,topo

def load_rows(path):
 rows=[]
 with open(path,'r',encoding='utf-8') as f:
  for ln in f:
   if not ln.strip():continue
   x=json.loads(ln);ph=x.get('physics_target') or {}
   if not fin(ph.get('J',ph.get('merit_J'))) or x.get('evaluator_config_hash') in (None,'invalid'):continue
   rows.append(x)
 return rows

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--data',required=True)
 ap.add_argument('--checkpoint',default='/var/lib/cots-lamcts/checkpoints/opt10_completion_v1/mlm_contrast_base_seed17.pt')
 ap.add_argument('--out-dir',default='/var/lib/cots-lamcts/checkpoints/alpha_lense_v01')
 ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--batch',type=int,default=64);ap.add_argument('--seed',type=int,default=17)
 ap.add_argument('--device',default='cuda:0');a=ap.parse_args();seedall(a.seed)
 device=torch.device(a.device if torch.cuda.is_available() else 'cpu')
 if device.type!='cuda':raise RuntimeError('Alpha Lense pretraining requires CUDA')
 rows=load_rows(a.data)
 if len(rows)<8:raise RuntimeError(f'too few Stage2 rows: {len(rows)}')
 mu,sd=optv1_norm();tok=[];mask=[];spec=[];dy=[];dm=[];topo=[];mer=[];val=[];feas=[];rfeas=[];vio=[];rvio=[];spl=[]
 for x in rows:
  t,m,scale=tokenize(x,mu,sd);y,ym,tp=delta_targets(x,scale)
  tok.append(t);mask.append(m);spec.append(specvec(x.get('design_spec') or {}));dy.append(y);dm.append(ym);topo.append(tp)
  mer.append(float(x['merit_target']));val.append(float(x['value_target']));feas.append(float(bool(x['feasible_target'])));rfeas.append(float(bool(x['reachable_feasible_target'])))
  vio.append(math.log1p(max(0.0,float(x['violation_target']))));rvio.append(math.log1p(max(0.0,float(x['reachable_violation_target']))));spl.append(x.get('split','train'))
 tok=np.asarray(tok);mask=np.asarray(mask);spec=np.asarray(spec);dy=np.asarray(dy);dm=np.asarray(dm);topo=np.asarray(topo,np.float32)
 mer=np.asarray(mer,np.float32);val=np.asarray(val,np.float32);feas=np.asarray(feas,np.float32);rfeas=np.asarray(rfeas,np.float32);vio=np.asarray(vio,np.float32);rvio=np.asarray(rvio,np.float32)
 tr=np.asarray([i for i,s in enumerate(spl) if s=='train'],int);va=np.asarray([i for i,s in enumerate(spl) if s=='val'],int)
 if len(va)<4:
  rng=np.random.default_rng(a.seed+91);perm=rng.permutation(tr);nv=max(4,min(len(tr)//5,64));va=perm[:nv];tr=perm[nv:]
 mmu=float(mer[tr].mean());msd=float(mer[tr].std()+1e-6);vmu=float(val[tr].mean());vsd=float(val[tr].std()+1e-6)
 merz=(mer-mmu)/msd;valz=(val-vmu)/vsd
 tensors=[torch.from_numpy(x).to(device) for x in (tok,mask,spec,dy,dm,topo,merz,valz,feas,rfeas,vio,rvio)]
 TT,TM,TS,TD,TDM,TTO,TMER,TVAL,TF,TRF,TV,TRV=tensors
 model=AlphaLensePretrainModel().to(device);init=load_optv1_encoder(model,a.checkpoint)
 enc=list(model.optical.parameters());enc_ids={id(p) for p in enc};heads=[p for p in model.parameters() if id(p) not in enc_ids]
 opt=torch.optim.AdamW([{'params':enc,'lr':8e-5},{'params':heads,'lr':5e-4}],weight_decay=2e-3)
 rng=np.random.default_rng(a.seed+1001);best=1e30;bestsd=None;history=[]
 def loss_for(ix):
  o=model(TT[ix],TM[ix],TS[ix])
  l_mer=F.smooth_l1_loss(o['merit'],TMER[ix],beta=.5);l_val=F.smooth_l1_loss(o['value'],TVAL[ix],beta=.5)
  l_f=F.binary_cross_entropy_with_logits(o['feasibility_logit'],TF[ix]);l_rf=F.binary_cross_entropy_with_logits(o['reachable_feasibility_logit'],TRF[ix])
  l_v=F.smooth_l1_loss(o['violation'],TV[ix],beta=.25);l_rv=F.smooth_l1_loss(o['reachable_violation'],TRV[ix],beta=.25)
  md=TDM[ix]
  if bool(md.any()):l_d=F.smooth_l1_loss(o['surface_delta'][md],TD[ix][md],beta=.2)
  else:l_d=torch.zeros((),device=device)
  l_t=F.smooth_l1_loss(o['topology_count'],TTO[ix],beta=.5)
  total=l_mer+l_val+.25*(l_f+l_rf)+.15*(l_v+l_rv)+.8*l_d+.35*l_t
  return total,{'merit':l_mer,'value':l_val,'feas':l_f,'rfeas':l_rf,'vio':l_v,'rvio':l_rv,'delta':l_d,'topo':l_t}
 for ep in range(a.epochs):
  model.train();perm=rng.permutation(tr);els=[]
  for p in range(0,len(perm),a.batch):
   ix=torch.from_numpy(perm[p:p+a.batch]).to(device);loss,_=loss_for(ix);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();els.append(float(loss.detach()))
  model.eval()
  with torch.no_grad():vl,parts=loss_for(torch.from_numpy(va).to(device));vlf=float(vl)
  rec={'epoch':ep+1,'train_loss':float(np.mean(els)),'val_loss':vlf,**{f'val_{k}':float(v) for k,v in parts.items()}};history.append(rec)
  print(json.dumps(rec),flush=True)
  if vlf<best:best=vlf;bestsd={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);model.load_state_dict(bestsd)
 ck=out/f'alpha_lense_optv1_seed{a.seed}.pt'
 torch.save({'model':model.state_dict(),'optv1_checkpoint':a.checkpoint,'optv1_sha256':file_sha(a.checkpoint),'normalization':{'mu':mu.tolist(),'sd':sd.tolist()},'target_norm':{'merit_mean':mmu,'merit_sd':msd,'value_mean':vmu,'value_sd':vsd},'schema':'AlphaLense-pretrain-v1'},ck)
 summary={'schema':'AlphaLense-pretrain-v1','data':a.data,'rows':len(rows),'train_n':len(tr),'val_n':len(va),'device':str(device),'optv1_init':init,'optv1_sha256':file_sha(a.checkpoint),'epochs':a.epochs,'best_val_loss':best,'checkpoint':str(ck),'checkpoint_sha256':file_sha(ck),'history':history}
 (out/f'alpha_lense_optv1_seed{a.seed}.json').write_text(json.dumps(summary,indent=2))
 ad=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','')); 
 if str(ad):ad.mkdir(parents=True,exist_ok=True);(ad/'alpha_lense_pretrain_summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps({k:summary[k] for k in ('schema','rows','train_n','val_n','device','epochs','best_val_loss','checkpoint','checkpoint_sha256')},indent=2),flush=True)
if __name__=='__main__':main()
