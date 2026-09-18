#!/usr/bin/env python3
"""Production Alpha Lense Stage0->Stage1->Stage2 data runner.

Stage0 is independently persistent. Stage1 uses a persistent dual-GPU process
pool, prescription-hash physics cache, and per-seed resume shards. Stage2 is
deterministic from Stage1 records. Model training is intentionally separate.
"""
from __future__ import annotations
import argparse,json,os,shutil,sys,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from alpha_lense.stage_pipeline_v01 import (
 Surface,Prescription,PhysicsRecord,parse_explicit_surfaces,
 generate_stage1_candidates,records_from_evaluations,assemble_stage2,
)
from scripts.alpha_lense_collect_corpus_v01 import known_rows

def fetch_parse(r):
 req=urllib.request.Request(r['url'],headers={'User-Agent':'AlphaLenseResearch/0.8'})
 with urllib.request.urlopen(req,timeout=12) as x:text=x.read().decode('utf-8','replace')
 fam=r['filename'].split('_Example')[0]
 return parse_explicit_surfaces(text,r['filename'],fam,{'confidence':'production_associated','url':r['url'],'lens_name':r.get('lens_name')})

def seed_record(p):
 return {'schema':2,'source_id':p.source_id,'family':p.family,'optical_hash':p.optical_hash(),
  'stop_after':p.stop_after,'stop_z_mm':p.stop_z_mm,'image_z_mm':p.image_z_mm,'source_config_index':p.source_config_index,
  'design_spec':p.design_spec,'provenance':p.provenance,'surfaces':[asdict(s) for s in p.surfaces]}

def seed_from_record(x):
 ss=tuple(Surface(**s) for s in x['surfaces'])
 return Prescription(ss,int(x['stop_after']),x['source_id'],x['family'],dict(x.get('design_spec') or {}),dict(x.get('provenance') or {}),
  x.get('stop_z_mm'),x.get('image_z_mm'),int(x.get('source_config_index',0)))

def _write_jsonl(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+('\n' if rows else ''))

def _atomic_jsonl(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.tmp-{os.getpid()}')
 _write_jsonl(tmp,rows);os.replace(tmp,path)

def _publish_artifacts(out,names):
 ad=os.environ.get('COTS_JOB_ARTIFACT_DIR')
 if not ad:return
 dst=Path(ad);dst.mkdir(parents=True,exist_ok=True)
 for name in names:
  p=out/name
  if p.is_file():shutil.copy2(p,dst/name)

def _persist_files(out,pdir,names):
 pdir.mkdir(parents=True,exist_ok=True)
 for name in names:
  p=out/name
  if p.is_file():shutil.copy2(p,pdir/name)

def _load_jsonl(path):
 return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def _stage0(a,out,persistent_root):
 if a.stage0_from:
  rows=_load_jsonl(a.stage0_from);seeds=[seed_from_record(x) for x in rows];qu=[]
  s0={'attempted':len(seeds),'reconstructed':len(seeds),'quarantine':0,'workers':0,'source':'canonical_file','source_path':a.stage0_from}
 else:
  rows=known_rows();rows=rows[:a.max_seeds] if a.max_seeds else rows;seeds=[];qu=[]
  with ThreadPoolExecutor(max_workers=a.stage0_workers) as ex:
   fut={ex.submit(fetch_parse,r):r for r in rows}
   for i,f in enumerate(as_completed(fut),1):
    r=fut[f]
    try:seeds.append(f.result())
    except Exception as e:qu.append({'row':r,'error_type':type(e).__name__,'error':repr(e)})
    if i%50==0:print(json.dumps({'stage':0,'done':i,'total':len(rows),'reconstructed':len(seeds),'quarantine':len(qu)}),flush=True)
  s0={'attempted':len(rows),'reconstructed':len(seeds),'quarantine':len(qu),'workers':a.stage0_workers,'source':'p2p_manifest'}
  (out/'stage0_summary.json').write_text(json.dumps(s0,indent=2));_write_jsonl(out/'stage0_quarantine.jsonl',qu);_write_jsonl(out/'stage0_seeds.jsonl',[seed_record(p) for p in seeds])
  if a.stage0_dataset_id:_persist_files(out,persistent_root/'stage0'/a.stage0_dataset_id,['stage0_summary.json','stage0_quarantine.jsonl','stage0_seeds.jsonl'])
 _publish_artifacts(out,['stage0_summary.json','stage0_quarantine.jsonl','stage0_seeds.jsonl'])
 print(json.dumps({'stage':0,'complete':s0}),flush=True)
 if not seeds:raise RuntimeError('Stage0 produced no reconstructable prescriptions')
 return seeds,qu,s0

def _records_from_shard(path):
 return [PhysicsRecord(**x) for x in _load_jsonl(path)]

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--out',default='data/alpha_lense/stage0_2_v03')
 ap.add_argument('--max-seeds',type=int,default=0);ap.add_argument('--stage0-workers',type=int,default=32);ap.add_argument('--stage0-only',action='store_true')
 ap.add_argument('--stage0-from');ap.add_argument('--stage0-dataset-id')
 ap.add_argument('--perturb-per-seed',type=int,default=100);ap.add_argument('--smoke-seeds',type=int,default=2);ap.add_argument('--smoke-perturb',type=int,default=2)
 ap.add_argument('--gpus',default='0,1');ap.add_argument('--workers-per-gpu',type=int,default=1)
 ap.add_argument('--run-id',default='alpha_lense_v03');ap.add_argument('--persistent-root',default='/var/lib/cots-lamcts/alpha_lense')
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);persistent_root=Path(a.persistent_root)
 seeds,qu,s0=_stage0(a,out,persistent_root)
 if a.stage0_only:
  print(json.dumps({'stage':0,'status':'finished_stage0_only'}),flush=True);return
 if a.smoke_seeds<=0:raise ValueError('Stage1 requires smoke_seeds > 0')

 from evaluator.prescription_q4096_pool import PrescriptionQ4096Pool
 gpus=tuple(int(x) for x in a.gpus.split(',') if x.strip())
 resume=persistent_root/'stage1_resume'/a.run_id;resume.mkdir(parents=True,exist_ok=True)
 run_dir=persistent_root/'runs'/a.run_id;run_dir.mkdir(parents=True,exist_ok=True)
 (run_dir/'run_manifest.json').write_text(json.dumps({'schema':1,'run_id':a.run_id,'git_commit':os.environ.get('COTS_GIT_COMMIT'),'stage0_count':len(seeds),'perturb_per_seed':a.perturb_per_seed,'gpus':gpus,'workers_per_gpu':a.workers_per_gpu},indent=2))
 allrec=[]
 with PrescriptionQ4096Pool(gpus=gpus,workers_per_gpu=a.workers_per_gpu) as pool:
  smoke_pairs=[]
  for j,p in enumerate(seeds[:a.smoke_seeds]):
   smoke_pairs.extend((p,c,ed) for c,ed in generate_stage1_candidates(p,a.smoke_perturb,1000+j,max_depth=2))
  smoke_ph=pool.evaluate([c for _,c,_ in smoke_pairs],bypass_cache=True)
  smoke=[]
  by_seed={}
  for triple,ph in zip(smoke_pairs,smoke_ph):
   p,c,ed=triple;by_seed.setdefault(p.optical_hash(),(p,[]))[1].append((c,ed,ph))
  for _,(p,items) in by_seed.items():
   smoke.extend(records_from_evaluations(p,[(c,ed) for c,ed,_ in items],[ph for _,_,ph in items]))
  if not smoke or any(x.physics.get('config_hash') in (None,'invalid') for x in smoke) or any('cuda' not in str(x.physics.get('backend','')).lower() for x in smoke):
   raise RuntimeError('real-prescription CUDA Q4096 smoke gate failed')
  _write_jsonl(out/'stage1_smoke.jsonl',[x.__dict__ for x in smoke]);print(json.dumps({'stage':1,'smoke_records':len(smoke),'status':'passed','pool_capacity':pool.capacity}),flush=True)

  for j,p in enumerate(seeds):
   shard=resume/f"{p.optical_hash()}.jsonl";rr=None
   if shard.is_file():
    try:
     tmp=_records_from_shard(shard)
     if len(tmp)==a.perturb_per_seed+1:rr=tmp
    except Exception:rr=None
   if rr is None:
    cand=generate_stage1_candidates(p,a.perturb_per_seed,20260917+j)
    ph=pool.evaluate([x for x,_ in cand])
    rr=records_from_evaluations(p,cand,ph)
    _atomic_jsonl(shard,[x.__dict__ for x in rr])
   allrec.extend(rr)
   if (j+1)%10==0 or j+1==len(seeds):
    print(json.dumps({'stage':1,'seeds_done':j+1,'seeds_total':len(seeds),'physics_records':len(allrec),'pool_capacity':pool.capacity}),flush=True)

 _write_jsonl(out/'stage1_physics.jsonl',[x.__dict__ for x in allrec])
 targets=assemble_stage2(allrec);_write_jsonl(out/'stage2_pretrain.jsonl',targets)
 summary={'schema':4,'stage0_reconstructed':len(seeds),'stage0_quarantine':len(qu),'stage1_physics_records':len(allrec),'stage2_targets':len(targets),'splits':{s:sum(x['split']==s for x in targets) for s in ('train','val','test')},'training_started':False,'run_id':a.run_id}
 (out/'dataset_summary.json').write_text(json.dumps(summary,indent=2))
 _persist_files(out,run_dir,['stage1_smoke.jsonl','stage1_physics.jsonl','stage2_pretrain.jsonl','dataset_summary.json'])
 _publish_artifacts(out,['stage1_smoke.jsonl','dataset_summary.json'])
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
