#!/usr/bin/env python3
"""Unified Stage0->Stage1->Stage2 Alpha Lense data generation runner."""
from __future__ import annotations
import argparse,json,sys,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from alpha_lense.stage_pipeline_v01 import parse_explicit_surfaces,generate_stage1,assemble_stage2
from evaluator.prescription_q4096 import evaluate_prescription
from scripts.alpha_lense_collect_corpus_v01 import known_rows

def fetch_parse(r):
 req=urllib.request.Request(r['url'],headers={'User-Agent':'AlphaLenseResearch/0.5'})
 with urllib.request.urlopen(req,timeout=12) as x:text=x.read().decode('utf-8','replace')
 fam=r['filename'].split('_Example')[0]
 return parse_explicit_surfaces(text,r['filename'],fam,{'confidence':'production_associated','url':r['url'],'lens_name':r.get('lens_name')})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='data/alpha_lense/stage0_2_v02');ap.add_argument('--max-seeds',type=int,default=0);ap.add_argument('--stage0-workers',type=int,default=32);ap.add_argument('--perturb-per-seed',type=int,default=100);ap.add_argument('--smoke-seeds',type=int,default=2);ap.add_argument('--smoke-perturb',type=int,default=2);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 rows=known_rows();rows=rows[:a.max_seeds] if a.max_seeds else rows;seeds=[];qu=[]
 with ThreadPoolExecutor(max_workers=a.stage0_workers) as ex:
  fut={ex.submit(fetch_parse,r):r for r in rows}
  for i,f in enumerate(as_completed(fut),1):
   r=fut[f]
   try:seeds.append(f.result())
   except Exception as e:qu.append({'row':r,'error':repr(e)})
   if i%50==0:print(json.dumps({'stage':0,'done':i,'total':len(rows),'reconstructed':len(seeds),'quarantine':len(qu)}),flush=True)
 s0={'attempted':len(rows),'reconstructed':len(seeds),'quarantine':len(qu),'workers':a.stage0_workers};(out/'stage0_summary.json').write_text(json.dumps(s0,indent=2));(out/'stage0_quarantine.jsonl').write_text('\n'.join(json.dumps(x) for x in qu)+'\n');print(json.dumps({'stage':0,'complete':s0}),flush=True)
 if not seeds:raise RuntimeError('Stage0 produced no reconstructable prescriptions; parser/source grammar mismatch')
 smoke=[]
 for j,p in enumerate(seeds[:a.smoke_seeds]):
  rr=generate_stage1(p,lambda x:evaluate_prescription(x,device=j%2),a.smoke_perturb,1000+j,max_depth=2);smoke.extend(rr);print(json.dumps({'stage':1,'smoke_seed':j+1,'records':len(rr),'backends':list({x.physics.get('backend') for x in rr}),'errors':[x.physics.get('error') for x in rr if x.physics.get('error')]}),flush=True)
 if not smoke or any(x.physics.get('config_hash') is None for x in smoke) or any('cuda' not in str(x.physics.get('backend','')).lower() for x in smoke):raise RuntimeError('real-prescription CUDA Q4096 smoke gate failed')
 (out/'stage1_smoke.jsonl').write_text('\n'.join(json.dumps(x.__dict__) for x in smoke)+'\n');print(json.dumps({'stage':1,'smoke_records':len(smoke),'status':'passed'}),flush=True)
 allrec=[]
 for j,p in enumerate(seeds):
  rr=generate_stage1(p,lambda x,d=j%2:evaluate_prescription(x,device=d),a.perturb_per_seed,20260917+j);allrec.extend(rr)
  print(json.dumps({'stage':1,'seeds_done':j+1,'seeds_total':len(seeds),'physics_records':len(allrec),'device':j%2}),flush=True)
 (out/'stage1_physics.jsonl').write_text('\n'.join(json.dumps(x.__dict__) for x in allrec)+'\n');targets=assemble_stage2(allrec);(out/'stage2_pretrain.jsonl').write_text('\n'.join(json.dumps(x) for x in targets)+'\n')
 summary={'schema':2,'stage0_reconstructed':len(seeds),'stage0_quarantine':len(qu),'stage1_physics_records':len(allrec),'stage2_targets':len(targets),'splits':{s:sum(x['split']==s for x in targets) for s in ('train','val','test')},'training_started':False};(out/'dataset_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
