#!/usr/bin/env python3
"""Unified Stage0->Stage1->Stage2 Alpha Lense data generation runner.

Runs a mandatory real-prescription Q4096 smoke gate before scaling Stage1.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from alpha_lense.stage_pipeline_v01 import parse_explicit_surfaces,generate_stage1,assemble_stage2
from evaluator.prescription_q4096 import evaluate_prescription
from scripts.alpha_lense_collect_corpus_v01 import known_rows
import urllib.request

def fetch(url):
 req=urllib.request.Request(url,headers={'User-Agent':'AlphaLenseResearch/0.4'})
 with urllib.request.urlopen(req,timeout=15) as r:return r.read().decode('utf-8','replace')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='data/alpha_lense/stage0_2_v01');ap.add_argument('--max-seeds',type=int,default=0);ap.add_argument('--perturb-per-seed',type=int,default=100);ap.add_argument('--smoke-seeds',type=int,default=2);ap.add_argument('--smoke-perturb',type=int,default=2);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 seeds=[];qu=[]
 rows=known_rows();rows=rows[:a.max_seeds] if a.max_seeds else rows
 for i,r in enumerate(rows):
  try:
   text=fetch(r['url']);fam=r['filename'].split('_Example')[0];p=parse_explicit_surfaces(text,r['filename'],fam,{'confidence':'production_associated','url':r['url'],'lens_name':r.get('lens_name')});seeds.append(p)
  except Exception as e:qu.append({'row':r,'error':repr(e)})
  if (i+1)%100==0:print(json.dumps({'stage':0,'done':i+1,'reconstructed':len(seeds),'quarantine':len(qu)}),flush=True)
 (out/'stage0_summary.json').write_text(json.dumps({'attempted':len(rows),'reconstructed':len(seeds),'quarantine':len(qu)},indent=2));(out/'stage0_quarantine.jsonl').write_text('\n'.join(json.dumps(x) for x in qu)+'\n')
 if not seeds:raise RuntimeError('Stage0 produced no exact-enough prescription; inspect fixtures/parser before physics generation')
 # mandatory physics smoke: no scale-out unless arbitrary prescription Q4096 actually returns finite authoritative labels
 smoke=[]
 for j,p in enumerate(seeds[:a.smoke_seeds]):smoke.extend(generate_stage1(p,evaluate_prescription,a.smoke_perturb,1000+j,max_depth=2))
 if not smoke or any(x.physics.get('config_hash') is None for x in smoke):raise RuntimeError('prescription Q4096 smoke failed')
 (out/'stage1_smoke.jsonl').write_text('\n'.join(json.dumps(x.__dict__) for x in smoke)+'\n');print(json.dumps({'stage':1,'smoke_records':len(smoke),'status':'passed'}),flush=True)
 allrec=[]
 for j,p in enumerate(seeds):
  rr=generate_stage1(p,evaluate_prescription,a.perturb_per_seed,20260917+j);allrec.extend(rr)
  if (j+1)%10==0:print(json.dumps({'stage':1,'seeds_done':j+1,'physics_records':len(allrec)}),flush=True)
 (out/'stage1_physics.jsonl').write_text('\n'.join(json.dumps(x.__dict__) for x in allrec)+'\n')
 targets=assemble_stage2(allrec);(out/'stage2_pretrain.jsonl').write_text('\n'.join(json.dumps(x) for x in targets)+'\n')
 summary={'schema':1,'stage0_reconstructed':len(seeds),'stage0_quarantine':len(qu),'stage1_physics_records':len(allrec),'stage2_targets':len(targets),'splits':{s:sum(x['split']==s for x in targets) for s in ('train','val','test')},'training_started':False};(out/'dataset_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
