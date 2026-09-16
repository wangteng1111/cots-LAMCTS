#!/usr/bin/env python3
"""Stage 0: prepare real-lens-associated seeds without blind sibling crawling."""
from __future__ import annotations
import argparse, hashlib, json, re, sys, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.alpha_lense_collect_corpus_v01 import known_rows
NUM=re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?')
PAT=re.compile(r'((?:US|JP|CN|WO|EP|DE|KR|TW)[A-Z0-9-]+)_Example',re.I)
def fetch(url):
 req=urllib.request.Request(url,headers={'User-Agent':'AlphaLenseResearch/0.3'})
 with urllib.request.urlopen(req,timeout=15) as r:return r.read()
def norm(b):return '\n'.join(' '.join(x.split()) for x in b.decode('utf-8','replace').replace('\r','\n').splitlines() if x.strip())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='data/alpha_lense/stage0');a=ap.parse_args();out=Path(a.out);raw=out/'raw';raw.mkdir(parents=True,exist_ok=True)
 records=[]
 for i,r in enumerate(known_rows(),1):
  p=raw/r['filename']
  try:
   b=p.read_bytes() if p.exists() else fetch(r['url']);p.write_bytes(b);t=norm(b);h=hashlib.sha256(t.encode()).hexdigest();m=PAT.search(r['filename']);numeric=sum(len(NUM.findall(x))>=2 for x in t.splitlines())
   records.append({**r,'status':'ok','normalized_sha256':h,'patent_family':m.group(1).upper() if m else None,'numeric_lines':numeric,'ingestion_ok':numeric>=4 and len(t)>80,'local':str(p)})
  except Exception as e:records.append({**r,'status':'error','error':repr(e),'ingestion_ok':False})
  if i%200==0:print(json.dumps({'stage':0,'done':i}),flush=True)
 uniq={}
 for r in records:
  if r.get('status')=='ok':uniq.setdefault(r['normalized_sha256'],r)
 seeds=list(uniq.values());usable=[x for x in seeds if x['ingestion_ok']];quarantine=[x for x in records if not x.get('ingestion_ok')]
 (out/'stage0_seeds.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in usable)+'\n');(out/'stage0_quarantine.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in quarantine)+'\n')
 s={'schema':1,'known_manifest':len(records),'unique':len(seeds),'usable_ingestion':len(usable),'quarantine':len(quarantine),'note':'Stage0 ingestion only; exact prescription reconstruction is required before Stage1.'};(out/'stage0_summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s),flush=True)
if __name__=='__main__':main()
