#!/usr/bin/env python3
"""Build the complete Alpha Lense pretraining prescription corpus.

Resumable, concurrent and provenance preserving. Starts from named real-lens
prescriptions, expands sibling patent examples, parses conservative structural
signals, deduplicates, family-splits, and writes review manifests. This stage
creates data only; it never trains a model.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, hashlib, json, re, sys, urllib.error, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.alpha_lense_collect_corpus_v01 import known_rows,sibling_candidates
NUM_RE=re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?')
PAT_RE=re.compile(r'((?:US|JP|CN|WO|EP|DE|KR|TW)[A-Z0-9-]+)_Example',re.I)

def get(url,timeout=12):
 req=urllib.request.Request(url,headers={'User-Agent':'AlphaLenseResearch/0.2 dataset'})
 with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()
def sha(b):return hashlib.sha256(b).hexdigest()
def norm(raw):
 s=raw.decode('utf-8','replace').replace('\r','\n');return '\n'.join(' '.join(x.strip().split()) for x in s.splitlines() if x.strip())
def parse(raw):
 t=norm(raw);lo=t.lower();lines=t.splitlines();numeric=sum(len(NUM_RE.findall(x))>=2 for x in lines)
 return {'normalized_sha256':sha(t.encode()),'bytes':len(raw),'lines':len(lines),'numeric_lines':numeric,'parseable_conservative':numeric>=4 and len(t)>80,'has_stop_marker':('stop' in lo or 'aperture' in lo),'has_asphere_marker':('asph' in lo or 'conic' in lo),'has_glass_signal':any(k in lo for k in ('glass','abbe',' nd',' vd'))}
def fetch_one(r,rawdir):
 p=rawdir/r['filename']
 try:
  if p.exists() and p.stat().st_size>20:b=p.read_bytes()
  else:
   b=get(r['url']);h=b[:256].lower()
   if b'<html' in h or b'<!doctype' in h:raise ValueError('html')
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
  q=parse(b);m=PAT_RE.search(r['filename']);return {**r,'status':'ok','sha256':sha(b),'local':str(p),'patent_family':m.group(1).upper() if m else None,**q}
 except urllib.error.HTTPError as e:return {**r,'status':'missing' if e.code==404 else 'error','http':e.code}
 except Exception as e:return {**r,'status':'error','error':repr(e)}
def family_split(fam):
 h=int(hashlib.sha256((fam or 'unknown').encode()).hexdigest()[:8],16)%100
 return 'train' if h<90 else ('val' if h<95 else 'test')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='data/alpha_lense/pretrain_dataset_v01');ap.add_argument('--max-example',type=int,default=24);ap.add_argument('--workers',type=int,default=24);a=ap.parse_args()
 out=Path(a.out);raw=out/'raw';out.mkdir(parents=True,exist_ok=True)
 known=known_rows();rows=known+sibling_candidates(known,a.max_example);print(json.dumps({'known':len(known),'candidates':len(rows),'workers':a.workers}),flush=True)
 rec=[]
 with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
  futs=[ex.submit(fetch_one,r,raw) for r in rows]
  for i,f in enumerate(cf.as_completed(futs),1):
   rec.append(f.result())
   if i%1000==0:print(json.dumps({'done':i,'ok':sum(x['status']=='ok' for x in rec)}),flush=True)
 ok=[x for x in rec if x['status']=='ok'];uniq={}
 for r in ok:
  k=r['normalized_sha256'];p=uniq.get(k)
  if p is None or (p['confidence']!='production_associated' and r['confidence']=='production_associated'):uniq[k]=r
 unique=list(uniq.values())
 for r in unique:r['split']=family_split(r['patent_family'] or r['normalized_sha256'])
 usable=[r for r in unique if r['parseable_conservative']]
 summary={'schema':2,'known_manifest':len(known),'attempted':len(rows),'downloaded':len(ok),'raw_unique':len({x['sha256'] for x in ok}),'normalized_unique':len(unique),'parseable_conservative':len(usable),'production_associated':sum(x['confidence']=='production_associated' for x in unique),'patent_only':sum(x['confidence']=='patent_only' for x in unique),'patent_families':len({x['patent_family'] for x in unique if x['patent_family']}),'asphere_marker':sum(x['has_asphere_marker'] for x in unique),'glass_signal':sum(x['has_glass_signal'] for x in unique),'missing':sum(x['status']=='missing' for x in rec),'errors':sum(x['status']=='error' for x in rec),'splits':{s:sum(x['split']==s for x in usable) for s in ('train','val','test')},'gate':'NOT training-ready until exact reconstruction + physics validation'}
 (out/'records.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in unique)+'\n');(out/'usable_ingestion.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in usable)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
