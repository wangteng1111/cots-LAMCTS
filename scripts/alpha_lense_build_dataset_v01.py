#!/usr/bin/env python3
"""Alpha Lense v0.1 dataset generation/audit pipeline.

Stage A: recover known production-associated P2P prescriptions from the public
known-lens manifest.
Stage B: expand sibling patent examples while preserving patent_only provenance.
Stage C: parse prescription text conservatively, deduplicate normalized optical
content, and emit a reviewable dataset funnel. No training is started here.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.alpha_lense_collect_corpus_v01 import known_rows,sibling_candidates,download

NUM=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?'
FLOAT_RE=re.compile(NUM)
PAT_RE=re.compile(r'((?:US|JP|CN|WO|EP|DE|KR|TW)[A-Z0-9-]+)_Example',re.I)

def normalized_text(raw:bytes)->str:
    s=raw.decode('utf-8','replace').replace('\r\n','\n').replace('\r','\n')
    lines=[]
    for ln in s.splitlines():
        q=' '.join(ln.strip().split())
        if q: lines.append(q)
    return '\n'.join(lines)

def parse_one(path:Path):
    raw=path.read_bytes(); text=normalized_text(raw); low=text.lower(); lines=text.splitlines()
    numeric=[]
    for ln in lines:
        vals=FLOAT_RE.findall(ln)
        if len(vals)>=2: numeric.append(ln)
    # Conservative structural signals. We deliberately do not hallucinate a
    # surface schema until a format-specific parser is verified on fixtures.
    has_stop=('stop' in low or 'aperture' in low)
    has_asphere=('asph' in low or 'conic' in low or 'even asphere' in low)
    glass_tokens=sum(k in low for k in ('glass','nd','vd','abbe','index'))
    parseable=len(numeric)>=4 and len(text)>80
    return {'normalized_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'bytes':len(raw),'lines':len(lines),'numeric_lines':len(numeric),
            'parseable_conservative':parseable,'has_stop_marker':has_stop,
            'has_asphere_marker':has_asphere,'glass_signal_count':glass_tokens}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='data/alpha_lense/dataset_v01');ap.add_argument('--max-example',type=int,default=12);ap.add_argument('--sleep',type=float,default=.01);a=ap.parse_args()
    out=Path(a.out);raw=out/'raw';out.mkdir(parents=True,exist_ok=True)
    known=known_rows(); siblings=sibling_candidates(known,a.max_example)
    print(json.dumps({'known':len(known),'sibling_probes':len(siblings),'max_example':a.max_example}),flush=True)
    idx=download(known+siblings,raw,a.sleep,progress_every=500)
    records=[]
    for x in idx:
        if x.get('status')!='ok':continue
        p=raw/x['filename']; q=parse_one(p); m=PAT_RE.search(x['filename'])
        records.append({**x,**q,'patent_family':m.group(1).upper() if m else None})
    bynorm={}
    for r in records:
        # Prefer production-associated record when normalized content duplicates.
        k=r['normalized_sha256']; prev=bynorm.get(k)
        if prev is None or (prev['confidence']!='production_associated' and r['confidence']=='production_associated'):bynorm[k]=r
    unique=list(bynorm.values()); usable=[r for r in unique if r['parseable_conservative']]
    summary={'schema':1,'known_manifest':len(known),'sibling_probes':len(siblings),'attempted':len(idx),
      'downloaded':len(records),'raw_unique_sha256':len({r['sha256'] for r in records}),
      'normalized_unique':len(unique),'parseable_conservative':len(usable),
      'production_associated_unique':sum(r['confidence']=='production_associated' for r in unique),
      'patent_only_unique':sum(r['confidence']=='patent_only' for r in unique),
      'patent_families':len({r['patent_family'] for r in unique if r['patent_family']}),
      'asphere_marker':sum(r['has_asphere_marker'] for r in unique),'stop_marker':sum(r['has_stop_marker'] for r in unique),
      'missing':sum(x.get('status')=='missing' for x in idx),'errors':sum(x.get('status')=='error' for x in idx),
      'note':'parseable_conservative is an ingestion gate, not yet authoritative optical reconstruction/physics validation'}
    (out/'index.json').write_text(json.dumps(idx,ensure_ascii=False,indent=2));(out/'records.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in unique)+'\n');(out/'usable_manifest.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in usable)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
