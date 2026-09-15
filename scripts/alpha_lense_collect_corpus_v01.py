#!/usr/bin/env python3
"""Alpha Lense v0.1 corpus collection.

The P2P Hub table is JS-populated, so static HTML scraping returns zero.  This
collector bootstraps from the public flaresim_nuke known-lens manifest, whose
entries point to P2P OpticalBench prescription files and production lens names.
It then optionally probes sibling patent examples to enlarge the patent-only
pool while preserving provenance.
"""
from __future__ import annotations
import argparse, hashlib, json, re, time, urllib.request, urllib.error
from pathlib import Path

MANIFEST='https://raw.githubusercontent.com/LocalStarlight/flaresim_nuke/master/lenses/all_lenses.sh'
P2P_RE=re.compile(r'https://www\.photonstophotos\.net/GeneralTopics/Lenses/OpticalBench/Data/([^\s]+?\.txt)')
OUT_RE=re.compile(r'-o\s+(?:["\']?)([^\n]+?\.lens)(?:["\']?)\s*$')
EX_RE=re.compile(r'^(.*?_Example)(\d+)([^/]*)\.txt$',re.I)

def get(url,timeout=20):
    req=urllib.request.Request(url,headers={'User-Agent':'AlphaLenseResearch/0.1 optical-prescription corpus'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()

def sha(b):return hashlib.sha256(b).hexdigest()

def known_rows():
    text=get(MANIFEST).decode('utf-8','replace'); rows=[]
    for line in text.splitlines():
        m=P2P_RE.search(line)
        if not m:continue
        fn=m.group(1); om=OUT_RE.search(line); lens=Path(om.group(1)).stem if om else None
        rows.append({'filename':fn,'url':'https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/Data/'+fn,
                     'confidence':'production_associated','source':'flaresim_nuke_known_lens_manifest','lens_name':lens})
    # URL identity is authoritative; retain first associated name.
    return list({r['url']:r for r in rows}.values())

def sibling_candidates(rows,max_example):
    seen={r['filename'] for r in rows}; out=[]
    # Patent family identity is filename prefix before _Example.
    families={}
    for r in rows:
        m=EX_RE.match(r['filename'])
        if m:families.setdefault(m.group(1),set()).add(m.group(3))
    # Probe common suffixes observed for each patent, plus plain and P.
    for prefix,suffixes in families.items():
        suffixes=set(suffixes)|{'','P'}
        for n in range(1,max_example+1):
            for suf in suffixes:
                fn=f'{prefix}{n:02d}{suf}.txt'
                if fn in seen:continue
                seen.add(fn);out.append({'filename':fn,'url':'https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/Data/'+fn,
                    'confidence':'patent_only','source':'sibling_example_probe','lens_name':None})
    return out

def download(rows,raw,sleep,progress_every=100):
    index=[];raw.mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(rows,1):
        try:
            b=get(r['url']);
            # Reject HTML error pages that happen to return 200.
            head=b[:256].lower()
            if b'<html' in head or b'<!doctype' in head:raise ValueError('html-not-prescription')
            dest=raw/r['filename'];dest.write_bytes(b)
            x={**r,'status':'ok','bytes':len(b),'sha256':sha(b),'local':str(dest)}
        except urllib.error.HTTPError as e:x={**r,'status':'missing' if e.code==404 else 'error','http':e.code}
        except Exception as e:x={**r,'status':'error','error':repr(e)}
        index.append(x)
        if i%progress_every==0:print(json.dumps({'i':i,'n':len(rows),'ok':sum(z['status']=='ok' for z in index)}),flush=True)
        if sleep:time.sleep(sleep)
    return index

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='data/alpha_lense/corpus_v01');ap.add_argument('--max-example',type=int,default=0);ap.add_argument('--sleep',type=float,default=.02);a=ap.parse_args()
    out=Path(a.out);known=known_rows();print(json.dumps({'known_manifest':len(known)}),flush=True)
    idx=download(known,out/'raw',a.sleep)
    if a.max_example:
        siblings=sibling_candidates(known,a.max_example);print(json.dumps({'sibling_probes':len(siblings)}),flush=True)
        idx+=download(siblings,out/'raw',a.sleep)
    ok=[x for x in idx if x['status']=='ok']; uniq={x['sha256'] for x in ok}
    summary={'known_manifest':len(known),'attempted':len(idx),'downloaded':len(ok),'unique_sha256':len(uniq),
             'production_associated':sum(x['status']=='ok' and x['confidence']=='production_associated' for x in idx),
             'patent_only':sum(x['status']=='ok' and x['confidence']=='patent_only' for x in idx),
             'missing':sum(x['status']=='missing' for x in idx),'errors':sum(x['status']=='error' for x in idx)}
    out.mkdir(parents=True,exist_ok=True);(out/'index.json').write_text(json.dumps(idx,ensure_ascii=False,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
