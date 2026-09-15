"""Collect PhotonsToPhotos Optical Bench prescription files for Alpha Lense v0.1.

Two modes:
1) hub: discover production-associated prescriptions from OpticalBenchHub.
2) patent list: fetch explicit Data/<patent>_ExampleNN.txt references from a manifest.

This collector preserves provenance and raw text. It deliberately does not call a
patent prescription a production lens unless the source metadata supports that.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,re,time,urllib.parse,urllib.request
from pathlib import Path

BASE='https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/'
HUB=BASE+'OpticalBenchHub.htm'
DATA_RE=re.compile(r'(?:OpticalBench\.htm#)?(Data/[^\"\'<> ]+\.txt)',re.I)

def get(url,timeout=30):
    req=urllib.request.Request(url,headers={'User-Agent':'AlphaLenseResearch/0.1 (+academic optical design corpus)'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()

def sha256(b):return hashlib.sha256(b).hexdigest()

def discover_hub():
    html=get(HUB).decode('utf-8','replace');paths=sorted(set(DATA_RE.findall(html)))
    return [{'source':'photons_to_photos_hub','confidence':'production_associated','path':p,'url':urllib.parse.urljoin(BASE,p),'hub_url':HUB} for p in paths]

def load_manifest(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        s=line.strip()
        if not s or s.startswith('#'):continue
        p=s if s.startswith('Data/') else 'Data/'+s
        rows.append({'source':'photons_to_photos_patent','confidence':'patent_only','path':p,'url':urllib.parse.urljoin(BASE,p),'hub_url':None})
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='data/alpha_lense/opticalbench');ap.add_argument('--manifest');ap.add_argument('--sleep',type=float,default=.15);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    out=Path(a.out);raw=out/'raw';raw.mkdir(parents=True,exist_ok=True)
    rows=discover_hub();
    if a.manifest:rows+=load_manifest(a.manifest)
    dedup={r['url']:r for r in rows};rows=list(dedup.values());rows=rows[:a.limit or None]
    index=[]
    for i,r in enumerate(rows,1):
        try:
            b=get(r['url']);name=Path(urllib.parse.urlparse(r['url']).path).name;dest=raw/name;dest.write_bytes(b)
            x={**r,'local':str(dest),'bytes':len(b),'sha256':sha256(b),'status':'ok'}
        except Exception as e:x={**r,'status':'error','error':repr(e)}
        index.append(x);print(json.dumps({'i':i,'n':len(rows),'status':x['status'],'path':r['path']}),flush=True);time.sleep(a.sleep)
    (out/'index.json').write_text(json.dumps(index,indent=2,ensure_ascii=False))
    print(json.dumps({'discovered':len(rows),'ok':sum(x['status']=='ok' for x in index),'errors':sum(x['status']!='ok' for x in index),'out':str(out)}))
if __name__=='__main__':main()
