"""Parse Optical Bench raw text conservatively into a training inventory.

The raw source remains authoritative. Parser records topology and numeric surface
rows when recognizable, and emits explicit completeness flags rather than
inventing missing optical data.
"""
from __future__ import annotations
import argparse,json,re,hashlib
from pathlib import Path
NUM=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?'
PAT=re.compile(r'(US|JP|EP|WO|CN|KR|DE|TW)0*([0-9A-Z]+).*?Example0*([0-9]+)',re.I)

def f(x):
    try:return float(x)
    except:return None

def parse(path):
    text=Path(path).read_text(errors='replace');lines=text.splitlines();surfaces=[];stop_count=0;asphere=False
    for line in lines:
        lo=line.lower();asphere|=('aspher' in lo or 'conic' in lo)
        if 'stop' in lo:stop_count+=1
        nums=re.findall(NUM,line)
        # Conservative generic row: surface index, radius, thickness plus optional nd/vd/semi-dia.
        if len(nums)>=3 and re.match(r'^\s*\d+\b',line):
            vals=[f(x) for x in nums[:7]]
            if vals[0] is not None and vals[1] is not None and vals[2] is not None:surfaces.append(vals)
    m=PAT.search(Path(path).name);patent=(m.group(1).upper()+m.group(2)) if m else None;example=int(m.group(3)) if m else None
    # Approximate element count from positive-thickness glass-like rows cannot be safely inferred generically;
    # retain surface count and defer exact element topology to source-specific parser/validator.
    return {'file':str(path),'patent':patent,'example':example,'surface_rows':len(surfaces),'surfaces_numeric':surfaces,'has_asphere_text':asphere,'stop_mentions':stop_count,'parseable_minimal':len(surfaces)>=2,'topology_exact':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--out');a=ap.parse_args();root=Path(a.root);idx=json.loads((root/'index.json').read_text());rows=[]
    for src in idx:
        if src.get('status')!='ok':continue
        p=parse(src['local']);rows.append({**src,**p})
    out=Path(a.out or root/'inventory.json');out.write_text(json.dumps(rows,indent=2,ensure_ascii=False));
    print(json.dumps({'records':len(rows),'minimal_parseable':sum(r['parseable_minimal'] for r in rows),'production_associated':sum(r['confidence']=='production_associated' for r in rows),'patent_only':sum(r['confidence']=='patent_only' for r in rows),'out':str(out)}))
if __name__=='__main__':main()
