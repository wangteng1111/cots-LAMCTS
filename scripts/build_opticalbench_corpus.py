#!/usr/bin/env python3
from __future__ import annotations
import concurrent.futures as cf, gzip, hashlib, html, json, math, os, re, time, urllib.request
from pathlib import Path

HUB='https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/OpticalBenchHub.htm'
BASE='https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/Data/'
ROOT=Path('/var/lib/cots-lamcts/corpora/opticalbench_hub'); RAW=ROOT/'raw'; RAW.mkdir(parents=True,exist_ok=True)
ART=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts')); ART.mkdir(parents=True,exist_ok=True)
UA={'User-Agent':'Mozilla/5.0 (compatible; COTS-LAMCTS research corpus builder)'}

def get(url,timeout=45):
    return urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=timeout).read().decode('utf-8','replace')

def fnum(x):
    try:
        if str(x).strip().lower() in {'infinity','inf','undefined',''}: return None
        v=float(str(x).strip()); return v if math.isfinite(v) else None
    except: return None

def parse_sections(txt):
    sec=None; d={}
    for ln in txt.splitlines():
        if ln.startswith('[') and ln.rstrip().endswith(']'):
            sec=ln.strip()[1:-1].lower(); d.setdefault(sec,[]); continue
        if sec is not None and ln.strip(): d[sec].append(ln.rstrip('\r\n').split('\t'))
    return d

def parse_one(meta,txt):
    d=parse_sections(txt)
    vars={r[0]:r[1:] for r in d.get('variable distances',[]) if len(r)>=2}
    const={r[0]:r[1:] for r in d.get('constants',[]) if len(r)>=2}
    f=None
    for src in (vars,const):
        if 'Focal Length' in src and src['Focal Length']:
            f=fnum(src['Focal Length'][0]);
            if f is not None: break
    if f is None or f<=0.5: return None
    def val(s):
        v=fnum(s)
        if v is not None:return v
        if s in vars and vars[s]: return fnum(vars[s][0])
        return None
    rows=[]
    for p in d.get('lens data',[]):
        if len(p)<3: continue
        sid=str(p[0]); rad=str(p[1]).strip(); thi=val(p[2]) or 0.0
        nd=fnum(p[3]) if len(p)>3 else None; vd=fnum(p[5]) if len(p)>5 else None
        if rad.upper()=='AS': rows.append(['STOP',None,float(thi),None,None,sid]); continue
        if rad.upper()=='FS': rows.append(['FIELDSTOP',None,float(thi),None,None,sid]); continue
        if rad.upper()=='CG': R=None
        elif rad.lower() in {'infinity','inf'}: R=None
        else:
            R=fnum(rad)
            if R is None: continue
        rows.append(['SURF',R,float(thi),nd,vd,sid])
    if len([r for r in rows if r[0]=='SURF'])<4:return None
    asph={str(p[0]) for p in d.get('aspherical data',[]) if p}
    title=''
    for p in d.get('descriptive data',[]):
        if p and p[0].lower()=='title' and len(p)>1:title=p[1]
    return {'name':meta['description'],'patent':meta['patent'],'example':meta['example'],'tale':meta['tale'],'filename':meta['filename'],'title':title,'f':float(f),'rows':rows,'asph':sorted(asph)}

def token_hash(L):
    f=L['f']; z=0.; vals=[]
    for typ,R,th,nd,vd,sid in L['rows']:
        curv=0. if R is None or abs(R)<1e-12 else max(-8.,min(8.,f/R))
        vals.append([typ,round(curv,5),round(th/f,5),round((nd or 1.)-1.,5),round(0. if not vd else min(4.,50./vd),5),round(z/f,5),int(sid in set(L['asph']))])
        z+=th
    return hashlib.sha1(json.dumps(vals,separators=(',',':')).encode()).hexdigest()

def main():
    t=time.time(); hub=get(HUB)
    pat=re.compile(r"\{\s*PatentID:\s*'([^']*)',\s*ExampleID:\s*'([^']*)',\s*TaleID:\s*'([^']*)',\s*Description:\s*'([^']*)'\s*\}")
    metas=[]
    for patent,example,tale,desc in pat.findall(hub):
        fn=f'{patent}_{example}'+(f'_{tale}' if tale else '')+'.txt'
        metas.append({'patent':patent,'example':example,'tale':tale,'description':html.unescape(desc),'filename':fn})
    # unique URLs preserving table order
    uu=[]; seen=set()
    for m in metas:
        if m['filename'] not in seen:seen.add(m['filename']);uu.append(m)
    metas=uu
    def fetch_parse(m):
        p=RAW/m['filename']; txt=None; err=None
        for k in range(3):
            try:
                if p.exists() and p.stat().st_size>20: txt=p.read_text(errors='replace')
                else:
                    txt=get(BASE+m['filename'],timeout=30); p.write_text(txt)
                q=parse_one(m,txt)
                return {'meta':m,'lens':q,'error':None}
            except Exception as e:
                err=f'{type(e).__name__}: {e}'; time.sleep(.3*(k+1))
        return {'meta':m,'lens':None,'error':err}
    rows=[]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        fut=[ex.submit(fetch_parse,m) for m in metas]
        for i,f in enumerate(cf.as_completed(fut),1):
            rows.append(f.result())
            if i%100==0: print('fetched',i,'/',len(metas),flush=True)
    good=[r['lens'] for r in rows if r['lens']]
    uniq=[]; hs=set(); dup=0
    for L in good:
        h=token_hash(L)
        if h in hs:dup+=1;continue
        hs.add(h);L['structure_hash']=h;uniq.append(L)
    out=ROOT/'parsed_lenses.json.gz'
    with gzip.open(out,'wt',encoding='utf-8') as f:json.dump(uniq,f,ensure_ascii=False)
    errors=[{'filename':r['meta']['filename'],'error':r['error']} for r in rows if not r['lens']]
    summary={'schema':'opticalbench-corpus-v1','hub_entries':len(metas),'parsed':len(good),'unique_prescriptions':len(uniq),'duplicates_removed':dup,'failed':len(errors),'elapsed_sec':time.time()-t,'persistent_path':str(out),'errors':errors[:100]}
    (ART/'opticalbench_corpus_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    (ART/'opticalbench_corpus_names.json').write_text(json.dumps([{'name':x['name'],'filename':x['filename'],'f':x['f'],'surfaces':sum(r[0]=='SURF' for r in x['rows']),'aspheres':len(x['asph'])} for x in uniq],indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='errors'},indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
