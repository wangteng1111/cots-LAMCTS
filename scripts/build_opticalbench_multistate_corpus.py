#!/usr/bin/env python3
from __future__ import annotations
import gzip, hashlib, json, math, os, re, time
from pathlib import Path
ROOT=Path('/var/lib/cots-lamcts/corpora/opticalbench_hub')
RAW=ROOT/'raw'; OUT=ROOT/'parsed_multistate.json.gz'
ART=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts')); ART.mkdir(parents=True,exist_ok=True)

def fnum(x):
    try:
        s=str(x).strip()
        if s.lower() in {'infinity','inf','undefined',''}: return None
        v=float(s); return v if math.isfinite(v) else None
    except: return None

def sections(txt):
    sec=None; d={}
    for ln in txt.splitlines():
        t=ln.rstrip('\r\n')
        if t.startswith('[') and t.endswith(']'):
            sec=t[1:-1].lower(); d.setdefault(sec,[]); continue
        if sec is not None and t.strip(): d[sec].append(t.split('\t'))
    return d

def parse_file(p:Path):
    d=sections(p.read_text(errors='replace'))
    vars={r[0]:r[1:] for r in d.get('variable distances',[]) if len(r)>=2}
    const={r[0]:r[1:] for r in d.get('constants',[]) if len(r)>=2}
    fl=vars.get('Focal Length', const.get('Focal Length', []))
    if not fl: return []
    nstate=max([len(fl)]+[len(v) for v in vars.values()] or [1])
    asph={str(r[0]) for r in d.get('aspherical data',[]) if r}
    title=''
    for r in d.get('descriptive data',[]):
        if r and r[0].lower()=='title' and len(r)>1: title=r[1]
    out=[]
    def pick(a,k):
        if not a: return None
        return a[k] if k<len(a) else a[-1]
    def val(s,k):
        v=fnum(s)
        if v is not None:return v
        if s in vars:return fnum(pick(vars[s],k))
        return None
    for k in range(nstate):
        f=fnum(pick(fl,k))
        if f is None or f<=0.5: continue
        rows=[]
        for r in d.get('lens data',[]):
            if len(r)<3: continue
            sid=str(r[0]); rad=str(r[1]).strip(); th=val(r[2],k) or 0.0
            nd=fnum(r[3]) if len(r)>3 else None; vd=fnum(r[5]) if len(r)>5 else None
            ru=rad.upper()
            if ru=='AS': rows.append(['STOP',None,float(th),None,None,sid]); continue
            if ru=='FS': rows.append(['FIELDSTOP',None,float(th),None,None,sid]); continue
            if ru=='CG' or rad.lower() in {'infinity','inf'}: R=None
            else:
                R=fnum(rad)
                if R is None: continue
            rows.append(['SURF',R,float(th),nd,vd,sid])
        if sum(x[0]=='SURF' for x in rows)<4: continue
        out.append({'name':title or p.stem,'filename':p.name,'state_idx':k,'state_count':nstate,'f':float(f),'rows':rows,'asph':sorted(asph)})
    return out

def token_hash(L):
    f=L['f']; z=0.; a=[]; A=set(L['asph'])
    for typ,R,th,nd,vd,sid in L['rows']:
        curv=0. if R is None or abs(R)<1e-12 else max(-8.,min(8.,f/R))
        a.append([typ,round(curv,6),round(th/f,6),round((nd or 1.)-1.,6),round(0 if not vd else min(4.,50./vd),6),round(z/f,6),int(sid in A)])
        z+=th
    return hashlib.sha1(json.dumps(a,separators=(',',':')).encode()).hexdigest()

def main():
    t=time.time(); allcfg=[]; bad=[]
    for p in sorted(RAW.glob('*.txt')):
        try: allcfg.extend(parse_file(p))
        except Exception as e: bad.append([p.name,type(e).__name__,str(e)])
    uniq=[]; seen=set(); by_file={}; zoomfiles=set()
    for L in allcfg:
        h=token_hash(L)
        if h in seen: continue
        seen.add(h); L['structure_hash']=h; uniq.append(L)
        by_file[L['filename']]=by_file.get(L['filename'],0)+1
        if L['state_count']>1: zoomfiles.add(L['filename'])
    with gzip.open(OUT,'wt',encoding='utf-8') as f: json.dump(uniq,f,ensure_ascii=False)
    counts=sorted(by_file.values())
    s={'schema':'opticalbench-multistate-v1','raw_files':len(list(RAW.glob('*.txt'))),'raw_configs':len(allcfg),'unique_configs':len(uniq),'zoom_files':len(zoomfiles),'files_with_configs':len(by_file),'max_states_per_file':max(counts) if counts else 0,'mean_states_per_file':sum(counts)/len(counts) if counts else 0,'failed_files':len(bad),'elapsed_sec':time.time()-t,'persistent_path':str(OUT)}
    (ART/'opticalbench_multistate_summary.json').write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2),flush=True)
if __name__=='__main__':main()
