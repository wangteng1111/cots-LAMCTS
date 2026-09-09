#!/usr/bin/env python3
from pathlib import Path
import os, sys, runpy, importlib.util, re, math
VENV_ROOT=Path('/var/lib/cots-lamcts/venv'); VENV=VENV_ROOT/'bin/python'
if Path(sys.prefix).resolve()!=VENV_ROOT.resolve() and VENV.exists():
    os.execv(str(VENV),[str(VENV),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
target=ROOT/'validation/opt10_workstation_pretrain_meta.py'
spec=importlib.util.spec_from_file_location('opt10meta',str(target)); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def fixed_parse(p:Path):
    lines=p.read_text(errors='ignore').splitlines(); sec=''; variables={}; f=None; rows=[]; asph=set()
    for ln in lines:
        t=ln.strip()
        if not t: continue
        if t.startswith('[') and t.endswith(']'): sec=t.lower(); continue
        parts=[x.strip() for x in re.split(r'\t+',t)]
        if sec in {'[variable distances]','[constants]'} and len(parts)>=2:
            v=mod.fnum(parts[1])
            if sec=='[variable distances]' and v is not None: variables[parts[0]]=v
            if parts[0].lower()=='focal length' and v is not None: f=v
        elif sec=='[lens data]' and len(parts)>=2:
            key=parts[0]
            if key.upper() in {'AS','FS','STOP'}:
                th=mod.fnum(parts[1]) if len(parts)>1 else 0.0
                if th is None: th=variables.get(parts[1],0.0)
                rows.append(('STOP',None,float(th or 0),None,None)); continue
            if key.upper()=='CG': continue
            try: no=int(key)
            except: continue
            R=None if parts[1].lower() in {'infinity','inf'} else mod.fnum(parts[1])
            if R is None and parts[1].lower() not in {'infinity','inf'}: continue
            th=0.0
            if len(parts)>2:
                th=mod.fnum(parts[2])
                if th is None: th=variables.get(parts[2],0.0)
            nd=mod.fnum(parts[3]) if len(parts)>3 and parts[3] else None
            vd=mod.fnum(parts[5]) if len(parts)>5 and parts[5] else None
            rows.append((no,R,float(th or 0),nd,vd))
        elif sec=='[aspherical data]' and parts:
            try: asph.add(int(parts[0]))
            except: pass
    if f is None or f<=1 or len([r for r in rows if r[0]!='STOP'])<4: return None
    return {'name':p.parent.name+'/'+p.name,'f':float(f),'rows':rows,'asph':asph}

mod.parse_goptical_file=fixed_parse
if __name__=='__main__':
    mod.main()
