#!/usr/bin/env python3
from pathlib import Path
import os, sys
root=Path(__file__).resolve().parents[1]
target=root/'validation'/'opt10_workstation_meta.py'
s=target.read_text()
old="elif sec=='[constants]' and len(cols)>=2 and cols[0].lower()=='focal length':"
new="elif len(cols)>=2 and cols[0].lower()=='focal length':"
if old not in s:
    raise SystemExit('expected parser condition not found; refusing silent patch')
s=s.replace(old,new,1)
target.write_text(s)
print('patched goptical focal-length parser in detached worktree',flush=True)
os.execv(sys.executable,[sys.executable,'-u',str(target)])
