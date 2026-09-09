#!/usr/bin/env python3
from pathlib import Path
import os, sys, runpy
venv_root=Path('/var/lib/cots-lamcts/venv')
venv=venv_root/'bin/python'
root=Path(__file__).resolve().parents[1]
target=root/'validation/opt10_workstation_pretrain_meta.py'
if not venv.exists():
    raise SystemExit('persistent workstation venv is missing')
if Path(sys.prefix).resolve()!=venv_root.resolve():
    os.execv(str(venv), [str(venv), '-u', str(Path(__file__).resolve()), *sys.argv[1:]])
sys.path.insert(0,str(root))
sys.argv=[str(target),*sys.argv[1:]]
runpy.run_path(str(target),run_name='__main__')
