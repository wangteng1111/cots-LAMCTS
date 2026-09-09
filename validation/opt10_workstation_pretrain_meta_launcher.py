#!/usr/bin/env python3
from pathlib import Path
import os, sys
venv=Path('/var/lib/cots-lamcts/venv/bin/python')
target=Path(__file__).resolve().parent/'opt10_workstation_pretrain_meta.py'
if not venv.exists():
    raise SystemExit('persistent workstation venv is missing')
os.execv(str(venv), [str(venv), '-u', str(target), *sys.argv[1:]])
