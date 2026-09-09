#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

VENV_PY = Path('/var/lib/cots-lamcts/venv/bin/python')
ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ('search/', 'validation/', 'scripts/', 'benchmarks/')

if len(sys.argv) < 2:
    raise SystemExit('usage: run_with_workstation_venv.py <repo-relative-script.py> [args...]')
rel = sys.argv[1].replace('\\', '/').lstrip('/')
if '..' in Path(rel).parts or not rel.endswith('.py') or not any(rel.startswith(p) for p in ALLOWED):
    raise SystemExit(f'disallowed target: {rel}')
target = (ROOT / rel).resolve()
if ROOT not in target.parents or not target.is_file():
    raise SystemExit(f'target not found in checkout: {rel}')
if not VENV_PY.is_file():
    raise SystemExit(f'workstation venv missing: {VENV_PY}; run bootstrap_workstation_env.py first')
os.execv(str(VENV_PY), [str(VENV_PY), '-u', str(target), *sys.argv[2:]])
