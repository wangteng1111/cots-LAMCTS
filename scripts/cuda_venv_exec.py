#!/usr/bin/env python3
"""Execute a repo-local Python entrypoint with the validated workstation CUDA venv."""
from __future__ import annotations
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY=Path("/var/lib/cots-lamcts/venv/bin/python")
if len(sys.argv)<2:raise SystemExit("usage: cuda_venv_exec.py <repo-relative.py> [args...]")
target=(ROOT/sys.argv[1]).resolve()
if ROOT not in target.parents or target.suffix!=".py" or not target.is_file():raise SystemExit("invalid repo-local Python target")
if not (str(target.relative_to(ROOT)).startswith("validation/") or str(target.relative_to(ROOT)).startswith("scripts/")):raise SystemExit("target must be under validation/ or scripts/")
os.execv(str(PY),[str(PY),"-u",str(target),*sys.argv[2:]])
