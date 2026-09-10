#!/usr/bin/env python3
"""Runtime wrapper for OPTv1 Mandler validation.

Keeps the exact validation code/algorithm unchanged, but relocates the per-state
Q4096 cache from COTS_JOB_ARTIFACT_DIR into COTS_JOB_DIR so the workstation
bridge uploads only the final scientific artifacts instead of ~1200 cache JSONs.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'optv1_lamcts_mandler_meta.py'
TMP = HERE / '_optv1_lamcts_mandler_meta_runtime.py'
text = SRC.read_text()
old = "cache = ART / 'q4096_mandler_fixed'"
new = "cache = Path(os.environ.get('COTS_JOB_DIR', str(ART))) / 'q4096_mandler_fixed'"
if old not in text:
    raise RuntimeError('expected cache line not found; refusing to run a silently different script')
TMP.write_text(text.replace(old, new, 1))
os.execv(sys.executable, [sys.executable, '-u', str(TMP), *sys.argv[1:]])
