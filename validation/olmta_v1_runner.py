#!/usr/bin/env python3
"""Execution wrapper for OLMTA v1.0.

The OPT regression head emits shape (N,1).  OLMTA acquisition must rank N
candidates, so mu/sigma are flattened to (N,) before tree coordinates and
acquisition sorting.  This wrapper fail-closes on any unexpected shape.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = Path('/var/lib/cots-lamcts/venv/bin/python')
if sys.prefix != str(VENV.parent.parent) and VENV.exists():
    os.execv(str(VENV), [str(VENV), '-u', str(Path(__file__).resolve()), *sys.argv[1:]])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from validation import olmta_v1_mandler_meta as M


def predict_flat(models, states4, mu, sd, device, ym, ys):
    pmu, psd, _ = M.BASE.predict_and_anchor(
        models, states4, mu, sd, device, ym, ys, need_anchor=False
    )
    pmu = np.asarray(pmu, np.float32).reshape(-1)
    psd = np.asarray(psd, np.float32).reshape(-1)
    if len(pmu) != len(states4) or len(psd) != len(states4):
        raise RuntimeError(
            f'OPTv1 prediction shape mismatch: states={len(states4)} '
            f'mu={pmu.shape} sigma={psd.shape}'
        )
    return pmu, psd


M.predict = predict_flat

if __name__ == '__main__':
    M.cli()
