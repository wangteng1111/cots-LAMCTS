#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = Path('/var/lib/cots-lamcts/venv/bin/python')
if VENV.exists() and sys.prefix != str(VENV.parent.parent):
    os.execv(str(VENV), [str(VENV), '-u', str(Path(__file__).resolve()), *sys.argv[1:]])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = Path('/var/lib/cots-lamcts/q4096_mandler_fixed_shared')
HASH = '0dcde5089bd66deba991214558637fe906f6c2a52530619850e6c5bab0d684da'


def load_sample(n=16):
    rows = []
    for p in sorted(CACHE.glob('*.json')):
        try:
            r = json.loads(p.read_text())
            st = tuple(map(int, r.get('state', [])))
            if len(st) != 8 or st[4:] != (2,2,2,2):
                continue
            if r.get('config_hash') != HASH or 'J' not in r:
                continue
            rows.append((st, float(r['J'])))
            if len(rows) >= n:
                break
        except Exception:
            pass
    if len(rows) < n:
        raise RuntimeError(f'only found {len(rows)} valid cached OLMFA Q4096 states')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--workers-per-gpu', default='1,2,4')
    ns = ap.parse_args()
    gpus = tuple(int(x) for x in ns.gpus.split(','))
    levels = [int(x) for x in ns.workers_per_gpu.split(',')]
    sample = load_sample(ns.n)
    states = [x[0] for x in sample]
    expected = {x[0]: x[1] for x in sample}

    from evaluator.q4096_cuda_pool import MultiGPUQ4096Pool
    out = {'schema':'q4096-cuda-pool-benchmark-v1','n':len(states),'gpus':list(gpus),'runs':[]}
    for w in levels:
        with MultiGPUQ4096Pool(gpus=gpus, workers_per_gpu=w, write_cache=False) as pool:
            t0=time.perf_counter(); pool.warmup(); warm=time.perf_counter()-t0
            t0=time.perf_counter(); rr=pool.evaluate_states8(states,bypass_cache=True); sec=time.perf_counter()-t0
        bystate={tuple(r['state']):r for r in rr}
        diffs=[abs(float(bystate[st]['J'])-expected[st]) for st in states]
        gpu_counts={str(g):sum(1 for r in rr if int(r['gpu'])==g) for g in gpus}
        run={
            'workers_per_gpu':w,
            'capacity':w*len(gpus),
            'warmup_sec':warm,
            'batch_sec':sec,
            'states_per_sec':len(states)/sec,
            'effective_sec_per_state':sec/len(states),
            'max_abs_J_diff':max(diffs),
            'gpu_counts':gpu_counts,
        }
        out['runs'].append(run)
        print(json.dumps(run),flush=True)
    out['best']=max(out['runs'],key=lambda r:r['states_per_sec'])
    art=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'));art.mkdir(parents=True,exist_ok=True)
    (art/'q4096_cuda_pool_benchmark.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':
    main()
