#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation import opt10_pretrain_completion_run as C

KNOWN = [
    ((0, 2, 4, 6), 11.906835229630682, 'reference'),
    ((0, 45, 4, 62), 17.985040982462767, 'olmfa-late-best'),
    ((16, 45, 44, 62), 18.156177194169597, 'olmfa-prev-best'),
]
FIXED_GAPS = (2, 2, 2, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ns = ap.parse_args()
    os.environ['COTS_EVAL_CUDA_DEVICE'] = str(ns.gpu)
    from evaluator import market_final_direct_q4096_v4_cuda as G
    G.set_cuda_device(ns.gpu)

    art = Path(os.environ.get('COTS_JOB_ARTIFACT_DIR', 'artifacts'))
    art.mkdir(parents=True, exist_ok=True)

    # One warm-up evaluation creates the CUDA context and allocator state.
    s4, _, _ = KNOWN[0]
    d, gaps = C.decode_state(tuple(s4) + FIXED_GAPS)
    t0 = time.perf_counter(); wr = G.evaluate_final_with_gaps(d, gaps, 4096); G.synchronize(); warmup = time.perf_counter()-t0
    print('WARMUP', warmup, wr['merit_J'], flush=True)

    rows = []
    for s4, expected, label in KNOWN:
        d, gaps = C.decode_state(tuple(s4) + FIXED_GAPS)
        t0 = time.perf_counter(); r = G.evaluate_final_with_gaps(d, gaps, 4096); G.synchronize(); sec = time.perf_counter()-t0
        row = {
            'label': label,
            'state4': list(s4),
            'expected_J': expected,
            'gpu_J': float(r['merit_J']),
            'abs_J_diff': abs(float(r['merit_J'])-expected),
            'rel_J_diff': abs(float(r['merit_J'])-expected)/max(abs(expected),1e-12),
            'seconds': sec,
            'config_hash': r['config_hash'],
        }
        rows.append(row)
        print('GPU', label, s4, 'J', row['gpu_J'], 'expected', expected, 'dJ', row['abs_J_diff'], 'sec', sec, flush=True)

    out = {
        'schema': 'q4096-cuda-known-label-v1',
        'backend': G.backend_info(),
        'warmup_seconds': warmup,
        'rows': rows,
        'mean_seconds': sum(x['seconds'] for x in rows)/len(rows),
        'max_abs_J_diff': max(x['abs_J_diff'] for x in rows),
        'max_rel_J_diff': max(x['rel_J_diff'] for x in rows),
        'all_config_hash_equal': all(x['config_hash'] == G.CONFIG_HASH for x in rows),
    }
    (art/'q4096_cuda_known_labels.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
