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
from evaluator import market_final_direct_q4096_v4 as CPU

DEFAULT_STATES4 = [
    (0, 2, 4, 6),          # hidden reference components
    (0, 45, 4, 62),        # late OLMFA v1.0 best
    (16, 45, 44, 62),      # previous late best
]
FIXED_GAPS = (2, 2, 2, 2)
METRICS = ['merit_J','efl','fno','mtf_mean','mtf_geomean_reg','mtf_p10','onaxis_pupil_fill','min_illum','dist_max','lca']


def eval_one(mod, s4):
    st8 = tuple(s4) + FIXED_GAPS
    d, g = C.decode_state(st8)
    t0 = time.perf_counter()
    r = mod.evaluate_final_with_gaps(d, g, 4096)
    if hasattr(mod, 'synchronize'):
        mod.synchronize()
    dt = time.perf_counter() - t0
    return r, dt


def compact(r):
    return {k: float(r[k]) for k in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ns = ap.parse_args()
    art = Path(os.environ.get('COTS_JOB_ARTIFACT_DIR', 'artifacts'))
    art.mkdir(parents=True, exist_ok=True)

    cpu_rows = []
    for s4 in DEFAULT_STATES4:
        r, dt = eval_one(CPU, s4)
        cpu_rows.append((s4, r, dt))
        print('CPU', s4, 'J', r['merit_J'], 'sec', dt, flush=True)

    os.environ['COTS_EVAL_CUDA_DEVICE'] = str(ns.gpu)
    from evaluator import market_final_direct_q4096_v4_cuda as GPU
    GPU.set_cuda_device(ns.gpu)

    # Warm-up CUDA context/kernels; excluded from timing table.
    wr, wdt = eval_one(GPU, DEFAULT_STATES4[0])
    print('GPU_WARMUP', DEFAULT_STATES4[0], 'J', wr['merit_J'], 'sec', wdt, flush=True)

    gpu_rows = []
    comparisons = []
    for s4, cr, cdt in cpu_rows:
        gr, gdt = eval_one(GPU, s4)
        gpu_rows.append((s4, gr, gdt))
        diffs = {}
        for k in METRICS:
            a, b = float(cr[k]), float(gr[k])
            diffs[k] = {'abs': abs(a-b), 'rel': abs(a-b)/max(abs(a),1e-12)}
        comparisons.append({
            'state4': list(s4),
            'cpu_sec': cdt,
            'gpu_sec': gdt,
            'speedup': cdt / max(gdt, 1e-12),
            'cpu': compact(cr),
            'gpu': compact(gr),
            'diff': diffs,
            'config_hash_equal': cr['config_hash'] == gr['config_hash'] == CPU.CONFIG_HASH,
        })
        print('GPU', s4, 'J', gr['merit_J'], 'sec', gdt, 'speedup', cdt/max(gdt,1e-12), 'dJ', abs(float(cr['merit_J'])-float(gr['merit_J'])), flush=True)

    result = {
        'schema': 'q4096-cuda-backend-benchmark-v1',
        'gpu': ns.gpu,
        'backend': GPU.backend_info(),
        'config_hash': CPU.CONFIG_HASH,
        'warmup_sec': wdt,
        'states': comparisons,
        'mean_cpu_sec': sum(x[2] for x in cpu_rows)/len(cpu_rows),
        'mean_gpu_sec': sum(x[2] for x in gpu_rows)/len(gpu_rows),
    }
    result['mean_speedup'] = result['mean_cpu_sec']/max(result['mean_gpu_sec'],1e-12)
    result['max_abs_J_diff'] = max(x['diff']['merit_J']['abs'] for x in comparisons)
    result['max_rel_metric_diff'] = max(v['rel'] for x in comparisons for v in x['diff'].values())
    (art/'q4096_cuda_benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
