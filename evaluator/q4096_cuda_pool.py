"""Persistent multi-GPU pool for the authoritative Q4096 evaluator.

Designed for OLMFA/OLMTA search loops.  CUDA worker processes are spawned once
and reused across MCTS rounds, avoiding Python/Torch/CUDA startup per lens.
The pool is a compute backend only: it does not alter the Q4096 optical merit.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable, Sequence

EXPECTED_CONFIG_HASH = "0dcde5089bd66deba991214558637fe906f6c2a52530619850e6c5bab0d684da"
DEFAULT_CACHE = Path('/var/lib/cots-lamcts/q4096_mandler_fixed_shared')
GAP_GRIDS = (
    (0.20, 0.29, 0.38, 0.47, 0.56),
    (7.50, 8.32, 9.14, 9.96, 10.78),
    (11.00, 12.18, 13.36, 14.54, 15.72),
    (0.20, 0.29, 0.38, 0.47, 0.56),
)
METRIC_KEYS = ('efl','fno','mtf_mean','mtf_geomean_reg','mtf_p10','onaxis_pupil_fill','min_illum','dist_max','lca')

_WORKER_EVAL = None
_WORKER_GPU = None


def _decode_state8(st8: Sequence[int]):
    st = tuple(map(int, st8))
    if len(st) != 8:
        raise ValueError('state8 must contain 8 integers')
    design = tuple((x // 2, bool(x % 2)) for x in st[:4])
    gaps = tuple(GAP_GRIDS[i][st[4+i]] for i in range(4))
    return design, gaps


def state4_to8(st4: Sequence[int], fixed_gap_indices=(2,2,2,2)) -> tuple[int, ...]:
    st = tuple(map(int, st4))
    if len(st) != 4:
        raise ValueError('state4 must contain 4 integers')
    return st + tuple(map(int, fixed_gap_indices))


def cache_key(st8: Sequence[int]) -> str:
    return hashlib.sha1(json.dumps(tuple(map(int, st8))).encode()).hexdigest()[:16]


def _init_worker(gpu: int) -> None:
    global _WORKER_EVAL, _WORKER_GPU
    _WORKER_GPU = int(gpu)
    os.environ['COTS_EVAL_CUDA_DEVICE'] = str(_WORKER_GPU)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    from evaluator import market_final_direct_q4096_v4_cuda as G
    G.set_cuda_device(_WORKER_GPU)
    _WORKER_EVAL = G


def _worker_eval(st8: Sequence[int]) -> dict:
    if _WORKER_EVAL is None:
        raise RuntimeError('Q4096 CUDA worker was not initialized')
    st = tuple(map(int, st8))
    d, gaps = _decode_state8(st)
    r = _WORKER_EVAL.evaluate_final_with_gaps(d, gaps, 4096)
    _WORKER_EVAL.synchronize()
    return {
        'state': list(st),
        'J': float(r['merit_J']),
        'score': float(r['score']),
        'metrics': {k: float(r[k]) for k in METRIC_KEYS},
        'costs': r['costs'],
        'config_hash': r['config_hash'],
        'backend': _WORKER_EVAL.BACKEND,
        'gpu': int(_WORKER_GPU),
    }


def _read_cache(path: Path) -> dict | None:
    try:
        if not path.is_file():
            return None
        r = json.loads(path.read_text())
        if r.get('config_hash') != EXPECTED_CONFIG_HASH or 'J' not in r:
            return None
        return r
    except Exception:
        return None


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


class MultiGPUQ4096Pool:
    def __init__(self, gpus=(0,1), workers_per_gpu: int = 1, cache_dir: str | Path = DEFAULT_CACHE, write_cache: bool = True):
        self.gpus = tuple(map(int, gpus))
        if not self.gpus:
            raise ValueError('at least one GPU is required')
        self.workers_per_gpu = max(1, int(workers_per_gpu))
        self.cache_dir = Path(cache_dir)
        self.write_cache = bool(write_cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ctx = mp.get_context('spawn')
        self.executors = [
            cf.ProcessPoolExecutor(
                max_workers=self.workers_per_gpu,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(gpu,),
            )
            for gpu in self.gpus
        ]
        self._rr = 0

    @property
    def capacity(self) -> int:
        return len(self.gpus) * self.workers_per_gpu

    def _submit(self, st8):
        ex = self.executors[self._rr % len(self.executors)]
        self._rr += 1
        return ex.submit(_worker_eval, tuple(map(int, st8)))

    def evaluate_states8(self, states8: Iterable[Sequence[int]], bypass_cache: bool = False) -> list[dict]:
        states = [tuple(map(int, s)) for s in states8]
        result: list[dict | None] = [None] * len(states)
        positions: dict[tuple[int,...], list[int]] = {}
        for i, st in enumerate(states):
            positions.setdefault(st, []).append(i)

        futures = {}
        for st, pos in positions.items():
            p = self.cache_dir / f'{cache_key(st)}.json'
            cached = None if bypass_cache else _read_cache(p)
            if cached is not None:
                for i in pos:
                    result[i] = cached
                continue
            fut = self._submit(st)
            futures[fut] = (st, pos, p)

        for fut in cf.as_completed(futures):
            st, pos, p = futures[fut]
            r = fut.result()
            if r.get('config_hash') != EXPECTED_CONFIG_HASH:
                raise RuntimeError(f'Q4096 config hash mismatch for {st}: {r.get("config_hash")}')
            if self.write_cache and not bypass_cache:
                _atomic_write_json(p, r)
            for i in pos:
                result[i] = r

        return [r for r in result if r is not None]

    def evaluate_states4(self, states4: Iterable[Sequence[int]], fixed_gap_indices=(2,2,2,2), bypass_cache: bool = False) -> list[dict]:
        return self.evaluate_states8([state4_to8(s, fixed_gap_indices) for s in states4], bypass_cache=bypass_cache)

    def warmup(self, state4=(0,2,4,6), fixed_gap_indices=(2,2,2,2)) -> None:
        # Submit enough independent tasks to force all worker processes to start.
        base = list(map(int, state4))
        states = []
        for i in range(self.capacity):
            s = list(base)
            s[0] = (s[0] + 2*i) % 72
            states.append(state4_to8(s, fixed_gap_indices))
        self.evaluate_states8(states, bypass_cache=True)

    def close(self) -> None:
        for ex in self.executors:
            ex.shutdown(wait=True, cancel_futures=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
