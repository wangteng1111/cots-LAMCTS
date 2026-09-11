#!/usr/bin/env python3
"""OLMTA v1.0: OPTv1 + Transformer-only LA-MCTS + triggered adaptation.

Naming:
  OPT    = Optical Pretrained Transformer
  OPTv1  = original pretrained Transformer checkpoints (immutable initialization)
  OLMFA  = OPT-LA-MCTS Full Adaptive
  OLMTA  = OPT-LA-MCTS Triggered Adaptive

OLMTA v1.0 starts from the exact same original OPTv1 checkpoints used to
initialize OLMFA v1.0.  It NEVER loads an OLMFA-updated checkpoint.

Search semantics intentionally match OLMFA v1.0 where possible:
* hidden-reference Mandler 72^4 component search, fixed gaps/scaffold;
* Transformer-only LA-MCTS partition coordinates [mu, sigma, mu+0.35 sigma];
* same proposal mixture and 12% persistent unranked global exploration;
* Q4096 is the sole authoritative optical label.

The two deliberate changes are:
1) online OPT updates are event-triggered rather than every round;
2) Q4096 is computed by the numerically-equivalent persistent multi-GPU CUDA
   backend (compute backend only; authoritative merit/config hash unchanged).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = Path('/var/lib/cots-lamcts/venv/bin/python')
if sys.prefix != str(VENV.parent.parent) and VENV.exists():
    os.execv(str(VENV), [str(VENV), '-u', str(Path(__file__).resolve()), *sys.argv[1:]])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import numpy as np
import torch

# Reuse only stable OPTv1 representation/training/search utilities from the
# OLMFA source.  Its SVM tree/eval_many/search loop are NOT called here.
from validation import optv1_lamcts_mandler_meta as BASE
from validation import opt10_pretrain_completion_run as C
from evaluator.q4096_cuda_pool import MultiGPUQ4096Pool, EXPECTED_CONFIG_HASH

ART = Path(os.environ.get('COTS_JOB_ARTIFACT_DIR', 'artifacts'))
ART.mkdir(parents=True, exist_ok=True)

REF_STATE = BASE.REF_STATE
FIXED_GAPS = BASE.FIXED_GAPS
SPACE_N = BASE.SPACE_N
PRETRAIN_VARIANT = BASE.PRETRAIN_VARIANT
PRETRAIN_SEEDS = BASE.PRETRAIN_SEEDS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def optv1_manifest(nmodels: int) -> list[dict]:
    out = []
    for seed in PRETRAIN_SEEDS[:nmodels]:
        ck = C.CK / f'{PRETRAIN_VARIANT}_seed{seed}.pt'
        if not ck.is_file():
            raise FileNotFoundError(f'missing immutable OPTv1 checkpoint: {ck}')
        # Explicitly reject accidental OLMFA/OLMTA checkpoint substitution.
        low = str(ck).lower()
        if 'olmfa' in low or 'olmta' in low:
            raise RuntimeError(f'non-OPTv1 checkpoint path rejected: {ck}')
        out.append({'seed': int(seed), 'path': str(ck), 'sha256': sha256_file(ck)})
    return out


def transformer_coords(mu, sigma):
    mu = np.asarray(mu, np.float32)
    sigma = np.asarray(sigma, np.float32)
    return np.column_stack([mu, sigma, mu + 0.35 * sigma]).astype(np.float32)


def build_transformer_tree(T, true_y, leaf_size=36, max_depth=6):
    """LA-MCTS hierarchy using only OPT output coordinates.

    Q4096-observed reward chooses the threshold that best separates observed
    quality, but no SVM/GP/RF/KNN/scaler/surrogate is fitted.
    """
    T = np.asarray(T, np.float64)
    true_y = np.asarray(true_y, np.float64)
    ranks = np.argsort(np.argsort(true_y)).astype(np.float64) / max(1, len(true_y)-1)
    leaves = []
    split_log = []

    def rec(ix, path, depth):
        ix = np.asarray(ix, int)
        if len(ix) <= leaf_size or depth >= max_depth:
            leaves.append((ix, path)); return
        best = None
        n = len(ix)
        for axis in range(T.shape[1]):
            vals = T[ix, axis]
            if not np.isfinite(vals).all() or np.ptp(vals) < 1e-10:
                continue
            for q in (0.25, 0.375, 0.5, 0.625, 0.75):
                thr = float(np.quantile(vals, q))
                hi = ix[vals >= thr]
                lo = ix[vals < thr]
                if len(hi) < 10 or len(lo) < 10:
                    continue
                sep = abs(float(true_y[hi].mean() - true_y[lo].mean()))
                balance = 2.0 * min(len(hi), len(lo)) / n
                gain = sep * math.sqrt(max(balance, 1e-12))
                cand = (gain, axis, thr, hi, lo)
                if best is None or cand[0] > best[0]:
                    best = cand
        if best is None or best[0] <= 1e-12:
            leaves.append((ix, path)); return
        gain, axis, thr, hi, lo = best
        split_log.append({'depth': depth, 'n': int(n), 'axis': int(axis),
                          'threshold': float(thr), 'gain': float(gain),
                          'n_hi': int(len(hi)), 'n_lo': int(len(lo))})
        rec(hi, path + [(int(axis), float(thr), 1)], depth + 1)
        rec(lo, path + [(int(axis), float(thr), 0)], depth + 1)

    rec(np.arange(len(true_y)), [], 0)
    N = len(true_y)

    def ucb(item):
        ix, _ = item
        return float(ranks[ix].mean()) + .24 * math.sqrt(math.log(N+1) / max(1, len(ix)))

    best_leaf = max(leaves, key=ucb)
    info = {
        'partition': 'transformer_outputs_only',
        'coordinates': ['OPT_mu', 'OPT_sigma', 'OPT_mu_plus_0p35_sigma'],
        'external_estimator': None,
        'leaves': len(leaves),
        'selected_n': int(len(best_leaf[0])),
        'selected_ucb': float(ucb(best_leaf)),
        'splits': split_log,
    }
    return best_leaf[0], best_leaf[1], info


def leaf_membership(T, path):
    T = np.asarray(T, np.float64)
    mask = np.ones(len(T), dtype=bool)
    for axis, thr, side in path:
        if side == 1:
            mask &= (T[:, axis] >= thr)
        else:
            mask &= (T[:, axis] < thr)
    return mask


def spearman_simple(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


class TriggerController:
    """Event-triggered OPT adaptation policy for OLMTA v1.0.

    Signals intentionally depend only on information available online:
    ensemble uncertainty, prediction-vs-Q4096 ranking, search stagnation,
    eligible-region collapse, and accumulated new authoritative labels.
    """
    def __init__(self, pool_n: int):
        self.pool_n = int(pool_n)
        self.last_update_n = 0
        self.last_best_n = 0
        self.sigma_ref_p90 = 1e-6
        self.pred_since_update = []
        self.sigma_since_update = []
        self.true_since_update = []
        self.update_count = 0

        # v1.0 fixed policy.  These are part of the experiment definition.
        self.min_signal_labels = 64
        self.min_structural_labels = 256
        self.max_labels_without_update = 1024
        self.stagnation_evals = 768
        self.uncertainty_multiplier = 1.50
        self.uncertainty_frequency = 0.35
        self.ranking_rho_min = 0.25
        self.eligible_floor = max(64, int(round(0.002 * self.pool_n)))
        self.recent_window = 128

    def policy_dict(self):
        return {
            'min_signal_labels': self.min_signal_labels,
            'min_structural_labels': self.min_structural_labels,
            'max_labels_without_update': self.max_labels_without_update,
            'stagnation_evals': self.stagnation_evals,
            'uncertainty_multiplier': self.uncertainty_multiplier,
            'uncertainty_frequency': self.uncertainty_frequency,
            'ranking_rho_min': self.ranking_rho_min,
            'eligible_floor': self.eligible_floor,
            'recent_window': self.recent_window,
        }

    def reset_after_update(self, n, best_n, sigma_ref_p90):
        self.last_update_n = int(n)
        self.last_best_n = int(best_n)
        self.sigma_ref_p90 = max(float(sigma_ref_p90), 1e-6)
        self.pred_since_update.clear()
        self.sigma_since_update.clear()
        self.true_since_update.clear()
        self.update_count += 1

    def note_batch(self, pred_mu, pred_sigma, true_reward):
        self.pred_since_update.extend(map(float, pred_mu))
        self.sigma_since_update.extend(map(float, pred_sigma))
        self.true_since_update.extend(map(float, true_reward))

    def decide(self, n, best_n, eligible):
        new_n = int(n) - self.last_update_n
        no_best = int(n) - int(best_n)
        k = min(self.recent_window, len(self.pred_since_update))
        pred = np.asarray(self.pred_since_update[-k:], float) if k else np.empty(0)
        sig = np.asarray(self.sigma_since_update[-k:], float) if k else np.empty(0)
        tru = np.asarray(self.true_since_update[-k:], float) if k else np.empty(0)
        rho = spearman_simple(pred, tru) if k >= self.min_signal_labels else float('nan')
        hi_frac = float(np.mean(sig > self.uncertainty_multiplier * self.sigma_ref_p90)) if k else 0.0

        reasons = []
        if new_n >= self.min_signal_labels and hi_frac >= self.uncertainty_frequency:
            reasons.append('uncertainty_frequency')
        if new_n >= self.min_signal_labels and np.isfinite(rho) and rho < self.ranking_rho_min:
            reasons.append('ranking_degradation')
        if new_n >= self.min_structural_labels and no_best >= self.stagnation_evals:
            reasons.append('stagnation')
        if new_n >= self.min_structural_labels and int(eligible) < self.eligible_floor:
            reasons.append('eligible_collapse')
        if new_n >= self.max_labels_without_update:
            reasons.append('label_accumulation')

        stats = {
            'new_labels_since_update': int(new_n),
            'evals_since_best': int(no_best),
            'recent_n': int(k),
            'recent_spearman': None if not np.isfinite(rho) else float(rho),
            'recent_high_uncertainty_fraction': float(hi_frac),
            'sigma_ref_p90': float(self.sigma_ref_p90),
            'eligible_pool': int(eligible),
            'eligible_floor': int(self.eligible_floor),
        }
        return bool(reasons), reasons, stats


def predict(models, states4, mu, sd, device, ym, ys):
    pmu, psd, _ = BASE.predict_and_anchor(models, states4, mu, sd, device, ym, ys, need_anchor=False)
    return pmu, psd


def sigma_reference(models, states, mu, sd, device, ym, ys):
    _, sig = predict(models, states, mu, sd, device, ym, ys)
    return float(np.quantile(np.asarray(sig, float), .90))


def eval_states(pool, states4, label):
    rows = pool.evaluate_states4(states4, fixed_gap_indices=FIXED_GAPS)
    if len(rows) != len(states4):
        raise RuntimeError(f'{label}: evaluator returned {len(rows)} rows for {len(states4)} states')
    for i, (s, r) in enumerate(zip(states4, rows)):
        if r.get('config_hash') != EXPECTED_CONFIG_HASH or 'J' not in r:
            raise RuntimeError(f'{label}: invalid evaluator row {i}: {r}')
        print('Q', label, i, tuple(s), r['J'], 'gpu', r.get('gpu', 'cache'), flush=True)
    return rows


def write_progress(obj):
    p = ART / 'olmta_v1_progress.json'
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, p)


def search_olmta(seed, budget, init_n, batch_n, pool_n, nmodels, qpool, device):
    rng = np.random.default_rng(seed)
    multi, mu, sd = BASE.norm_moments()
    audit = BASE.leakage_audit(multi)
    if not audit['valid_independent_pretrain']:
        raise RuntimeError('OPTv1 pretraining leakage audit failed: ' + json.dumps(audit['exact_or_1e-4_matches']))

    manifest = optv1_manifest(nmodels)
    models = BASE.load_pretrained_models(device, nmodels=nmodels)
    print('OPTV1_INIT', json.dumps(manifest), flush=True)

    seen = set(); states = []
    while len(states) < init_n:
        s = BASE.random_state4(rng)
        if s not in seen:
            seen.add(s); states.append(s)
    rows = eval_states(qpool, states, f'init-s{seed}')
    Js = [float(r['J']) for r in rows]

    # Mandatory target-domain calibration, identical in spirit to OLMFA round 1.
    # Crucially, it starts from immutable OPTv1, not an OLMFA-updated checkpoint.
    ym, ys, init_train = BASE.online_train(models, states, Js, mu, sd, device, epochs=60)
    best_i = int(np.argmin(Js)); best_n = len(states); best_J = float(Js[best_i])
    controller = TriggerController(pool_n)
    sig_ref = sigma_reference(models, states, mu, sd, device, ym, ys)
    controller.reset_after_update(len(states), best_n, sig_ref)
    trigger_events = [{
        'round': 0, 'n': len(states), 'type': 'mandatory_initial_calibration',
        'reasons': ['initial_96_q4096_labels'], 'epochs': 60,
        'sigma_ref_p90': sig_ref, 'train': init_train,
    }]

    trace = []
    round_id = 0
    while len(states) < budget:
        round_id += 1
        true_y = np.asarray([-math.log(max(1e-9, j)) for j in Js], np.float64)
        omu, osig = predict(models, states, mu, sd, device, ym, ys)
        OT = transformer_coords(omu, osig)
        leaf_ix, path, tree_info = build_transformer_tree(OT, true_y)

        candidates = BASE.proposal_pool(rng, seen, states, leaf_ix, true_y, n=pool_n)
        pmu, psd = predict(models, candidates, mu, sd, device, ym, ys)
        PT = transformer_coords(pmu, psd)
        elig = leaf_membership(PT, path)
        acq = pmu + .35 * psd

        need = min(batch_n, budget - len(states))
        n_global = max(1, int(round(.12 * need)))
        n_exploit = need - n_global
        eidx = np.flatnonzero(elig)
        if len(eidx) < n_exploit:
            eidx = np.arange(len(candidates))
        order = eidx[np.argsort(acq[eidx])[::-1]]

        chosen = []
        chosen_pred = []
        chosen_sig = []
        for ix in order:
            s = candidates[int(ix)]
            if s not in seen and s not in chosen:
                chosen.append(s); chosen_pred.append(float(pmu[int(ix)])); chosen_sig.append(float(psd[int(ix)]))
            if len(chosen) >= n_exploit:
                break

        # Keep OLMFA v1.0's 12% unranked global exploration unchanged.
        while len(chosen) < need:
            s = BASE.random_state4(rng)
            if s not in seen and s not in chosen:
                gm, gs = predict(models, [s], mu, sd, device, ym, ys)
                chosen.append(s); chosen_pred.append(float(gm[0])); chosen_sig.append(float(gs[0]))

        rr = eval_states(qpool, chosen, f'round{round_id}-s{seed}')
        batch_true = []
        old_best = best_J
        for s, r in zip(chosen, rr):
            j = float(r['J'])
            seen.add(s); states.append(s); Js.append(j); rows.append(r)
            batch_true.append(-math.log(max(1e-9, j)))
            if j < best_J:
                best_J = j; best_n = len(states)

        controller.note_batch(chosen_pred, chosen_sig, batch_true)
        do_update, reasons, trigger_stats = controller.decide(len(states), best_n, int(elig.sum()))
        train_info = None
        if do_update:
            ym, ys, train_info = BASE.online_train(models, states, Js, mu, sd, device, epochs=34)
            sig_ref = sigma_reference(models, states, mu, sd, device, ym, ys)
            controller.reset_after_update(len(states), best_n, sig_ref)
            event = {
                'round': round_id, 'n': len(states), 'type': 'triggered_update',
                'reasons': reasons, 'epochs': 34, 'trigger_stats': trigger_stats,
                'sigma_ref_p90_after_update': sig_ref, 'train': train_info,
            }
            trigger_events.append(event)
            print('TRIGGER_UPDATE', round_id, len(states), reasons, json.dumps(trigger_stats), flush=True)

        bi = int(np.argmin(Js)); best4 = states[bi]
        rec = {
            'round': round_id, 'n': len(states), 'best_J': float(Js[bi]), 'best_state4': list(best4),
            'improved_this_round': bool(float(Js[bi]) < old_best),
            'eligible_pool': int(elig.sum()), 'pool_n': len(candidates), 'tree': tree_info,
            'pred_mu_max': float(np.max(pmu)), 'pred_sigma_mean': float(np.mean(psd)),
            'triggered_update': bool(do_update), 'trigger_reasons': reasons,
            'trigger_stats': trigger_stats,
        }
        trace.append(rec)
        print('ROUND', round_id, 'n', len(states), 'bestJ', Js[bi], 'best', best4,
              'eligible', int(elig.sum()), 'update', int(do_update),
              'updates_total', controller.update_count, flush=True)
        write_progress({
            'schema': 'OLMTA-v1.0-progress', 'round': round_id, 'n': len(states),
            'budget': budget, 'best_J': float(Js[bi]), 'best_state4': list(best4),
            'updates_total_including_initial': controller.update_count,
            'last_trigger_reasons': reasons, 'eligible_pool': int(elig.sum()),
        })

    order = np.argsort(Js)
    best_i = int(order[0])
    return {
        'schema': 'OLMTA-v1.0-search',
        'seed': seed, 'budget': budget, 'init_n': init_n, 'batch_n': batch_n, 'pool_n': pool_n,
        'space_n': SPACE_N,
        'initial_network': 'OPTv1-original-pretrained-checkpoints',
        'optv1_pretrain_variant': PRETRAIN_VARIANT,
        'optv1_checkpoint_manifest': manifest,
        'leakage_audit': audit,
        'trigger_policy': controller.policy_dict(),
        'update_count_including_initial': controller.update_count,
        'trigger_events': trigger_events,
        'best_state4': list(states[best_i]),
        'best_state8': list(BASE.state4_to8(states[best_i])),
        'best_J': float(Js[best_i]),
        'top20': [rows[int(i)] for i in order[:20]],
        'trace': trace,
        'evaluated_states4': [list(s) for s in states],
        'evaluated_J': [float(j) for j in Js],
    }


def component_hamming(a4, b4):
    return int(sum(int(x) != int(y) for x, y in zip(a4, b4)))


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for OLMTA v1.0')
    t0 = time.time()
    device = torch.device(f'cuda:{args.opt_gpu}')
    gpus = tuple(int(x) for x in args.eval_gpus.split(','))
    print('OLMTA_V1_CONFIG', json.dumps({
        'seed': args.seed, 'budget': args.budget, 'init': args.init, 'batch': args.batch,
        'pool': args.pool, 'ensemble': args.ensemble, 'opt_gpu': args.opt_gpu,
        'eval_gpus': gpus, 'eval_workers_per_gpu': args.eval_workers_per_gpu,
        'initial_network': 'immutable original OPTv1',
    }), flush=True)

    with MultiGPUQ4096Pool(gpus=gpus, workers_per_gpu=args.eval_workers_per_gpu) as qpool:
        if not args.no_warmup:
            qpool.warmup()
        result = search_olmta(args.seed, args.budget, args.init, args.batch, args.pool,
                              args.ensemble, qpool, device)
        # Hidden reference is evaluated only after search for post-hoc reporting.
        ref = eval_states(qpool, [REF_STATE[:4]], 'reference-posthoc')[0]

    best4 = tuple(result['best_state4'])
    evaluated = {tuple(x) for x in result['evaluated_states4']}
    summary = {
        'schema': 'OLMTA-v1.0-Mandler-hidden-reference',
        'git_commit': os.environ.get('COTS_GIT_COMMIT'),
        'naming': {
            'OPT': 'Optical Pretrained Transformer',
            'OPTv1': 'first trained OPT network/checkpoint family',
            'OLMFA': 'OPT-LA-MCTS Full Adaptive',
            'OLMTA': 'OPT-LA-MCTS Triggered Adaptive',
        },
        'evaluator': {
            'authoritative': 'Q4096', 'config_hash': EXPECTED_CONFIG_HASH,
            'backend': 'persistent multi-GPU CUDA float64, numerically equivalent to CPU v4',
            'gpus': list(gpus), 'workers_per_gpu': args.eval_workers_per_gpu,
        },
        'benchmark': {
            'search_space': '72^4 oriented component combinations, Example-1 gaps/scaffold fixed',
            'space_n': SPACE_N,
            'hidden_reference_state8': list(REF_STATE),
            'hidden_reference_J': float(ref['J']),
            'hidden_reference_metrics': ref['metrics'],
            'reference_used_by_search': False,
        },
        'olmta_v1': result,
        'success': {
            'exact_reference_recovered': tuple(REF_STATE[:4]) in evaluated,
            'best_component_hamming_to_reference': component_hamming(best4, REF_STATE[:4]),
            'best_J_over_reference_J': float(result['best_J'] / ref['J']),
            'within_5pct_reference_J': bool(result['best_J'] <= 1.05 * ref['J']),
            'within_10pct_reference_J': bool(result['best_J'] <= 1.10 * ref['J']),
        },
        'elapsed_sec': time.time() - t0,
    }
    out = ART / 'olmta_v1_mandler_meta_results.json'
    out.write_text(json.dumps(summary, indent=2) + '\n')
    print('OLMTA_FINAL', json.dumps({
        'best_J': result['best_J'], 'best_state4': result['best_state4'],
        'updates_total_including_initial': result['update_count_including_initial'],
        'J_over_ref': summary['success']['best_J_over_reference_J'],
        'elapsed_sec': summary['elapsed_sec'],
    }), flush=True)


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--budget', type=int, default=10000)
    ap.add_argument('--init', type=int, default=96)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--pool', type=int, default=60000)
    ap.add_argument('--ensemble', type=int, default=3, choices=[1,2,3])
    ap.add_argument('--opt-gpu', type=int, default=0)
    ap.add_argument('--eval-gpus', default='0,1')
    ap.add_argument('--eval-workers-per-gpu', type=int, default=4)
    ap.add_argument('--no-warmup', action='store_true')
    args = ap.parse_args()
    main(args)


if __name__ == '__main__':
    cli()
