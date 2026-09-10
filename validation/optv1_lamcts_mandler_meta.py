#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENV = Path('/var/lib/cots-lamcts/venv/bin/python')
if sys.prefix != str(VENV.parent.parent) and VENV.exists():
    os.execv(str(VENV), [str(VENV), '-u', str(Path(__file__).resolve()), *sys.argv[1:]])

for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluator import market_final_direct_q4096_v4 as E
from validation import opt10_workstation_pretrain_meta as B
from validation import opt10_pretrain_completion_run as C
from core import mandler_benchmark_core as M

ART = Path(os.environ.get('COTS_JOB_ARTIFACT_DIR', 'artifacts'))
ART.mkdir(parents=True, exist_ok=True)
EXPECTED_HASH = '0dcde5089bd66deba991214558637fe906f6c2a52530619850e6c5bab0d684da'
if E.CONFIG_HASH != EXPECTED_HASH:
    raise RuntimeError(f'evaluator hash mismatch {E.CONFIG_HASH}')

# IMPORTANT: the hidden reference is used only by post-hoc audit/reporting.
# Search/proposal functions below do not receive or inspect REF_STATE.
# Example-1 components E1G1..E1G4, unflipped. Fixed scaffold uses middle gap index = 2.
REF_STATE = (0, 2, 4, 6, 2, 2, 2, 2)
FIXED_GAPS = (2, 2, 2, 2)
SPACE_N = 72 ** 4
PRETRAIN_VARIANT = 'mlm_contrast_base'
PRETRAIN_SEEDS = [17, 43, 89]


def state4_to8(s4):
    return tuple(map(int, s4)) + FIXED_GAPS


def random_state4(rng):
    return tuple(int(x) for x in rng.integers(0, 72, size=4))


def norm_moments():
    # Reproduce the exact OPT1.0 checkpoint normalization without using any J labels.
    multi = C.seq_items(C.MULTI)
    with gzip.open(C.DATA, 'rt') as f:
        ds = json.load(f)
    geom_only = [B.cots_seq(r['state']) for r in ds['train_192']]
    stat = np.concatenate([x['seq'] for x in multi] + geom_only, axis=0)
    mu = stat[:, :5].mean(0)
    sd = stat[:, :5].std(0) + 1e-5
    return multi, mu.astype(np.float32), sd.astype(np.float32)


def canonical_for_leak(seq):
    q = np.asarray(seq, np.float64).copy()
    # Translation-invariant axial coordinate for direct prescription comparison.
    stop = np.flatnonzero(q[:, 5] > 0.5)
    if len(stop):
        q[:, 4] -= q[int(stop[0]), 4]
    else:
        q[:, 4] -= q[0, 4]
    return q


def leakage_audit(multi):
    ref = canonical_for_leak(B.cots_seq(REF_STATE))
    rows = []
    exact = []
    for x in multi:
        q = canonical_for_leak(x['seq'])
        if q.shape == ref.shape:
            d = float(np.sqrt(np.mean((q - ref) ** 2)))
            if np.allclose(q, ref, rtol=0, atol=1e-4):
                exact.append({'family': x['family'], 'state': int(x.get('state', 0)), 'distance': d})
        else:
            n = min(len(q), len(ref))
            d = float(np.sqrt(np.mean((q[:n] - ref[:n]) ** 2)) + 0.25 * abs(len(q) - len(ref)))
        rows.append((d, x['family'], int(x.get('state', 0)), len(q)))
    rows.sort(key=lambda z: z[0])
    return {
        'exact_or_1e-4_matches': exact,
        'nearest': [
            {'distance': float(d), 'family': fam, 'state': st, 'surfaces': ln}
            for d, fam, st, ln in rows[:12]
        ],
        'valid_independent_pretrain': len(exact) == 0,
    }


def load_pretrained_models(device, nmodels=3):
    mods = []
    for seed in PRETRAIN_SEEDS[:nmodels]:
        ck = C.CK / f'{PRETRAIN_VARIANT}_seed{seed}.pt'
        if not ck.exists():
            raise FileNotFoundError(f'missing OPT1.0 checkpoint: {ck}')
        m = C.model_for(PRETRAIN_VARIANT).to(device)
        m.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        # Pretraining regression head was not trained; reset before target-domain online learning.
        C.seedall(seed + 700000)
        m.reg = nn.Sequential(nn.Linear(m.d, 48), nn.GELU(), nn.Linear(48, 1)).to(device)
        mods.append(m)
    return mods


def seq_batch(states4, mu, sd):
    seqs = []
    for s4 in states4:
        q = B.cots_seq(state4_to8(s4))
        seqs.append(B.normseq(q, mu, sd))
    return B.pad(seqs)


def online_train(models, states4, Js, mu, sd, device, epochs):
    X, VM = seq_batch(states4, mu, sd)
    y = np.asarray([-math.log(max(1e-9, float(j))) for j in Js], np.float32)
    ym = float(y.mean())
    ys = float(y.std() + 1e-6)
    z = (y - ym) / ys
    q75 = float(np.quantile(y, .75))
    wt = 1.0 + 1.5 * (y >= q75).astype(np.float32)
    tx = torch.from_numpy(X).to(device)
    tm = torch.from_numpy(VM).to(device)
    tz = torch.from_numpy(z).to(device)
    tw = torch.from_numpy(wt).to(device)
    n = len(states4)
    infos = []
    for mi, model in enumerate(models):
        model.train()
        # Preserve the pretrained optical representation with a lower encoder LR.
        encpars = list(model.proj.parameters()) + list(model.enc.parameters()) + [model.cls, model.pos]
        opt = torch.optim.AdamW([
            {'params': encpars, 'lr': 8e-5},
            {'params': model.reg.parameters(), 'lr': 6e-4},
        ], weight_decay=2e-3)
        rng = np.random.default_rng(901000 + mi * 1009 + n)
        losses = []
        for _ in range(epochs):
            perm = rng.permutation(n)
            el = []
            for p in range(0, n, 64):
                ii = perm[p:p+64]
                ix = torch.from_numpy(ii).to(device)
                pred = model.forward_reg(tx[ix], tm[ix])
                lr = F.smooth_l1_loss(pred, tz[ix], beta=.5, reduction='none')
                loss_reg = (lr * tw[ix]).mean()
                loss_rank = torch.zeros((), device=device)
                if len(ii) >= 4:
                    jj = ii[rng.permutation(len(ii))]
                    jx = torch.from_numpy(jj).to(device)
                    pred2 = model.forward_reg(tx[jx], tm[jx])
                    dtrue = tz[ix] - tz[jx]
                    mask = torch.abs(dtrue) > .08
                    if bool(mask.any()):
                        sgn = torch.sign(dtrue[mask])
                        loss_rank = F.softplus(-sgn * (pred[mask] - pred2[mask])).mean()
                loss = loss_reg + .22 * loss_rank
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                el.append(float(loss.detach()))
            losses.append(float(np.mean(el)))
        model.eval()
        infos.append({'model': mi, 'epochs': epochs, 'final_loss': losses[-1], 'mean_loss': float(np.mean(losses))})
    return ym, ys, infos


def predict_and_anchor(models, states4, mu, sd, device, ym, ys, batch=4096, need_anchor=True):
    X, VM = seq_batch(states4, mu, sd)
    tx = torch.from_numpy(X)
    tm = torch.from_numpy(VM)
    preds = []
    anchor = []
    for mi, model in enumerate(models):
        pp = []
        aa = []
        model.eval()
        with torch.no_grad():
            for p in range(0, len(states4), batch):
                xx = tx[p:p+batch].to(device)
                mm = tm[p:p+batch].to(device)
                h = model.encode(xx, mm)[:, 0]
                pp.append((model.reg(h).cpu().numpy() * ys + ym).astype(np.float32))
                if mi == 0 and need_anchor:
                    aa.append(h.cpu().numpy().astype(np.float32))
        preds.append(np.concatenate(pp))
        if mi == 0 and need_anchor:
            anchor = np.concatenate(aa)
    P = np.stack(preds, axis=0)
    return P.mean(0), P.std(0), anchor


def build_lamcts_tree(Z, true_y, leaf_size=36, max_depth=6):
    scaler = StandardScaler().fit(Z)
    ZZ = scaler.transform(Z)
    ranks = np.argsort(np.argsort(true_y)).astype(np.float64) / max(1, len(true_y)-1)
    leaves = []
    def rec(ix, path, depth):
        ix = np.asarray(ix, int)
        if len(ix) <= leaf_size or depth >= max_depth:
            leaves.append((ix, path)); return
        lab = (ranks[ix] >= np.median(ranks[ix])).astype(int)
        if lab.min() == lab.max():
            leaves.append((ix, path)); return
        clf = SVC(kernel='rbf', C=2.5, gamma='scale').fit(ZZ[ix], lab)
        pr = clf.predict(ZZ[ix])
        a = ix[pr == 1]; b = ix[pr == 0]
        if len(a) < 10 or len(b) < 10:
            leaves.append((ix, path)); return
        rec(a, path + [(clf, 1)], depth+1)
        rec(b, path + [(clf, 0)], depth+1)
    rec(np.arange(len(true_y)), [], 0)
    N = len(true_y)
    def ucb(item):
        ix, _ = item
        return float(ranks[ix].mean()) + .24 * math.sqrt(math.log(N+1) / max(1, len(ix)))
    best = max(leaves, key=ucb)
    return scaler, best[0], best[1], {'leaves': len(leaves), 'selected_n': int(len(best[0])), 'selected_ucb': float(ucb(best))}


def oriented_neighbors(k=12):
    a = np.asarray(M.SIGMAT, np.float64)
    a = (a - a.mean(0)) / (a.std(0) + 1e-6)
    out = []
    for i in range(len(a)):
        d = np.sum((a - a[i]) ** 2, axis=1)
        idx = np.argsort(d)
        out.append([int(x) for x in idx[1:k+1]])
    return out

NEIGH = oriented_neighbors()


def dedupe_add(out, seen_local, st):
    t = tuple(map(int, st))
    if t in seen_local:
        return False
    seen_local.add(t); out.append(t); return True


def proposal_pool(rng, seen, data_states, leaf_ix, true_y, n=60000):
    # Complete-design proposals only. No edit/action state is ever evaluated.
    out = []
    ss = set(seen)
    leaf = [data_states[int(i)] for i in leaf_ix]
    top_ix = np.argsort(true_y)[-min(32, len(true_y)):]
    tops = [data_states[int(i)] for i in top_ix]

    # 45% uniform OOD/global candidates.
    ng = int(n * .45)
    while len(out) < ng:
        dedupe_add(out, ss, random_state4(rng))

    # 30% rank-weighted recombination from the selected LA-MCTS region.
    target = int(n * .75)
    lranks = np.argsort(np.argsort(true_y[leaf_ix])).astype(np.float64)
    w = np.exp(2.2 * lranks / max(1, len(lranks)-1)); w /= w.sum()
    while len(out) < target:
        st = []
        for slot in range(4):
            parent = leaf[int(rng.choice(len(leaf), p=w))]
            v = int(parent[slot])
            if rng.random() < .24:
                v = int(rng.choice(NEIGH[v]))
            elif rng.random() < .05:
                v = int(rng.integers(0,72))
            st.append(v)
        dedupe_add(out, ss, tuple(st))

    # 25% local physical neighborhoods around currently best complete designs.
    while len(out) < n:
        base = list(tops[int(rng.integers(0, len(tops)))])
        nm = 1 if rng.random() < .72 else 2
        for slot in rng.choice(4, size=nm, replace=False):
            if rng.random() < .84:
                base[int(slot)] = int(rng.choice(NEIGH[base[int(slot)]]))
            else:
                base[int(slot)] = int(rng.integers(0,72))
        dedupe_add(out, ss, tuple(base))
    return out


def leaf_membership(anchor, scaler, path):
    z = scaler.transform(anchor)
    mask = np.ones(len(z), dtype=bool)
    for clf, side in path:
        mask &= (clf.predict(z) == side)
    return mask


def eval_child(state8, out):
    d, g = C.decode_state(state8)
    r = E.evaluate_final_with_gaps(d, g, 4096)
    ans = {
        'state': list(map(int, state8)),
        'J': float(r['merit_J']),
        'score': float(r['score']),
        'metrics': {k: float(r[k]) for k in ['efl','fno','mtf_mean','mtf_geomean_reg','mtf_p10','onaxis_pupil_fill','min_illum','dist_max','lca']},
        'costs': r['costs'], 'config_hash': r['config_hash'],
    }
    Path(out).write_text(json.dumps(ans))


def eval_many(states4, label, workers=12):
    cache = ART / 'q4096_mandler_fixed'
    cache.mkdir(exist_ok=True)
    script = Path(__file__).resolve()
    def one(s4):
        st8 = state4_to8(s4)
        h = hashlib.sha1(json.dumps(st8).encode()).hexdigest()[:16]
        p = cache / f'{h}.json'
        if p.exists():
            return json.loads(p.read_text())
        cmd = [str(VENV), '-u', str(script), '--eval-state', json.dumps(list(st8)), '--eval-out', str(p)]
        last = ''
        for _ in range(3):
            try:
                z = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900,
                    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1'})
                last = z.stdout[-4000:]
                if z.returncode == 0 and p.exists():
                    return json.loads(p.read_text())
            except subprocess.TimeoutExpired:
                last = 'timeout'
        return {'state': list(st8), 'error': last}
    res = [None] * len(states4)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(one, s): i for i, s in enumerate(states4)}
        for f in cf.as_completed(fut):
            i = fut[f]; res[i] = f.result()
            print('Q', label, i, states4[i], res[i].get('J', res[i].get('error')), flush=True)
    bad = [r for r in res if 'J' not in r]
    if bad:
        raise RuntimeError(f'{label}: {len(bad)} evaluator failures; first={bad[0]}')
    return res


def search_optv1(seed, budget, init_n, batch_n, pool_n, nmodels, workers):
    rng = np.random.default_rng(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA required for OPT1.0 LA-MCTS validation')
    multi, mu, sd = norm_moments()
    audit = leakage_audit(multi)
    if not audit['valid_independent_pretrain']:
        raise RuntimeError('pretraining leakage audit found an exact hidden-reference prescription; run would be invalid: ' + json.dumps(audit['exact_or_1e-4_matches']))
    models = load_pretrained_models(device, nmodels=nmodels)

    seen = set(); states = []
    while len(states) < init_n:
        s = random_state4(rng)
        if s not in seen:
            seen.add(s); states.append(s)
    rows = eval_many(states, f'init-s{seed}', workers)
    Js = [float(r['J']) for r in rows]
    trace = []
    round_id = 0

    while len(states) < budget:
        round_id += 1
        epochs = 60 if round_id == 1 else 34
        ym, ys, train_info = online_train(models, states, Js, mu, sd, device, epochs)
        true_y = np.asarray([-math.log(max(1e-9,j)) for j in Js], np.float64)
        _, _, Z = predict_and_anchor(models, states, mu, sd, device, ym, ys, need_anchor=True)
        scaler, leaf_ix, path, tree_info = build_lamcts_tree(Z, true_y)
        pool = proposal_pool(rng, seen, states, leaf_ix, true_y, n=pool_n)
        pmu, psd, PZ = predict_and_anchor(models, pool, mu, sd, device, ym, ys, need_anchor=True)
        elig = leaf_membership(PZ, scaler, path)
        acq = pmu + .35 * psd
        need = min(batch_n, budget - len(states))
        n_global = max(1, int(round(.12 * need)))
        n_exploit = need - n_global
        eidx = np.flatnonzero(elig)
        if len(eidx) < n_exploit:
            eidx = np.arange(len(pool))
        order = eidx[np.argsort(acq[eidx])[::-1]]
        chosen = []
        for i in order:
            s = pool[int(i)]
            if s not in seen and s not in chosen:
                chosen.append(s)
            if len(chosen) >= n_exploit:
                break
        # Persistent global/OOD exploration is deliberately unranked.
        while len(chosen) < need:
            s = random_state4(rng)
            if s not in seen and s not in chosen:
                chosen.append(s)
        rr = eval_many(chosen, f'round{round_id}-s{seed}', workers)
        for s, r in zip(chosen, rr):
            seen.add(s); states.append(s); Js.append(float(r['J'])); rows.append(r)
        bi = int(np.argmin(Js)); best4 = states[bi]
        trace.append({
            'round': round_id, 'n': len(states), 'best_J': float(Js[bi]), 'best_state4': list(best4),
            'eligible_pool': int(elig.sum()), 'pool_n': len(pool), 'tree': tree_info,
            'pred_mu_max': float(np.max(pmu)), 'pred_sigma_mean': float(np.mean(psd)),
            'train': train_info,
        })
        print('ROUND', round_id, 'n', len(states), 'bestJ', Js[bi], 'best', best4, 'eligible', int(elig.sum()), flush=True)

    order = np.argsort(Js)
    best_i = int(order[0])
    return {
        'seed': seed, 'budget': budget, 'init_n': init_n, 'batch_n': batch_n, 'pool_n': pool_n,
        'space_n': SPACE_N, 'pretrain_variant': PRETRAIN_VARIANT, 'pretrain_seeds': PRETRAIN_SEEDS[:nmodels],
        'leakage_audit': audit, 'best_state4': list(states[best_i]), 'best_state8': list(state4_to8(states[best_i])),
        'best_J': float(Js[best_i]), 'top20': [rows[int(i)] for i in order[:20]], 'trace': trace,
        'evaluated_states4': [list(s) for s in states], 'evaluated_J': [float(j) for j in Js],
    }


def random_baseline(seed, budget, workers):
    rng = np.random.default_rng(seed)
    sts = []; seen = set()
    while len(sts) < budget:
        s = random_state4(rng)
        if s not in seen:
            seen.add(s); sts.append(s)
    rows = eval_many(sts, f'random-s{seed}', workers)
    j = np.asarray([r['J'] for r in rows], float)
    i = int(np.argmin(j))
    return {'seed': seed, 'budget': budget, 'best_J': float(j[i]), 'best_state4': list(sts[i]), 'best_state8': list(state4_to8(sts[i])),
            'median_J': float(np.median(j)), 'mean_J': float(np.mean(j)), 'top20': [rows[int(k)] for k in np.argsort(j)[:20]]}


def component_hamming(a4, b4):
    return int(sum(int(x) != int(y) for x, y in zip(a4, b4)))


def main(args):
    t0 = time.time()
    ref4 = REF_STATE[:4]
    # Authoritative hidden reference label, evaluated only for post-hoc comparison.
    ref = eval_many([ref4], 'reference-posthoc', args.workers)[0]
    result = search_optv1(args.seed, args.budget, args.init, args.batch, args.pool, args.ensemble, args.workers)
    rb = None
    if not args.no_random:
        rb = random_baseline(args.seed + 1000003, args.budget, args.workers)
    best4 = tuple(result['best_state4'])
    evaluated = {tuple(x) for x in result['evaluated_states4']}
    summary = {
        'schema': 'OPT1.0-LAMCTS-Mandler-hidden-reference-v1',
        'git_commit': os.environ.get('COTS_GIT_COMMIT'),
        'evaluator_hash': E.CONFIG_HASH,
        'benchmark': {
            'search_space': '72^4 oriented component combinations, Example-1 gaps/scaffold fixed',
            'space_n': SPACE_N,
            'hidden_reference_state8': list(REF_STATE),
            'hidden_reference_J': float(ref['J']),
            'hidden_reference_metrics': ref['metrics'],
            'reference_used_by_search': False,
        },
        'optv1_lamcts': result,
        'random': rb,
        'success': {
            'exact_reference_recovered': tuple(ref4) in evaluated,
            'best_component_hamming_to_reference': component_hamming(best4, ref4),
            'best_J_over_reference_J': float(result['best_J'] / ref['J']),
            'within_5pct_reference_J': bool(result['best_J'] <= 1.05 * ref['J']),
            'within_10pct_reference_J': bool(result['best_J'] <= 1.10 * ref['J']),
        },
        'elapsed_sec': time.time() - t0,
    }
    if rb is not None:
        summary['success']['optv1_best_vs_random_best_ratio'] = float(result['best_J'] / rb['best_J'])
    out = ART / 'optv1_lamcts_mandler_meta_results.json'
    out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({
        'reference_J': ref['J'], 'best_J': result['best_J'], 'best_state4': result['best_state4'],
        'exact_reference_recovered': summary['success']['exact_reference_recovered'],
        'hamming': summary['success']['best_component_hamming_to_reference'],
        'J_over_ref': summary['success']['best_J_over_reference_J'],
        'random_best_J': None if rb is None else rb['best_J'],
        'leakage_audit': result['leakage_audit'], 'elapsed_sec': summary['elapsed_sec'],
    }, indent=2), flush=True)


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--budget', type=int, default=600)
    ap.add_argument('--init', type=int, default=96)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--pool', type=int, default=60000)
    ap.add_argument('--ensemble', type=int, default=3, choices=[1,2,3])
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--no-random', action='store_true')
    ap.add_argument('--eval-state')
    ap.add_argument('--eval-out')
    a = ap.parse_args()
    if a.eval_state:
        eval_child(tuple(json.loads(a.eval_state)), a.eval_out); return
    main(a)

if __name__ == '__main__':
    cli()
