#!/usr/bin/env python3
"""Run the hidden-reference Mandler validation with OPT1.0 as the ONLY learned estimator.

This wrapper deliberately removes sklearn/SVM/StandardScaler from the executable
source and replaces LA-MCTS region partitioning with a deterministic hierarchical
partition over OPT1.0 ensemble outputs only:
    t0 = ensemble predicted reward E[-log J]
    t1 = ensemble predictive std
    t2 = t0 + 0.35*t1 (the same Transformer acquisition coordinate)

Q4096 remains the sole authoritative optical label. The tree uses Q4096-observed
rewards only for node statistics / choosing a threshold among Transformer-derived
coordinates; no additional surrogate or classifier is fitted.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'optv1_lamcts_mandler_meta.py'
TMP = HERE / '_optv1_transformer_only_lamcts_runtime.py'
text = SRC.read_text()

# Remove every external estimator/classifier dependency from the executable text.
text = text.replace('from sklearn.preprocessing import StandardScaler\n', '')
text = text.replace('from sklearn.svm import SVC\n', '')


def replace_block(src: str, begin: str, end: str, new: str) -> str:
    i = src.index(begin)
    j = src.index(end, i)
    return src[:i] + new.rstrip() + '\n\n\n' + src[j:]


predict_fn = r'''def predict_and_anchor(models, states4, mu, sd, device, ym, ys, batch=4096, need_anchor=True):
    # Third return is intentionally NOT a separately learned embedding classifier input.
    # It is a compact set of coordinates derived directly from the pretrained/fine-tuned
    # OPT1.0 ensemble's own scalar predictions and uncertainty.
    X, VM = seq_batch(states4, mu, sd)
    tx = torch.from_numpy(X)
    tm = torch.from_numpy(VM)
    preds = []
    for model in models:
        pp = []
        model.eval()
        with torch.no_grad():
            for p in range(0, len(states4), batch):
                xx = tx[p:p+batch].to(device)
                mm = tm[p:p+batch].to(device)
                h = model.encode(xx, mm)[:, 0]
                pp.append((model.reg(h).cpu().numpy() * ys + ym).astype(np.float32))
        preds.append(np.concatenate(pp))
    P = np.stack(preds, axis=0)
    pmu = P.mean(0).astype(np.float32)
    psd = P.std(0).astype(np.float32)
    transformer_coords = np.column_stack([pmu, psd, pmu + 0.35 * psd]).astype(np.float32)
    return pmu, psd, transformer_coords'''


tree_fn = r'''def build_lamcts_tree(T, true_y, leaf_size=36, max_depth=6):
    # Transformer-only LA-MCTS partition.
    # T contains ONLY OPT1.0-derived coordinates [mu, sigma, mu+0.35sigma].
    # No SVM/GP/RF/KNN/linear model/clusterer/scaler is fitted here.
    T = np.asarray(T, np.float64)
    true_y = np.asarray(true_y, np.float64)
    ranks = np.argsort(np.argsort(true_y)).astype(np.float64) / max(1, len(true_y)-1)
    leaves = []
    split_log = []

    def rec(ix, path, depth):
        ix = np.asarray(ix, int)
        if len(ix) <= leaf_size or depth >= max_depth:
            leaves.append((ix, path)); return

        # Search only simple thresholds on coordinates emitted by OPT1.0 itself.
        # The Q4096 reward is used to choose which such threshold best separates
        # high- and low-reward observed samples; it does not train another estimator.
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
        'coordinates': ['OPT1_mu', 'OPT1_sigma', 'OPT1_mu_plus_0p35_sigma'],
        'external_estimator': None,
        'leaves': len(leaves),
        'selected_n': int(len(best_leaf[0])),
        'selected_ucb': float(ucb(best_leaf)),
        'splits': split_log,
    }
    return None, best_leaf[0], best_leaf[1], info'''


membership_fn = r'''def leaf_membership(T, unused_scaler, path):
    # Membership is evaluated directly in OPT1.0 output coordinates.
    T = np.asarray(T, np.float64)
    mask = np.ones(len(T), dtype=bool)
    for axis, thr, side in path:
        if side == 1:
            mask &= (T[:, axis] >= thr)
        else:
            mask &= (T[:, axis] < thr)
    return mask'''

text = replace_block(text, 'def predict_and_anchor(', 'def build_lamcts_tree(', predict_fn)
text = replace_block(text, 'def build_lamcts_tree(', 'def oriented_neighbors(', tree_fn)
text = replace_block(text, 'def leaf_membership(', 'def eval_child(', membership_fn)

# Keep per-evaluation cache off the uploaded artifact directory and make it persistent
# across workstation jobs, so identical Q4096 states can be reused safely.
text = text.replace("cache = ART / 'q4096_mandler_fixed'",
                    "cache = Path('/var/lib/cots-lamcts/q4096_mandler_fixed_shared')")
text = text.replace("'OPT1.0-LAMCTS-Mandler-hidden-reference-v1'",
                    "'OPT1.0-TransformerOnly-LAMCTS-Mandler-hidden-reference-v2'")
text = text.replace("ART / 'optv1_lamcts_mandler_meta_results.json'",
                    "ART / 'optv1_transformer_only_lamcts_mandler_results.json'")

# Static fail-closed audit: executable source must contain no old estimator stack.
for forbidden in ('sklearn', 'SVC(', 'StandardScaler', 'RandomForest', 'GaussianProcess', 'KNeighbors'):
    if forbidden in text:
        raise RuntimeError(f'forbidden non-Transformer estimator survived source rewrite: {forbidden}')

TMP.write_text(text)
os.execv(sys.executable, [sys.executable, '-u', str(TMP), *sys.argv[1:]])
