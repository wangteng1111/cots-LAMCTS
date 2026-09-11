"""CUDA backend for market_final_direct_q4096_v4.

The authoritative optical definition, Q4096 sampling, merit J, and CONFIG_HASH
remain owned by market_final_direct_q4096_v4.  This module only replaces the
large vectorized ray/eikonal/direct-OTF arrays with Torch CUDA float64 kernels.
Surface construction, cardinal optics, chief rays, Sobol samples, merit terms,
and all scalar reporting stay identical to v4.

Use COTS_EVAL_CUDA_DEVICE (default 0) to select the GPU.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch
from scipy.stats import qmc

from evaluator import market_final_direct_q4096_v4 as E

BACKEND = "torch_cuda_float64_exact_spherical_sag_safe_v2"
CONFIG = E.CONFIG
CONFIG_HASH = E.CONFIG_HASH
QMC_SAMPLES = E.QMC_SAMPLES

_DEVICE_INDEX = int(os.environ.get("COTS_EVAL_CUDA_DEVICE", "0"))
_DEVICE = torch.device(f"cuda:{_DEVICE_INDEX}" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.float64


def set_cuda_device(index: int) -> None:
    global _DEVICE_INDEX, _DEVICE
    _DEVICE_INDEX = int(index)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    _DEVICE = torch.device(f"cuda:{_DEVICE_INDEX}")
    torch.cuda.set_device(_DEVICE)


def backend_info() -> dict[str, Any]:
    if _DEVICE.type != "cuda":
        return {"backend": BACKEND, "device": str(_DEVICE), "cuda": False}
    p = torch.cuda.get_device_properties(_DEVICE)
    return {
        "backend": BACKEND,
        "device": str(_DEVICE),
        "cuda": True,
        "name": p.name,
        "total_memory": int(p.total_memory),
        "torch": torch.__version__,
    }


def _dir(ax: float, ay: float = 0.0):
    tx, ty = math.tan(ax), math.tan(ay)
    dz = 1.0 / math.sqrt(1.0 + tx * tx + ty * ty)
    return dz, tx * dz, ty * dz


def _sag_ds_t(R: float, r: torch.Tensor):
    if math.isinf(R):
        return torch.zeros_like(r), torch.zeros_like(r), torch.ones_like(r, dtype=torch.bool)
    q = R * R - r * r
    ok = q > 0
    sq = torch.sqrt(torch.clamp(q, min=1e-30))
    sign = 1.0 if R >= 0 else -1.0
    return R - sign * sq, sign * r / sq, ok


def _safe_incident_start_t(surfs, ax: float, xv: torch.Tensor, yv: torch.Tensor):
    s0 = surfs[0]
    finite_R = [abs(s.R) for s in surfs if math.isfinite(s.R)]
    aperture_scale = min(60.0, max(20.0, 0.45 * min(finite_R) if finite_R else 20.0))
    z0 = s0.z - max(10.0, 2.5 * aperture_scale)
    dz0, dx0, dy0 = _dir(ax)
    dt = (s0.z - z0) / dz0
    return z0, xv - dt * dx0, yv - dt * dy0, dz0, dx0, dy0


def _surf_step_t(z, x, y, dz, dx, dy, s, opl=None, maxit: int = 14):
    tv = (s.z - z) / dz
    xv = x + tv * dx
    yv = y + tv * dy
    rv = torch.hypot(xv, yv)
    sg0, _, ok = _sag_ds_t(s.R, rv)
    t = torch.clamp((s.z + sg0 - z) / dz, min=1e-10)
    valid = ok & torch.isfinite(t)

    for _ in range(maxit):
        xx = x + t * dx
        yy = y + t * dy
        r = torch.hypot(xx, yy)
        sg, ds, ok2 = _sag_ds_t(s.R, r)
        radial = torch.where(r > 1e-14, (xx * dx + yy * dy) / r, torch.zeros_like(r))
        f = z + t * dz - s.z - sg
        df = dz - ds * radial
        good = valid & ok2 & (torch.abs(df) > 1e-14)
        step = torch.where(good, f / df, torch.zeros_like(t))
        t = t - step
        valid = valid & good
        if not bool(good.any().item()):
            break
        if float(torch.max(torch.abs(step[good])).item()) < 1e-10:
            break

    valid = valid & (t > 0) & torch.isfinite(t)
    zn = z + t * dz
    xn = x + t * dx
    yn = y + t * dy
    if opl is not None:
        opl = opl + s.n1 * torch.sqrt((zn - z) ** 2 + (xn - x) ** 2 + (yn - y) ** 2)

    r = torch.hypot(xn, yn)
    _, ds, okn = _sag_ds_t(s.R, r)
    nz = torch.ones_like(r)
    rr = r > 1e-14
    nx = torch.where(rr, -ds * xn / r, torch.zeros_like(r))
    ny = torch.where(rr, -ds * yn / r, torch.zeros_like(r))
    nn = torch.sqrt(nz * nz + nx * nx + ny * ny)
    nz, nx, ny = nz / nn, nx / nn, ny / nn
    dot = dz * nz + dx * nx + dy * ny
    flip = dot > 0
    nz = torch.where(flip, -nz, nz)
    nx = torch.where(flip, -nx, nx)
    ny = torch.where(flip, -ny, ny)

    eta = s.n1 / s.n2
    cosi = -(dz * nz + dx * nx + dy * ny)
    kk = 1.0 - eta * eta * (1.0 - cosi * cosi)
    valid = valid & okn & (kk >= 0)
    root = torch.sqrt(torch.clamp(kk, min=0.0))
    qq = eta * cosi - root
    oz = eta * dz + qq * nz
    ox = eta * dx + qq * nx
    oy = eta * dy + qq * ny
    norm = torch.sqrt(oz * oz + ox * ox + oy * oy)
    dz2, dx2, dy2 = oz / norm, ox / norm, oy / norm
    eps = 1e-8
    return zn + dz2 * eps, xn + dx2 * eps, yn + dy2 * eps, dz2, dx2, dy2, valid, opl


def _trace_stop_t(surfs, ax: float, x0: torch.Tensor, y0: torch.Tensor, stop_z: float = 0.0):
    z0, xs, ys, dz0, dx0, dy0 = _safe_incident_start_t(surfs, ax, x0, y0)
    z = torch.full_like(x0, z0)
    x, y = xs, ys
    dz = torch.full_like(x0, dz0)
    dx = torch.full_like(x0, dx0)
    dy = torch.full_like(x0, dy0)
    valid = torch.ones_like(x0, dtype=torch.bool)
    for s in [ss for ss in surfs if ss.z < stop_z - 1e-10]:
        z, x, y, dz, dx, dy, ok, _ = _surf_step_t(z, x, y, dz, dx, dy, s, None)
        valid = valid & ok
    tt = (stop_z - z) / dz
    xs = x + tt * dx
    ys = y + tt * dy
    valid = valid & torch.isfinite(xs) & torch.isfinite(ys) & (tt > 0)
    nan = torch.full_like(xs, float("nan"))
    return torch.stack([torch.where(valid, xs, nan), torch.where(valid, ys, nan)], dim=1), valid


def _solve_input_t(surfs, ax: float, targets: torch.Tensor, stop_z: float = 0.0, maxit: int = 7):
    fm = E.M.front_matrix(surfs, stop_z)
    N = targets.shape[0]
    if fm is None:
        return torch.full_like(targets, float("nan")), torch.zeros(N, dtype=torch.bool, device=targets.device), torch.full((N,), float("inf"), dtype=targets.dtype, device=targets.device)
    A, B = float(fm[0, 0]), float(fm[0, 1])
    if abs(A) < 1e-12:
        return torch.full_like(targets, float("nan")), torch.zeros(N, dtype=torch.bool, device=targets.device), torch.full((N,), float("inf"), dtype=targets.dtype, device=targets.device)

    q = torch.empty_like(targets)
    q[:, 0] = (targets[:, 0] - B * ax) / A
    q[:, 1] = targets[:, 1] / A
    conv = torch.zeros(N, dtype=torch.bool, device=targets.device)
    for _ in range(maxit):
        v, ok = _trace_stop_t(surfs, ax, q[:, 0], q[:, 1], stop_z)
        f = v - targets
        err = torch.linalg.vector_norm(f, dim=1)
        conv = conv | (ok & (err < 2e-6))
        active = ok & ~conv
        if not bool(active.any().item()):
            break
        h = 1e-3
        vx, okx = _trace_stop_t(surfs, ax, q[:, 0] + h, q[:, 1], stop_z)
        vy, oky = _trace_stop_t(surfs, ax, q[:, 0], q[:, 1] + h, stop_z)
        J00 = (vx[:, 0] - v[:, 0]) / h
        J10 = (vx[:, 1] - v[:, 1]) / h
        J01 = (vy[:, 0] - v[:, 0]) / h
        J11 = (vy[:, 1] - v[:, 1]) / h
        det = J00 * J11 - J01 * J10
        good = active & okx & oky & (torch.abs(det) > 1e-12)
        dxq = torch.where(good, (J11 * f[:, 0] - J01 * f[:, 1]) / det, torch.zeros_like(det))
        dyq = torch.where(good, (-J10 * f[:, 0] + J00 * f[:, 1]) / det, torch.zeros_like(det))
        q[:, 0] = q[:, 0] - dxq
        q[:, 1] = q[:, 1] - dyq
        kill = (~good) & active
        q[kill] = float("nan")

    q0 = torch.nan_to_num(q)
    v, ok = _trace_stop_t(surfs, ax, q0[:, 0], q0[:, 1], stop_z)
    err = torch.linalg.vector_norm(v - targets, dim=1)
    ok = ok & (err < 2e-5) & torch.isfinite(q).all(dim=1)
    q = torch.where(ok[:, None], q, torch.full_like(q, float("nan")))
    return q, ok, err


def _eikonal_t(surfs, ax: float, uv: torch.Tensor, ref, film: float = E.FILM, stop_radius: float = E.STOP_R):
    inside = torch.sum(uv * uv, dim=1) <= 1.0
    targets = uv * stop_radius
    q = torch.full_like(uv, float("nan"))
    ok = torch.zeros(len(uv), dtype=torch.bool, device=uv.device)
    if bool(inside.any().item()):
        qq, oo, _ = _solve_input_t(surfs, ax, targets[inside])
        q[inside] = qq
        ok[inside] = oo

    q0 = torch.nan_to_num(q)
    z0, xs, ys, dz0, dx0, dy0 = _safe_incident_start_t(surfs, ax, q0[:, 0], q0[:, 1])
    z = torch.full((len(uv),), z0, dtype=uv.dtype, device=uv.device)
    x, y = xs, ys
    dz = torch.full_like(z, dz0)
    dx = torch.full_like(z, dx0)
    dy = torch.full_like(z, dy0)
    opl = dx * x + dy * y
    valid = ok.clone()
    crossed = torch.zeros(len(uv), dtype=torch.bool, device=uv.device)

    for s in surfs:
        cross = (~crossed) & (z < 0.0) & (0.0 < s.z)
        if bool(cross.any().item()):
            tt = (0.0 - z) / dz
            sx = x + tt * dx
            sy = y + tt * dy
            valid = valid & ((~cross) | (torch.hypot(sx, sy) <= stop_radius * (1.0 + 1e-8)))
            crossed = crossed | cross
        z, x, y, dz, dx, dy, oo, opl = _surf_step_t(z, x, y, dz, dx, dy, s, opl)
        valid = valid & oo

    xr, yr = ref
    S = opl + torch.sqrt((film - z) ** 2 + (xr - x) ** 2 + (yr - y) ** 2)
    S = torch.where(valid, S, torch.full_like(S, float("nan")))
    return S, valid


def direct_otf_batch_cuda(surfs, lam, ax, fno, ref, seed, qmc_samples=QMC_SAMPLES):
    if _DEVICE.type != "cuda":
        raise RuntimeError("CUDA evaluator requested without CUDA")
    m = int(round(math.log2(qmc_samples)))
    if 2 ** m != int(qmc_samples):
        raise ValueError("qmc_samples must be power of 2")

    # Preserve the exact SciPy Sobol sequence used by the authoritative evaluator.
    base = qmc.Sobol(2, scramble=True, seed=seed).random_base2(m) * 2.0 - 1.0
    lm = lam * 1e-6
    arrays = [base]
    meta = []
    for f in E.FREQS:
        nu = lm * abs(fno) * f
        d = 2.0 * nu
        if nu >= 1.0:
            meta.append(None)
            continue
        idx = []
        for axis in (0, 1):
            shift = np.array([d / 2.0, 0.0]) if axis == 0 else np.array([0.0, d / 2.0])
            ip = len(arrays); arrays.append(base + shift)
            im = len(arrays); arrays.append(base - shift)
            idx.append((ip, im))
        meta.append(idx)

    lens = [len(a) for a in arrays]
    alluv = torch.as_tensor(np.vstack(arrays), dtype=_DTYPE, device=_DEVICE)
    with torch.inference_mode():
        S, V = _eikonal_t(surfs, ax, alluv, ref)
        starts = np.cumsum([0] + lens)
        ss = [S[int(starts[i]):int(starts[i + 1])] for i in range(len(arrays))]
        vv = [V[int(starts[i]):int(starts[i + 1])] for i in range(len(arrays))]
        den = int(torch.count_nonzero(vv[0]).item())
        ot, os_ = {}, {}
        if den == 0:
            return {"otf_t": {}, "otf_s": {}, "pf": 0.0}
        for f, idx in zip(E.FREQS, meta):
            if idx is None:
                ot[str(f)] = 0j; os_[str(f)] = 0j
                continue
            vals = []
            for ip, im in idx:
                good = vv[ip] & vv[im]
                if bool(good.any().item()):
                    phase = 2.0 * math.pi * (ss[ip][good] - ss[im][good]) / lm
                    num = torch.complex(torch.cos(phase), torch.sin(phase)).sum().item()
                else:
                    num = 0j
                vals.append(num / den)
            ot[str(f)], os_[str(f)] = vals

    n_circle = int(np.count_nonzero(np.sum(base * base, axis=1) <= 1.0))
    return {"otf_t": ot, "otf_s": os_, "pf": den / max(n_circle, 1)}


# Patch only the heavy direct-OTF backend.  E._evaluate_surfaces performs all
# authoritative scalar logic and resolves this global at call time.
E.direct_otf_batch = direct_otf_batch_cuda


def evaluate_final(design, qmc_samples=QMC_SAMPLES):
    return E._evaluate_surfaces(design, None, qmc_samples)


def evaluate_final_with_gaps(design, gaps, qmc_samples=QMC_SAMPLES):
    return E._evaluate_surfaces(design, tuple(float(x) for x in gaps), qmc_samples)


def synchronize() -> None:
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize(_DEVICE)
