"""Lens Factory-style single-candidate optical optimization.

The discrete element sequence is held fixed. Only internal air gaps and BFL
are optimized. Stage 1 uses sequential geometric ray tracing and MSSE
spot/OPD. Stage 2 uses a direct scalar Rayleigh-Sommerfeld-1 (RS1) integral,
then Fourier-derived MTF area. Final J is emitted only after all declared
field/wavelength conditions pass the Qfilm0.4 numerical convergence gates.

This is a research implementation of the published method, not the original
Lens Factory source code. Scalar transport remains provisional and is never
marked physics-certified or training-eligible by this module.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import copy
import hashlib
import json
import math
import time

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar

from evaluator.qfilm02 import _surfaces_at, pupil_grid, trace_pupil
from evaluator.optical_primitives import cardinal
from evaluator.qfilm04_wavefront import (
    WavefrontError, fixed_exit_plane, prepare_wavefront_bundle, integrate_fixed_film,
)
from evaluator.qfilm04_sampling import ConvergencePolicy, make_plan, solve_condition
from .contracts import LensFactoryConfig

VERSION = "LensFactorySingleCandidate-0.1.0"


def _finite(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _hashable(value):
    if isinstance(value, dict):
        return {str(k): _hashable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hashable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _digest(value):
    raw = json.dumps(_hashable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _prescription_dict(source):
    if isinstance(source, dict):
        p = copy.deepcopy(source.get("prescription", source))
    elif hasattr(source, "surfaces"):
        surfaces = [
            asdict(surface) if is_dataclass(surface) else dict(vars(surface))
            for surface in source.surfaces
        ]
        p = {
            "surfaces": surfaces,
            "stop_z_mm": source.stop_z_mm,
            "image_z_mm": source.image_z_mm,
            "design_spec": copy.deepcopy(source.design_spec),
            "source_id": getattr(source, "source_id", None),
            "source_provenance": copy.deepcopy(getattr(source, "provenance", {})),
        }
    else:
        raise TypeError("prescription must be a dict or a Prescription-like object")

    if not isinstance(p.get("surfaces"), list) or len(p["surfaces"]) < 2:
        raise ValueError("at least two sequential optical surfaces are required")
    p["surfaces"] = [dict(s) if isinstance(s, dict) else asdict(s) for s in p["surfaces"]]
    for i, surface in enumerate(p["surfaces"]):
        for key in ("radius", "thickness", "n_after"):
            if key not in surface:
                raise ValueError(f"surface {i} is missing {key}")
        radius = surface["radius"]
        if isinstance(radius, str) and radius.lower() in ("inf", "+inf", "infinity"):
            surface["radius"] = math.inf
        elif isinstance(radius, str) and radius.lower() == "-inf":
            surface["radius"] = -math.inf
        else:
            surface["radius"] = float(radius)
        surface["thickness"] = float(surface["thickness"])
        surface["n_after"] = float(surface["n_after"])
        if not _finite(surface["thickness"]) or surface["thickness"] < 0:
            raise ValueError(f"surface {i} has invalid following thickness")
        if not _finite(surface["n_after"]) or surface["n_after"] <= 0:
            raise ValueError(f"surface {i} has invalid refractive index")
        aperture = surface.get("clear_aperture")
        if aperture is None or not _finite(aperture) or float(aperture) <= 0:
            raise ValueError(f"surface {i} needs an explicit positive clear_aperture")
        surface["clear_aperture"] = float(aperture)
        surface.setdefault("v_after", 0.0)
        surface.setdefault("conic", 0.0)
        surface.setdefault("asphere", [])

    spec = p.get("design_spec")
    if not isinstance(spec, dict):
        raise ValueError("design_spec is required")
    for key in ("efl_target_mm", "stop_diameter_mm"):
        if not _finite(spec.get(key)) or float(spec[key]) <= 0:
            raise ValueError(f"design_spec.{key} must be positive and finite")
    p["design_spec"] = copy.deepcopy(spec)

    total = sum(float(s["thickness"]) for s in p["surfaces"])
    if p.get("image_z_mm") is None:
        p["image_z_mm"] = total
    else:
        p["image_z_mm"] = float(p["image_z_mm"])
        if abs(p["image_z_mm"] - total) > 1e-4:
            raise ValueError("image_z_mm must equal the sum of sequential surface thicknesses")
    if not _finite(p.get("stop_z_mm")) or not 0 <= float(p["stop_z_mm"]) < p["image_z_mm"]:
        raise ValueError("stop_z_mm must be finite and lie before the image plane")
    p["stop_z_mm"] = float(p["stop_z_mm"])
    return p


def _export_prescription(p):
    out = copy.deepcopy(p)
    for surface in out["surfaces"]:
        radius = float(surface["radius"])
        if math.isinf(radius):
            surface["radius"] = "inf" if radius > 0 else "-inf"
    return out


def _insert_stop_element(p):
    """Represent the aperture stop as a zero-power element in the sequence.

    Lens Factory defines each c_i as either a catalog lens or the stop, with
    independent air gaps between adjacent elements. A legacy prescription
    stores stop_z inside one air thickness; split that thickness into the
    pre-stop and post-stop gaps and insert an optically inert stop plane.
    """
    p = copy.deepcopy(p)
    surfaces = p["surfaces"]
    stop_z = float(p["stop_z_mm"])
    stop_index = p.get("stop_surface_index")
    inserted = False

    if stop_index is None:
        role_indices = [i for i, surf in enumerate(surfaces)
                        if surf.get("surface_role") == "aperture_stop"]
        if len(role_indices) > 1:
            raise ValueError("only one aperture-stop element is supported")
        if role_indices:
            stop_index = role_indices[0]
        else:
            positions = []
            z = 0.0
            for surf in surfaces:
                positions.append(z)
                z += float(surf["thickness"])
            candidates = []
            for i, surf in enumerate(surfaces[:-1]):
                begin = positions[i]
                finish = begin + float(surf["thickness"])
                if (begin - 1e-8 <= stop_z <= finish + 1e-8
                    and abs(float(surf["n_after"]) - 1.0) <= 1e-5
                    and float(surf["thickness"]) > 1e-8):
                    candidates.append((i, begin, finish))
            if not candidates:
                raise ValueError("stop must lie in an internal air gap between lens elements")
            i, begin, finish = min(
                candidates, key=lambda item: abs(stop_z - 0.5*(item[1]+item[2]))
            )
            left_gap = max(0.0, stop_z - begin)
            right_gap = max(0.0, finish - stop_z)
            surfaces[i]["thickness"] = left_gap
            stop_surface = {
                "radius": math.inf, "thickness": right_gap, "n_after": 1.0,
                "v_after": 0.0, "conic": 0.0, "asphere": [],
                "clear_aperture": float(p["design_spec"]["stop_diameter_mm"]),
                "surface_role": "aperture_stop", "surface_source_index": None,
            }
            surfaces.insert(i+1, stop_surface)
            stop_index = i+1
            p["stop_z_mm"] = begin + left_gap
            inserted = True

    stop_index = int(stop_index)
    if stop_index <= 0 or stop_index >= len(surfaces)-1:
        raise ValueError("v0.1 requires the aperture stop between lens elements")
    positions = []
    z = 0.0
    for surf in surfaces:
        positions.append(z)
        z += float(surf["thickness"])
    if abs(positions[stop_index] - float(p["stop_z_mm"])) > 1e-6:
        raise ValueError("explicit stop element position disagrees with stop_z_mm")
    stop_surface = surfaces[stop_index]
    if (not math.isinf(float(stop_surface["radius"]))
        or abs(float(stop_surface["n_after"]) - 1.0) > 1e-5):
        raise ValueError("aperture-stop element must be a zero-power plane in air")
    stop_surface["surface_role"] = "aperture_stop"
    stop_surface["clear_aperture"] = float(p["design_spec"]["stop_diameter_mm"])
    p["stop_surface_index"] = stop_index
    p["stop_z_mm"] = positions[stop_index]
    return p, {"stop_surface_index": stop_index, "stop_element_inserted": inserted}


def inspect_prescription(source):
    """Validate a prescription and expand its aperture stop into an element."""
    original = _prescription_dict(source)
    source_hash = _digest(original)
    p, stop_info = _insert_stop_element(original)
    surfaces = p["surfaces"]
    positions = []
    z = 0.0
    for surface in surfaces:
        positions.append(z)
        z += float(surface["thickness"])

    # Each air interval between adjacent elements is an independent variable.
    # The final interval is the BFL and is optimized separately.
    gap_indices = [
        i for i, surface in enumerate(surfaces[:-1])
        if abs(float(surface["n_after"]) - 1.0) <= 1e-5
        and float(surface["thickness"]) > 1e-5
    ]
    if not gap_indices:
        raise ValueError("no positive internal air gaps are available to optimize")
    bfl0 = float(surfaces[-1]["thickness"])
    if bfl0 <= 0:
        raise ValueError("positive initial BFL is required")
    return {
        "prescription": p, "surface_positions_mm": positions,
        "gap_indices": gap_indices,
        "initial_gaps_mm": [float(surfaces[i]["thickness"]) for i in gap_indices],
        **stop_info,
        "initial_bfl_mm": bfl0,
        "last_surface_vertex_z_mm": positions[-1],
        "input_hash": source_hash, "expanded_prescription_hash": _digest(p),
    }


def _apply_variables(info, gaps, bfl):
    p = copy.deepcopy(info["prescription"])
    if len(gaps) != len(info["gap_indices"]):
        raise ValueError("air-gap vector length mismatch")
    for i, value in zip(info["gap_indices"], gaps):
        value = float(value)
        if not _finite(value) or value < 0:
            raise ValueError("air gaps must be finite and nonnegative")
        p["surfaces"][i]["thickness"] = value
    if not _finite(bfl) or float(bfl) <= 0:
        raise ValueError("BFL must be positive and finite")
    p["surfaces"][-1]["thickness"] = float(bfl)

    z = 0.0
    stop_z = None
    for i, surface in enumerate(p["surfaces"]):
        if i == info["stop_surface_index"]:
            stop_z = z
        z += float(surface["thickness"])
    if stop_z is None:
        raise ValueError("failed to locate explicit stop element")
    p["stop_z_mm"] = stop_z
    p["image_z_mm"] = z
    return p


def _paraxial_metrics(p, wavelength_nm):
    surfs = _surfaces_at(p["surfaces"], wavelength_nm)
    efl, focus_z = cardinal(surfs)
    last_vertex = float(surfs[-1].z)
    return {
        "efl_mm": float(efl), "paraxial_focus_z_mm": float(focus_z),
        "paraxial_bfl_mm": float(focus_z - last_vertex),
        "last_surface_vertex_z_mm": last_vertex,
    }


def _efl_tolerance(p, config):
    target = float(p["design_spec"]["efl_target_mm"])
    explicit = p["design_spec"].get("efl_tol_mm")
    if _finite(explicit) and float(explicit) > 0:
        return float(explicit)
    return max(config.efl_tolerance_floor_mm, config.efl_tolerance_fraction * target)


def _efl_penalty(p, config, wavelength_nm):
    m = _paraxial_metrics(p, wavelength_nm)
    target = float(p["design_spec"]["efl_target_mm"])
    tol = _efl_tolerance(p, config)
    error = abs(m["efl_mm"] - target)
    if not math.isfinite(error):
        return 1e9, {**m, "target_efl_mm": target, "efl_tolerance_mm": tol,
                     "efl_error_mm": None, "efl_constraint_passed": False}
    excess = max(0.0, error - tol)
    penalty = 0.0 if excess == 0 else 1000.0 + (excess / tol) ** 2
    return penalty, {
        **m, "target_efl_mm": target, "efl_tolerance_mm": tol,
        "efl_error_mm": m["efl_mm"] - target, "efl_constraint_passed": error <= tol,
    }


def _bfl_bounds(p, config, wavelength_nm):
    paraxial = _paraxial_metrics(p, wavelength_nm)
    center = paraxial["paraxial_bfl_mm"]
    if not math.isfinite(center) or center <= 0:
        center = float(p["surfaces"][-1]["thickness"])
    half = config.bfl_search_half_width_mm
    lo = max(0.05, center - half)
    hi = max(center + half, lo + 0.1)
    return (lo, hi), paraxial


@torch.inference_mode()
def _build_ray_cache(p, config, pupil_n, device, bfl_bounds):
    cache = []
    uv, inside = pupil_grid(pupil_n, device)
    stop_z = float(p["stop_z_mm"])
    stop_r = float(p["design_spec"]["stop_diameter_mm"]) / 2.0
    target_efl = float(p["design_spec"]["efl_target_mm"])
    last_vertex = sum(float(s["thickness"]) for s in p["surfaces"][:-1])
    ref_z = last_vertex + max(100.0, bfl_bounds[1] + 10.0)
    frame_radius = math.hypot(*config.frame_mm) / 2.0

    for field_h in config.fields:
        alpha = math.atan(float(field_h) * frame_radius / target_efl)
        angles = torch.tensor([[alpha]], dtype=torch.float64, device=device)
        for wave in config.wavelengths_nm:
            surfs = _surfaces_at(p["surfaces"], float(wave))
            z, x, y, opl, xref, yref, valid = trace_pupil(
                surfs, angles, uv, stop_z, stop_r, ref_z
            )
            z, x, y, opl, xref, yref, valid = [
                v.reshape(-1) for v in (z, x, y, opl, xref, yref, valid)
            ]
            valid = valid & inside & torch.isfinite(z) & torch.isfinite(opl)
            dz = ref_z - z
            valid = valid & (dz > 1e-8)
            sx = (xref - x) / dz.clamp_min(1e-12)
            sy = (yref - y) / dz.clamp_min(1e-12)
            valid = valid & torch.isfinite(sx) & torch.isfinite(sy)
            if int(valid.sum()) < 32:
                raise WavefrontError(
                    "GEOMETRIC_RAY_FAILURE",
                    f"fewer than 32 valid pupil rays at field={field_h}, wavelength={wave}",
                )
            cache.append({
                "field_h": float(field_h), "wavelength_nm": float(wave),
                "z": z, "x": x, "y": y, "opl": opl, "sx": sx, "sy": sy,
                "valid": valid, "inside_count": int(inside.sum()),
            })
    return cache


@torch.inference_mode()
def _geometric_metrics(cache, last_vertex_z, bfl, metric):
    rows = []
    for ray in cache:
        image_z = last_vertex_z + float(bfl)
        d = image_z - ray["z"]
        valid = ray["valid"] & (d > 1e-8)
        if int(valid.sum()) < 32:
            raise WavefrontError("GEOMETRIC_RAY_FAILURE", "too few rays reach the proposed image plane")
        xi = ray["x"] + d * ray["sx"]
        yi = ray["y"] + d * ray["sy"]
        opl_image = ray["opl"] + d * torch.sqrt(1.0 + ray["sx"]**2 + ray["sy"]**2)
        cx = xi[valid].mean()
        cy = yi[valid].mean()
        spot_msse = ((xi[valid] - cx)**2 + (yi[valid] - cy)**2).mean()
        opd_msse = ((opl_image[valid] - opl_image[valid].mean())**2).mean()
        rows.append({
            "field_h": ray["field_h"], "wavelength_nm": ray["wavelength_nm"],
            "spot_msse_mm2": float(spot_msse), "opd_msse_mm2": float(opd_msse),
            "valid_ray_fraction": float(valid.sum()) / ray["inside_count"],
            "valid_ray_count": int(valid.sum()),
        })
    spot = float(np.mean([r["spot_msse_mm2"] for r in rows]))
    opd = float(np.mean([r["opd_msse_mm2"] for r in rows]))
    return {
        "spot_msse_mm2": spot, "opd_msse_mm2": opd,
        "spot_rms_um": math.sqrt(max(0.0, spot)) * 1000.0,
        "opd_rms_nm": math.sqrt(max(0.0, opd)) * 1e6,
        "minimum_valid_ray_fraction": min(r["valid_ray_fraction"] for r in rows),
        "conditions": rows, "objective": spot if metric == "spot" else opd,
        "objective_metric": metric,
    }


def _evaluate_gap_vector(info, gaps, config, device):
    p = _apply_variables(info, gaps, info["initial_bfl_mm"])
    mid_wave = config.wavelengths_nm[len(config.wavelengths_nm)//2]
    penalty, _ = _efl_penalty(p, config, mid_wave)
    if penalty:
        return {"objective": penalty, "candidate": p, "feasible": False}
    bfl_bounds, paraxial = _bfl_bounds(p, config, mid_wave)
    cache = _build_ray_cache(p, config, config.stage1_pupil_n, device, bfl_bounds)
    last_vertex = sum(float(s["thickness"]) for s in p["surfaces"][:-1])
    metric = config.stage1_metric

    def bfl_objective(bfl):
        try:
            return _geometric_metrics(cache, last_vertex, bfl, metric)["objective"]
        except WavefrontError:
            return 1e9

    res = minimize_scalar(
        bfl_objective, bounds=bfl_bounds, method="bounded",
        options={"xatol": 1e-3, "maxiter": 60},
    )
    bfl = float(res.x)
    p = _apply_variables(info, gaps, bfl)
    metrics = _geometric_metrics(cache, last_vertex, bfl, metric)
    penalty, final_paraxial = _efl_penalty(p, config, mid_wave)
    return {
        "objective": float(metrics["objective"] + penalty),
        "candidate": p, "gaps_mm": [float(x) for x in gaps],
        "bfl_mm": bfl, "metrics": metrics, "paraxial": final_paraxial,
        "feasible": penalty == 0 and bool(res.success),
        "bfl_bounds_mm": list(bfl_bounds), "bfl_optimizer_success": bool(res.success),
    }


def _stage1(info, config, device):
    gap_count = len(info["gap_indices"])
    gap_bounds = [tuple(config.gap_bounds_mm)] * gap_count
    seeds = [("catalog_original", np.asarray(info["initial_gaps_mm"], dtype=float))]
    if gap_count:
        seeds.extend(
            (f"uniform_{value:g}mm", np.full(gap_count, float(value)))
            for value in config.uniform_gap_seeds_mm
        )
    else:
        seeds = [("no_internal_air_gap", np.asarray([], dtype=float))]

    memo = {}
    def evaluate_gaps(gaps):
        key = tuple(round(float(v), 6) for v in gaps)
        if key not in memo:
            try:
                memo[key] = _evaluate_gap_vector(info, gaps, config, device)
            except (WavefrontError, ValueError, RuntimeError) as exc:
                memo[key] = {"objective": 1e9, "feasible": False, "error": str(exc)}
            if len(memo) > 48:
                memo.pop(next(iter(memo)))
        return memo[key]

    seed_reports = []
    best = None
    for seed_name, seed in seeds:
        seed = np.clip(seed, config.gap_bounds_mm[0], config.gap_bounds_mm[1])
        if gap_count and config.stage1_maxiter > 0:
            res = minimize(
                lambda x: evaluate_gaps(x)["objective"], seed,
                method="L-BFGS-B", bounds=gap_bounds,
                options={"maxiter": config.stage1_maxiter,
                         "maxfun": max(30, config.stage1_maxiter*(gap_count+1)*3),
                         "ftol": 1e-9},
            )
            candidate = evaluate_gaps(res.x)
            optimizer_success = bool(res.success)
            optimizer_message = str(res.message)
        else:
            candidate = evaluate_gaps(seed)
            optimizer_success = True
            optimizer_message = "optimization disabled or no internal air gaps"
        seed_reports.append({
            "seed": seed_name, "initial_gaps_mm": seed.tolist(),
            "best_objective": candidate["objective"],
            "best_gaps_mm": candidate.get("gaps_mm", []),
            "best_bfl_mm": candidate.get("bfl_mm"),
            "optimizer_success": optimizer_success,
            "optimizer_message": optimizer_message,
            "feasible": candidate.get("feasible", False),
            "error": candidate.get("error"),
        })
        if best is None or candidate["objective"] < best["objective"]:
            best = candidate

    if best is None or not best.get("feasible", False):
        raise RuntimeError("Stage 1 found no geometrically feasible gap/BFL solution")

    local_evaluations = 0
    gaps = np.asarray(best["gaps_mm"], dtype=float)
    for step in config.local_grid_steps_mm:
        for i in range(len(gaps)):
            local_best = best
            for delta in (-float(step), float(step)):
                trial = gaps.copy()
                trial[i] = np.clip(trial[i] + delta, *config.gap_bounds_mm)
                candidate = evaluate_gaps(trial)
                local_evaluations += 1
                if candidate["objective"] < local_best["objective"]:
                    local_best = candidate
            best = local_best
            gaps = np.asarray(best["gaps_mm"], dtype=float)

    return {
        "state": "completed", "metric": config.stage1_metric,
        "gap_indices": info["gap_indices"], "seed_runs": seed_reports,
        "local_grid_evaluations": local_evaluations,
        "optimized_gaps_mm": best["gaps_mm"],
        "optimized_bfl_mm": best["bfl_mm"],
        "objective": best["objective"],
        "geometric_metrics": best["metrics"], "paraxial": best["paraxial"],
        "candidate": best["candidate"], "bfl_bounds_mm": best["bfl_bounds_mm"],
    }


def _mtf_auc(mtf, frequencies):
    auc = np.trapz(np.asarray(mtf, dtype=float), np.asarray(frequencies, dtype=float), axis=0)
    return float(auc[0]), float(auc[1])


@torch.inference_mode()
def _coarse_mtf(p, config, device):
    frequencies = config.coarse_frequencies
    conditions = []
    auc_values = []
    for field_h in config.fields:
        for wave in config.wavelengths_nm:
            exit_z = fixed_exit_plane(p, float(wave))
            bundle = prepare_wavefront_bundle(
                p, float(wave), float(field_h), config.stage2_pupil_n, exit_z,
                center_mm=None, device=device, frame_mm=config.frame_mm,
                transport_model="uniform_stop_flux",
                max_bad_jacobian_fraction=config.max_bad_jacobian_fraction,
            )
            raw = integrate_fixed_film(
                bundle, config.stage2_detector_n, config.stage2_pitch_um,
                frequencies, pixel_block=config.stage2_pixel_block, kernel="rs1",
            )
            otf = np.asarray(raw["otf"], dtype=np.complex128)
            mtf = np.clip(np.abs(otf), 0.0, 1.0)
            auc_t, auc_s = _mtf_auc(mtf, frequencies)
            auc_values.extend((auc_t, auc_s))
            conditions.append({
                "field_h": float(field_h), "wavelength_nm": float(wave),
                "mtf_tangential_x": mtf[:, 0].tolist(),
                "mtf_sagittal_y": mtf[:, 1].tolist(),
                "mtf_auc_tangential_lpmm": auc_t,
                "mtf_auc_sagittal_lpmm": auc_s,
                "valid_rays": raw.get("valid_rays"),
                "bad_jacobian_fraction": raw.get("bad_jacobian_fraction"),
                "exit_z_mm": raw.get("exit_z_mm"), "support_mm": raw.get("support_mm"),
            })
    auc_lpmm = float(np.mean(auc_values))
    return {
        "state": "coarse_complete", "frequencies_lpmm": list(frequencies),
        "mtf_auc_mean_lpmm": auc_lpmm,
        "mtf_auc_normalized": auc_lpmm / config.mtf_max_frequency_lpmm,
        "conditions": conditions, "objective_is_optimization_only": True,
    }


def _stage2(info, stage1, config, device):
    gaps0 = np.asarray(stage1["optimized_gaps_mm"], dtype=float)
    bfl0 = float(stage1["optimized_bfl_mm"])
    x0 = np.concatenate((gaps0, np.asarray([bfl0])))
    bounds = [tuple(config.gap_bounds_mm)] * len(gaps0) + [tuple(stage1["bfl_bounds_mm"])]
    mid_wave = config.wavelengths_nm[len(config.wavelengths_nm)//2]
    memo = {}

    def objective(x):
        key = tuple(round(float(v), 6) for v in x)
        if key in memo:
            return memo[key][0]
        gaps, bfl = x[:-1], float(x[-1])
        try:
            p = _apply_variables(info, gaps, bfl)
            penalty, _ = _efl_penalty(p, config, mid_wave)
            if penalty:
                value = 1000.0 + penalty
                memo[key] = (value, None)
                return value
            coarse = _coarse_mtf(p, config, device)
            value = -float(coarse["mtf_auc_normalized"])
            memo[key] = (value, coarse)
            return value
        except (WavefrontError, ValueError, RuntimeError):
            memo[key] = (1e6, None)
            return 1e6

    start_value = objective(x0)
    if config.stage2_maxiter > 0 and start_value < 1e5:
        opt = minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": config.stage2_maxiter, "maxfun": config.stage2_maxfun, "ftol": 1e-6},
        )
        x_best = np.asarray(opt.x if objective(opt.x) <= start_value else x0, dtype=float)
        optimizer = {
            "success": bool(opt.success), "message": str(opt.message),
            "iterations": int(opt.nit), "function_evaluations": int(opt.nfev),
        }
    else:
        x_best = x0
        optimizer = {
            "success": True, "message": "MTF optimization disabled or initial coarse evaluation failed",
            "iterations": 0, "function_evaluations": 1,
        }

    gaps, bfl = x_best[:-1], float(x_best[-1])
    candidate = _apply_variables(info, gaps, bfl)
    coarse = _coarse_mtf(candidate, config, device)
    _, paraxial = _efl_penalty(candidate, config, mid_wave)
    return {
        "state": "coarse_optimized", "optimized_gaps_mm": [float(v) for v in gaps],
        "optimized_bfl_mm": bfl, "coarse_mtf": coarse, "optimizer": optimizer,
        "paraxial": paraxial, "candidate": candidate,
    }


def _final_verify(p, config, device):
    policy = ConvergencePolicy(
        pupil_candidates=tuple(config.final_pupils),
        support_candidates_mm=tuple(config.final_supports_mm),
        pitch_um=float(config.final_pitch_um),
        max_pair_evaluations=int(config.final_max_pair_evaluations),
        transport_model="uniform_stop_flux", kernel="rs1",
        max_bad_jacobian_fraction=float(config.max_bad_jacobian_fraction),
    )
    policy.validate()
    frequencies = config.final_frequencies
    conditions = []
    auc_weighted = []
    condition_ok = True
    for field_h in config.fields:
        for wave in config.wavelengths_nm:
            record = {
                "field_h": float(field_h), "wavelength_nm": float(wave),
                "state": "not_started", "plan_hash": None, "actual_device": str(device),
            }
            try:
                plan = make_plan(
                    p, float(wave), float(field_h), frequencies, policy,
                    frame_mm=config.frame_mm,
                )
                report = solve_condition(
                    p, plan, policy, device=device, pixel_block=config.final_pixel_block,
                )
                flags = report.get("flags", {})
                required = (
                    "geometry_valid", "wavefront_supported", "pupil_converged",
                    "support_converged", "pitch_converged", "joint_converged",
                )
                passed = report["state"] == "numerically_converged" and all(flags.get(k, False) for k in required)
                record.update({
                    "state": report["state"], "plan_hash": report.get("plan_hash"),
                    "flags": flags, "comparisons": report.get("comparisons"),
                    "pair_evaluations": report.get("pair_evaluations"),
                })
                selected = report.get("selected")
                if not passed or selected is None:
                    condition_ok = False
                else:
                    mtf = np.clip(np.abs(np.asarray(selected["otf"], dtype=np.complex128)), 0.0, 1.0)
                    auc_t, auc_s = _mtf_auc(mtf, frequencies)
                    record.update({
                        "mtf_tangential_x": mtf[:, 0].tolist(),
                        "mtf_sagittal_y": mtf[:, 1].tolist(),
                        "mtf_auc_tangential_lpmm": auc_t,
                        "mtf_auc_sagittal_lpmm": auc_s,
                        "selected_grid": {
                            "pupil_n": selected.get("pupil_n"),
                            "support_index": selected.get("support_index"),
                            "support_mm": selected.get("support_mm"),
                            "pitch_um": selected.get("pitch_um"),
                        },
                    })
                    weight = float(config.spectral_weights[config.wavelengths_nm.index(wave)])
                    auc_weighted.extend((
                        auc_t / config.mtf_max_frequency_lpmm * weight,
                        auc_s / config.mtf_max_frequency_lpmm * weight,
                    ))
            except (WavefrontError, ValueError, RuntimeError) as exc:
                condition_ok = False
                record.update(state=getattr(exc, "code", "NUMERICAL_FAILURE"), error=str(exc))
            conditions.append(record)

    spectral_sum = float(sum(config.spectral_weights))
    normalized_auc = None
    j_value = None
    if condition_ok and len(conditions) == len(config.fields) * len(config.wavelengths_nm):
        normalized_auc = float(sum(auc_weighted) / (2.0 * len(config.fields) * spectral_sum))
        j_value = 1.0 - normalized_auc
    return {
        "state": "numerically_converged" if condition_ok else "FINAL_NUMERICAL_GATE_NOT_PASSED",
        "frequencies_lpmm": list(frequencies), "conditions": conditions,
        "all_conditions_converged": condition_ok,
        "mtf_auc_normalized": normalized_auc,
        "mtf_auc_mean_lpmm": None if normalized_auc is None else normalized_auc * config.mtf_max_frequency_lpmm,
        "J_lensfactory": j_value, "score_direction": "lower_is_better",
        "score_definition": "1 - spectrally weighted mean MTF AUC normalized over 0..fmax, equal field and T/S weights",
        "reference_validated": False, "otf_domain": "finite PSF window", "model_scope_validated": False,
        "physics_certified": False, "empirically_calibrated": False,
        "training_eligible": False, "kernel": "direct scalar Rayleigh-Sommerfeld-1",
        "transport_model": "uniform stop-plane scalar flux; Fresnel/coating/polarization omitted",
    }


def evaluate(source, config=LensFactoryConfig(), backend="c", gpu_index=0,
             optimize=True, final_verify=True):
    """Optimize and evaluate one fixed element sequence.

    J is null unless every final field/wavelength condition passes all declared
    numerical gates. Stage-1 and coarse Stage-2 metrics are never promoted to J.
    """
    config.validate()
    if backend not in ("c", "g"):
        raise ValueError("backend must be 'c' (CPU) or 'g' (CUDA)")
    if backend == "g" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no implicit CPU fallback")
    device = "cpu" if backend == "c" else f"cuda:{int(gpu_index)}"
    t0 = time.perf_counter()
    info = inspect_prescription(source)
    stage1 = _stage1(info, config, device)
    if optimize:
        stage2 = _stage2(info, stage1, config, device)
    else:
        p = stage1["candidate"]
        coarse = _coarse_mtf(p, config, device)
        stage2 = {
            "state": "coarse_not_optimized",
            "optimized_gaps_mm": stage1["optimized_gaps_mm"],
            "optimized_bfl_mm": stage1["optimized_bfl_mm"],
            "coarse_mtf": coarse, "optimizer": {"success": True, "message": "optimization disabled"},
            "candidate": p,
        }
    candidate = stage2["candidate"]
    _, final_paraxial = _efl_penalty(
        candidate, config, config.wavelengths_nm[len(config.wavelengths_nm)//2]
    )
    final = _final_verify(candidate, config, device) if final_verify else {
        "state": "not_requested", "J_lensfactory": None,
        "all_conditions_converged": False, "training_eligible": False,
    }
    elapsed = time.perf_counter() - t0
    scored = final.get("J_lensfactory") is not None and final_paraxial["efl_constraint_passed"]
    return {
        "schema": "LensFactorySingleCandidateResult-v1", "version": VERSION,
        "state": ("NUMERICALLY_CONVERGED_PROVISIONAL" if scored else
                  "FINAL_VERIFY_NOT_PASSED" if final_verify else "COARSE_OPTIMIZATION_ONLY"),
        "backend": backend, "device": device, "input_hash": info["input_hash"],
        "configuration": config.as_dict(), "element_sequence_fixed": True,
        "optimized_prescription": _export_prescription(candidate),
        "optimization": {
            "stage1_geometric": {k: v for k, v in stage1.items() if k != "candidate"},
            "stage2_mtf": {k: v for k, v in stage2.items() if k != "candidate"},
        },
        "paraxial_final": final_paraxial, "final_evaluation": final,
        "J": final.get("J_lensfactory") if scored else None,
        "score_direction": "lower_is_better", "score_valid": bool(scored),
        "numerically_verified": bool(final.get("all_conditions_converged", False)),
        "reference_validated": False, "model_scope_validated": False, "physics_certified": False,
        "empirically_calibrated": False, "training_eligible": False,
        "limitations": [
            "fixed catalog element sequence; only internal air gaps and BFL are optimized",
            "scalar RS1 propagation with uniform stop-plane flux",
            "Fresnel transmission, AR coating transmission, polarization, scatter and manufacturing tolerances are not modeled",
            "model-scope and empirical validation are separate from numerical convergence",
        ],
        "elapsed_seconds": elapsed,
    }