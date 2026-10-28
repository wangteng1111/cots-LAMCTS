#!/usr/bin/env python3
"""Portable CPU pilot for 18 published photographic optical prescriptions.

Not empirical production-lens validation. Uses the unchanged Qfilm0.4-c
physical Final-J evaluator at center/edge, 550 nm; records import failures.
DeepLens source must be the documented pinned commit. No CUDA/Triton needed.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from validation.qfilm_deeplens_same_prescription import BK7
from deeplens import GeoLens
from evaluator.qfilm02 import Config, _surfaces_at, trace_pupil
from evaluator.optical_primitives import cardinal
from evaluator.qfilm04 import evaluate
from evaluator.qfilm03_physical_mode import PhysicalHuygensConfig
from validation.qfilm04_real_prescription_import import (
    CAMERA_SAMPLES, import_geolens_lens, extract_full_precision_lens)
from validation.qfilm04_secondary_patent_sources import ZEMAX_SAMPLES

DEEP_SHA = "911bb9d334fce7e20a4407c4506ac683b52ff246"
ZMX_SHA = "607ac6873b34b332c36cde428c70ace22ca43370"

def encode(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x).__name__)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deeplens-root", type=Path, required=True)
    ap.add_argument("--zmx-root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--import-only", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--names", nargs="*")
    args = ap.parse_args()
    actual = subprocess.check_output(["git", "-C", str(args.deeplens_root),
                                     "rev-parse", "HEAD"], text=True).strip()
    if actual != DEEP_SHA: raise ValueError("DeepLens source commit mismatch")
    if args.zmx_root:
        actual_zmx = subprocess.check_output(["git", "-C", str(args.zmx_root),
                                              "rev-parse", "HEAD"], text=True).strip()
        if actual_zmx != ZMX_SHA: raise ValueError("Zemax source commit mismatch")
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    samples = [("DeepLens_camera", n) for n in CAMERA_SAMPLES]
    if args.zmx_root: samples += [("PatentZMX", n) for n in ZEMAX_SAMPLES]
    if args.names: samples = [x for x in samples if x[1] in args.names]
    report = {"schema": "Qfilm04-portable-cpu-real-prescriptions-v1",
        "variant": "Qfilm0.4-c", "torch_version": str(torch.__version__),
        "source_sha": DEEP_SHA, "zmx_source_sha": ZMX_SHA if args.zmx_root else None,
        "protocol": {"fields": [0., 1.], "frame_mm": [36., 24.],
                     "wavelength_nm": [550.], "pupil_n": 256,
                     "detector_n": 128, "detector_pitch_um": 5.},
        "import_only": args.import_only, "manufactured_lens_empirical_validation": False,
        "independent_mtf_reference_run": False, "training_eligible": False, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, default=encode, indent=2,
                                           allow_nan=False) + "\n")
    for kind, name in samples:
        case = {"name": name, "source_kind": kind, "solver_ok": False}
        report["cases"].append(case)
        t0 = time.perf_counter()
        try:
            if kind == "DeepLens_camera":
                path = args.deeplens_root / "datasets/lenses/camera" / (name + ".json")
                raw = json.loads(path.read_text())
                url = f"https://github.com/vccimaging/DeepLens/blob/{DEEP_SHA}/datasets/lenses/camera/{name}.json"
            else:
                rel = ZEMAX_SAMPLES[name]
                path = args.zmx_root / rel
                url = f"https://github.com/Amarthgul/ZemaxSims/blob/{ZMX_SHA}/{rel}"
                raw = None
            case.update(source_url=url, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            lens = GeoLens(filename=str(path), device="cpu", dtype=torch.float64)
            if raw is None: raw = extract_full_precision_lens(lens, url)
            p = import_geolens_lens(lens, raw, name)
            qe = float(cardinal(_surfaces_at(p["surfaces"], 550.))[0])
            case.update(import_ok=True, surface_count=len(p["surfaces"]),
                reference_description=raw.get("info"),
                physical_efl_mm=float(lens.foclen), qfilm_efl_mm=qe,
                efl_abs_difference_mm=abs(qe-float(lens.foclen)),
                stop_z_mm=p["stop_z_mm"], film_z_mm=p["image_z_mm"])
            if abs(qe-float(lens.foclen)) > 5e-5:
                raise ValueError("paraxial EFL parity failed")
            if not args.import_only:
                cfg = Config(fields=2, pupil_n=256, fft_n=1024,
                    max_fields_per_chunk=1, wavelengths_nm=(550.,),
                    spectral_weights=(1.,), target_efl_mm=qe)
                with torch.inference_mode():
                    res = evaluate(p, cfg, PhysicalHuygensConfig(
                        pupil_n=256, detector_n=128, detector_pitch_um=5.,
                        detector_block=64, record_all_mtf=True), backend="c")
                case.update(solver_ok=True, quality=res["quality"],
                    variant=res["variant"], Q_image=res["Q_image"],
                    physical_diagnostics=res.get("physical_diagnostics"),
                    training_eligible=res["training_eligible"],
                    solver_seconds=res["timings"]["total_seconds"])
        except Exception as exc:
            case.update(error=repr(exc), traceback=traceback.format_exc()[-2200:])
        case["elapsed_seconds"] = time.perf_counter() - t0
        save()
        print("CPU_REAL_CASE", name, json.dumps({k:v for k,v in case.items()
            if k not in ("Q_image", "physical_diagnostics", "traceback")}, default=encode), flush=True)
    report["summary"] = {"attempted": len(samples),
        "import_ok": sum(c.get("import_ok", False) for c in report["cases"]),
        "solver_ok": sum(c["solver_ok"] for c in report["cases"]),
        "valid_final_score": sum(bool(c.get("quality", {}).get("valid")) for c in report["cases"])}
    save()
    print("CPU_REAL_SUMMARY", json.dumps(report["summary"]), flush=True)

if __name__ == "__main__": main()
