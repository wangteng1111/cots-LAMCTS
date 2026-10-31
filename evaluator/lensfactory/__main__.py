"""Command-line entry point for LensFactorySingleCandidateSolver."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from evaluator.qfilm04_io import atomic_json
from .contracts import LensFactoryConfig
from .solver import evaluate


def main():
    ap = argparse.ArgumentParser(description="Lens Factory-style single-candidate optical solver")
    backend = ap.add_mutually_exclusive_group(required=True)
    backend.add_argument("-c", action="store_true", help="CPU float64")
    backend.add_argument("-g", action="store_true", help="CUDA float64")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--prescription", type=Path, required=True)
    ap.add_argument("--case", help="case name when the input JSON contains a cases array")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--no-optimize", action="store_true", help="skip MTF-stage optimization; Stage 1 still optimizes air gaps and BFL")
    ap.add_argument("--no-final-verify", action="store_true", help="diagnostics only; J remains null")
    ap.add_argument("--quick", action="store_true", help="small smoke profile; never produces J")
    a = ap.parse_args()

    data = json.loads(a.prescription.read_text(encoding="utf-8"))
    if "cases" in data:
        if not a.case:
            ap.error("--case is required for a fixture containing multiple cases")
        matches = [c for c in data["cases"] if c.get("name") == a.case]
        if len(matches) != 1:
            ap.error(f"case not found or duplicated: {a.case}")
        prescription = matches[0]["prescription"]
    else:
        prescription = data

    config = LensFactoryConfig()
    final_verify = not a.no_final_verify
    if a.quick:
        config = replace(
            config, fields=(0.0,), wavelengths_nm=(550.0,), spectral_weights=(1.0,),
            stage1_pupil_n=32, stage1_maxiter=1, uniform_gap_seeds_mm=(3.0,),
            local_grid_steps_mm=(), stage2_pupil_n=24, stage2_detector_n=64,
            stage2_pitch_um=1.0, stage2_pixel_block=16, stage2_maxiter=0,
            coarse_mtf_points=7,
        )
        final_verify = False
    result = evaluate(
        prescription, config=config, backend="c" if a.c else "g",
        gpu_index=a.gpu_index, optimize=not a.no_optimize,
        final_verify=final_verify,
    )
    atomic_json(a.output, result)
    print(json.dumps({
        "version": result["version"], "state": result["state"],
        "J": result["J"], "score_valid": result["score_valid"],
        "device": result["device"], "elapsed_seconds": result["elapsed_seconds"],
        "training_eligible": result["training_eligible"],
    }, allow_nan=False))


if __name__ == "__main__":
    main()