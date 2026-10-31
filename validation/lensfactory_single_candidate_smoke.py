"""Small integration smoke test for the Lens Factory single-candidate solver."""
import argparse
import json
import math
from pathlib import Path

from evaluator.lensfactory import LensFactoryConfig, evaluate, inspect_prescription
from evaluator.qfilm04_io import atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("c", "g"), default="c")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--fixture", type=Path, default=Path("validation/fixtures/qfilmrw_real11.json"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/lensfactory_single_candidate_smoke.json"))
    a = ap.parse_args()

    data = json.loads(a.fixture.read_text(encoding="utf-8"))
    case = next(c for c in data["cases"] if c["name"] == "ef50mm_f1.8")
    p = case["prescription"]
    info = inspect_prescription(p)
    assert len(info["gap_indices"]) >= 1
    assert 0.0 <= info["stop_fraction"] <= 1.0

    config = LensFactoryConfig(
        fields=(0.0,), wavelengths_nm=(550.0,), spectral_weights=(1.0,),
        stage1_pupil_n=32, stage1_maxiter=1, uniform_gap_seeds_mm=(3.0,),
        local_grid_steps_mm=(), stage2_pupil_n=24, stage2_detector_n=64,
        stage2_pitch_um=1.0, stage2_pixel_block=16, stage2_maxiter=0,
        coarse_mtf_points=7,
    )
    result = evaluate(
        p, config=config, backend=a.backend, gpu_index=a.gpu_index,
        optimize=False, final_verify=False,
    )
    spot = result["optimization"]["stage1_geometric"]["geometric_metrics"]["spot_rms_um"]
    coarse = result["optimization"]["stage2_mtf"]["coarse_mtf"]
    assert math.isfinite(spot) and spot >= 0
    assert math.isfinite(coarse["mtf_auc_normalized"])
    assert 0 <= coarse["mtf_auc_normalized"] <= 1.0
    for condition in coarse["conditions"]:
        assert abs(condition["mtf_tangential_x"][0] - 1.0) < 1e-5
        assert abs(condition["mtf_sagittal_y"][0] - 1.0) < 1e-5
    assert result["J"] is None and not result["score_valid"]
    assert result["training_eligible"] is False

    atomic_json(a.output, result)
    print(json.dumps({
        "case": case["name"], "backend": result["backend"], "device": result["device"],
        "gap_count": len(info["gap_indices"]), "spot_rms_um": spot,
        "coarse_mtf_auc_normalized": coarse["mtf_auc_normalized"],
        "J": result["J"], "state": result["state"],
    }, allow_nan=False))


if __name__ == "__main__":
    main()
