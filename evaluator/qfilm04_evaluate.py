"""Qfilm0.4 convergence-gated orchestration, separate from the legacy scorer."""
from __future__ import annotations
from dataclasses import asdict
import time
import numpy as np
import torch
from evaluator.qfilm02 import evaluate as baseline_evaluate, score_image
from evaluator.qfilm04_wavefront import WavefrontError, MODEL_VERSION, EXTRACTOR_VERSION
from evaluator.qfilm04_sampling import (make_plan, solve_condition, digest,
                                       combine_spectral_otf)


def evaluate_converged(source,config,physical,policy,device,resolved_plans=None,
                       checkpoint_callback=None,progress_callback=None):
    config.validate();physical.validate(config);policy.validate()
    if (not np.isclose(policy.pitch_um,physical.detector_pitch_um) or
        not np.isclose(policy.support_candidates_mm[0],physical.detector_n*physical.detector_pitch_um*.001)
        or physical.pupil_n!=policy.pupil_candidates[0]
        or not np.isclose(physical.max_bad_exit_jacobian_fraction,policy.max_bad_jacobian_fraction)
        or physical.amplitude_mode!="sqrt_exit_jac"):
        raise ValueError("physical detector and convergence policy must have identical initial pitch/support")
    if len(config.frequencies_lpmm)!=10:
        raise ValueError("Q_image schema requires ten frequencies")
    start=time.perf_counter();diagnostics=[];plans=[]
    with torch.inference_mode():baseline=baseline_evaluate(source,config,device=device)
    q=np.asarray(baseline["Q_image"],dtype=float).copy()
    old_valid=q[:,30]>.5
    # Legacy quality remains explicitly provisional; no stale FFT MTF may escape.
    q[:,10:30]=np.nan;q[:,30]=0.
    supplied=None
    if resolved_plans is not None:
        supplied={(float(p["field_h"]),float(p["wavelength_nm"])):p for p in resolved_plans}
        if len(supplied)!=config.fields*len(config.wavelengths_nm):
            raise ValueError("complete, unique frozen plans required for every field/wavelength")
    for fi,h in enumerate(np.linspace(0,1,config.fields)):
        field=[]
        for wave in config.wavelengths_nm:
            try:
                if supplied is None:
                    plan=make_plan(source,float(wave),float(h),config.frequencies_lpmm,policy,
                        frame_mm=(config.frame_width_mm,config.frame_height_mm),progress_callback=progress_callback)
                else:plan=supplied[(float(h),float(wave))]
                if plan["frequencies_lpmm"]!=list(config.frequencies_lpmm):
                    raise ValueError("frozen extraction frequencies mismatch")
                if plan["frame_mm"]!=[config.frame_width_mm,config.frame_height_mm]:
                    raise ValueError("frozen physical frame mismatch")
                sample=solve_condition(source,plan,policy,device,physical.detector_block,
                    checkpoint_callback=checkpoint_callback,progress_callback=progress_callback)
                plans.append(sample["resolved_sampling_plan"])
            except WavefrontError as exc:
                sample={"state":exc.code,"error":str(exc),"flags":{
                    "geometry_valid":exc.code!="NO_TRANSMITTED_WAVEFRONT",
                    "wavefront_supported":False,"pupil_converged":False,
                    "support_converged":False,"pitch_converged":False,"joint_converged":False,
                    "reference_validated":False,"label_valid":False,"empirically_calibrated":False}}
            sample.update(field_h=float(h),wavelength_nm=float(wave),field_index=fi)
            field.append(sample);diagnostics.append(sample)
        complete=all(s["state"]=="numerically_converged" for s in field)
        if complete:
            samples=[s["selected"] for s in field]
            otf=combine_spectral_otf(samples,config.wavelengths_nm,config.spectral_weights,
                                     config.frequencies_lpmm)
            q[fi,10:20]=abs(otf[:,0]);q[fi,20:30]=abs(otf[:,1])
            # Keep historical task constraints; blocked-chief geometry can still
            # yield physical diagnostics while its legacy J remains unavailable.
            q[fi,30]=float(old_valid[fi] and np.isfinite(q[fi]).all())
    all_numerical=all(s["state"]=="numerically_converged" for s in diagnostics)
    constraints_passed=(len(plans)==config.fields*len(config.wavelengths_nm) and
        all(p["pilot_geometry"]["geometric_pupil_fill"]>.20 for p in plans))
    quality=score_image(q,baseline["area_weights"]) if all_numerical else {
        "valid":False,"J_task":None,"reason":"NUMERICAL_CONVERGENCE_REQUIRED"}
    quality.update(provisional=True,label_valid=False)
    flags={"geometry_valid":all(s["flags"]["geometry_valid"] for s in diagnostics),
        "wavefront_supported":all(s["flags"]["wavefront_supported"] for s in diagnostics),
        **{k:all(s["flags"][k] for s in diagnostics) for k in (
            "pupil_converged","support_converged","pitch_converged","joint_converged")},
        "constraints_passed":constraints_passed,"legacy_objective_requirements_passed":bool(old_valid.all()),
        "reference_validated":False,
        "label_valid":False,"empirically_calibrated":False}
    out=dict(baseline)
    out.update(Q_image=q,quality=quality,legacy_quality_provisional=baseline["quality"],
        objective_version="legacy-task-v1-convergence-gated",validity=flags,
        physical_diagnostics=diagnostics,resolved_sampling_plans=plans,
        convergence_policy=asdict(policy),spectral_model="normalized_fixed_weights",
        physics_status="provisional_scalar_model_with_explicit_numerical_gates",
        physics_hash=digest({"source":source,"config":asdict(config),"policy":asdict(policy),
            "plans":[p["plan_hash"] for p in plans],"model":MODEL_VERSION,"extractor":EXTRACTOR_VERSION}),
        execution_id=__import__("uuid").uuid4().hex,
        execution_records=[{"execution_id":s.get("execution_id"),"actual_device":s.get("actual_device"),
            "state":s["state"],"plan_hash":s.get("plan_hash")} for s in diagnostics],
        timings={"total_seconds":time.perf_counter()-start},
        pending_model_validation=["input ray-tube energy closure", "independent converged propagation reference",
            "exit-plane invariance", "film spectral response", "measured lens calibration"],
        training_eligible=False,physics_certified=False,certified=False)
    return out
