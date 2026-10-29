"""Qfilm0.4: common-physics CPU (-c) and GPU (-g) optical evaluator.

Both modes execute the SAME 256+ pupil, physical fixed-film Huygens/Jacobian
math, exactly the SAME seven-color physical OTF accumulation and final J.
Only tensor execution device differs. Never silently substitute a surrogate.

Scientific status remains provisional: measured production PSF/MTF, ray-tube
energy transport for arbitrary pupils, detector window and film photometry
must still be certified case by case.
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Literal
import json, argparse, hashlib, numpy as np, torch
from evaluator.qfilm02 import Config
from evaluator.qfilm03_physical_mode import (
    PhysicalHuygensConfig,evaluate as physical_evaluate)
from evaluator.qfilm04_sampling import ConvergencePolicy
from evaluator.qfilm04_evaluate import evaluate_converged

VERSION="Qfilm0.4"
CPU="Qfilm0.4-c"
GPU="Qfilm0.4-g"
# Retain the 0.64 mm support while removing the legacy 5 um point grid.
# Changing pitch alone with 128 pixels would shrink support twenty-fold.
DEFAULT_PHYSICAL=PhysicalHuygensConfig(detector_n=2560,detector_pitch_um=.25,
                                     detector_block=64)

def evaluate(source,config:Config=Config(),physical:PhysicalHuygensConfig=DEFAULT_PHYSICAL,
             backend:Literal["c","g"]="g",gpu_index:int=0,
             convergence:ConvergencePolicy|None=ConvergencePolicy(),resolved_plans=None,
             checkpoint_callback=None,progress_callback=None):
    """Evaluate complete candidate's physical image-plane final score on CPU or GPU."""
    if backend not in ("c","g"):
        raise ValueError("Qfilm0.4 backend must be 'c' (CPU) or 'g' (GPU)")
    if backend=="g" and not torch.cuda.is_available():
        raise RuntimeError("Qfilm0.4-g explicitly requires available CUDA; no implicit CPU fallback")
    device="cpu" if backend=="c" else f"cuda:{int(gpu_index)}"
    if convergence is None:
        r=physical_evaluate(source,config=config,device=device,physical=physical)
        r["sampling_mode"]="legacy_fixed_grid_unconverged"
    else:
        r=evaluate_converged(source,config,physical,convergence,device,resolved_plans,
            checkpoint_callback,progress_callback)
        r["sampling_mode"]="adaptive_convergence_gated"
    settings={"version":VERSION,"image":asdict(config),"physical":asdict(physical),
        "convergence":None if convergence is None else asdict(convergence)}
    # Physical pupil, image pitch and window must participate in cache identity.
    # CPU/GPU variants share this identity because they execute the same math.
    r["config_hash"]=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    r["physical_config"]=asdict(physical)
    # A 100lp/mm detector Nyquist exceeding the requested 80lp/mm is NOT
    # sufficient: the underlying optical PSF may contain much higher frequencies.
    # This is a paraxial warning estimate, not a finite-NA certification rule.
    nyquist_pitch_um=min(config.wavelengths_nm)*1e-3*float(r["fno"])/2.
    r["detector_sampling"]={"paraxial_optical_nyquist_pitch_um":nyquist_pitch_um,
        "actual_pitch_um":physical.detector_pitch_um,
        "pitch_to_paraxial_limit_ratio":physical.detector_pitch_um/nyquist_pitch_um,
        "sampling_converged":all(r.get("validity",{}).get(k,False) for k in (
            "pupil_converged","support_converged","pitch_converged","joint_converged")),
        "interpretation":"diagnostic estimate; requires independent pitch and support convergence"}
    r.update(schema="QfilmResult-0.4",version=VERSION,variant=f"{VERSION}-{backend}",
             math_backend="physical_fixed_film_exit_Huygens",
             numerical_execution_device=device,
             cpu_gpu_physics_identical=True,
             trained_surrogate_used=False,
             physics_certified=False,certified=False,training_eligible=False)
    return r

def main():
    ap=argparse.ArgumentParser(description="Qfilm0.4 fixed-film physical Huygens evaluator")
    be=ap.add_mutually_exclusive_group(required=True)
    be.add_argument("-c",action="store_true",help="Qfilm0.4-c: deterministic CPU torch float64")
    be.add_argument("-g",action="store_true",help="Qfilm0.4-g: CUDA GPU torch float64")
    ap.add_argument("--prescription",required=True,help="Qfilm-native sequential optical prescription JSON")
    ap.add_argument("--fields",type=int,default=100)
    ap.add_argument("--pupil",type=int,default=256)
    ap.add_argument("--detector",type=int,default=2560)
    ap.add_argument("--detector-pitch-um",type=float,default=.25)
    ap.add_argument("--detector-block",type=int,default=64)
    ap.add_argument("--gpu-index",type=int,default=0)
    ap.add_argument("--legacy-fixed-grid",action="store_true",help="Replay historical, uncertified fixed-grid solver")
    ap.add_argument("--max-pupil",type=int,default=4096)
    ap.add_argument("--supports-mm",type=float,nargs="+",default=[.64,.96,1.28])
    ap.add_argument("--max-pair-evaluations",type=int,default=150_000_000_000_000)
    ap.add_argument("--resolved-plans",help="JSON list of frozen plans shared by CPU and GPU")
    ap.add_argument("--checkpoint-dir",help="Per-condition atomic checkpoints; defaults beside --output")
    ap.add_argument("--output",help="Optional machine-readable full Q/J JSON output")
    args=ap.parse_args()
    data=json.loads(open(args.prescription,encoding="utf-8").read())
    cfg=Config(fields=args.fields,pupil_n=256,fft_n=1024,
        target_efl_mm=data["design_spec"]["efl_target_mm"])
    phy=PhysicalHuygensConfig(pupil_n=args.pupil,detector_n=args.detector,
        detector_pitch_um=args.detector_pitch_um,detector_block=args.detector_block)
    levels=[];n=args.pupil
    while n<=args.max_pupil:levels.append(n);n*=2
    policy=None if args.legacy_fixed_grid else ConvergencePolicy(pupil_candidates=tuple(levels),
        support_candidates_mm=tuple(args.supports_mm),pitch_um=args.detector_pitch_um,
        max_pair_evaluations=args.max_pair_evaluations)
    plans=None if args.resolved_plans is None else json.loads(open(args.resolved_plans).read())
    from pathlib import Path
    from evaluator.qfilm04_io import atomic_json,json_safe
    checkpoint_dir=Path(args.checkpoint_dir or ((args.output+".checkpoints") if args.output else "artifacts/qfilm04_checkpoints"))
    def checkpoint(value):
        plan=value["resolved_sampling_plan"]
        name=f"h{plan['field_h']:.12g}-w{plan['wavelength_nm']:.12g}"
        atomic_json(checkpoint_dir/(name+"-plan.json"),plan)
        atomic_json(checkpoint_dir/(name+"-"+("c" if args.c else "g")+"-checkpoint.json"),value)
    out=evaluate(data,config=cfg,physical=phy,convergence=policy,resolved_plans=plans,
        backend="c" if args.c else "g",gpu_index=args.gpu_index,checkpoint_callback=checkpoint)
    def conv(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,(np.integer,np.floating)):return x.item()
        raise TypeError(f"Cannot JSON encode {type(x)}")
    # Persist missing provisional metrics explicitly; unavailable values never
    # become zeros or accepted optical labels.
    s=json.dumps(json_safe(out),default=conv,allow_nan=False)
    if args.output:
        from pathlib import Path
        p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(s+"\n",encoding="utf-8")
        if policy is not None:atomic_json(Path(args.output+".plans.json"),out["resolved_sampling_plans"])
    print(json.dumps({"version":out["variant"],"valid":out["quality"]["valid"],
       "J_task":out["quality"]["J_task"],"fields":args.fields,
       "pupil":args.pupil,"device":out["numerical_execution_device"],
       "seconds":out["timings"]["total_seconds"],
       "training_eligible":False},allow_nan=False))
if __name__=="__main__":main()
