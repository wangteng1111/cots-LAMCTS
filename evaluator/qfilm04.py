"""Qfilm0.4: common-physics CPU (-c) and GPU (-g) optical evaluator.

Both modes execute the SAME 256+ pupil, physical fixed-film Huygens/Jacobian
math, exactly the SAME seven-color physical OTF accumulation and final J.
Only tensor execution device differs. Never silently substitute a surrogate.

Scientific status remains provisional: measured production PSF/MTF, ray-tube
energy transport for arbitrary pupils, detector window and film photometry
must still be certified case by case.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
from typing import Literal
import json, argparse, numpy as np, torch
from evaluator.qfilm02 import Config
from evaluator.qfilm03_physical_mode import (
    PhysicalHuygensConfig,evaluate as physical_evaluate)

VERSION="Qfilm0.4"
CPU="Qfilm0.4-c"
GPU="Qfilm0.4-g"

def evaluate(source,config:Config=Config(),physical:PhysicalHuygensConfig=PhysicalHuygensConfig(),
             backend:Literal["c","g"]="g",gpu_index:int=0):
    """Evaluate complete candidate's physical image-plane final score on CPU or GPU."""
    if backend not in ("c","g"):
        raise ValueError("Qfilm0.4 backend must be 'c' (CPU) or 'g' (GPU)")
    if backend=="g" and not torch.cuda.is_available():
        raise RuntimeError("Qfilm0.4-g explicitly requires available CUDA; no implicit CPU fallback")
    device="cpu" if backend=="c" else f"cuda:{int(gpu_index)}"
    r=physical_evaluate(source,config=config,device=device,physical=physical)
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
    ap.add_argument("--detector",type=int,default=128)
    ap.add_argument("--gpu-index",type=int,default=0)
    ap.add_argument("--output",help="Optional machine-readable full Q/J JSON output")
    args=ap.parse_args()
    data=json.loads(open(args.prescription,encoding="utf-8").read())
    cfg=Config(fields=args.fields,pupil_n=256,fft_n=1024,
        target_efl_mm=data["design_spec"]["efl_target_mm"])
    phy=PhysicalHuygensConfig(pupil_n=args.pupil,detector_n=args.detector)
    out=evaluate(data,config=cfg,physical=phy,
        backend="c" if args.c else "g",gpu_index=args.gpu_index)
    def conv(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,(np.integer,np.floating)):return x.item()
        raise TypeError(f"Cannot JSON encode {type(x)}")
    s=json.dumps(out,default=conv,allow_nan=False)
    if args.output:
        from pathlib import Path
        p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(s+"\n",encoding="utf-8")
    print(json.dumps({"version":out["variant"],"valid":out["quality"]["valid"],
       "J_task":out["quality"]["J_task"],"fields":args.fields,
       "pupil":args.pupil,"device":out["numerical_execution_device"],
       "seconds":out["timings"]["total_seconds"],
       "training_eligible":False},allow_nan=False))
if __name__=="__main__":main()
