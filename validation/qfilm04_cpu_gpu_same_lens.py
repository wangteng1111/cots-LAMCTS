#!/usr/bin/env python3
"""Same physical optical prescription, Qfilm0.4-g vs Qfilm0.4-c parity.

The only permitted difference is tensor execution device. Both use torch
float64, physical pupil256, 128x128 film PSF, 550nm, center and edge fields.
"""
from __future__ import annotations
import sys,os,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from validation.qfilm_deeplens_same_prescription import BK7
from deeplens import GeoLens
from evaluator.qfilm02 import Config
from evaluator.qfilm03_physical_mode import PhysicalHuygensConfig
from evaluator.qfilm04 import evaluate
from validation.qfilm04_real_prescription_import import import_geolens_lens
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm04_cpu_gpu_parity"
OUT.mkdir(parents=True,exist_ok=True)
FILE=OUT/"real_ef50mm_f1p8_same_physics_CPU_GPU.json"
def main():
 torch.set_num_threads(12)
 path=Path("/var/lib/cots-lamcts/deeplens_upstream/datasets/lenses/camera/ef50mm_f1.8.json")
 p0=json.loads(path.read_text())
 lens=GeoLens(filename=str(path),device="cuda:0",dtype=torch.float64)
 p=import_geolens_lens(lens,p0,"Canon EF50mm f/1.8 published patent example")
 cfg=Config(fields=2,pupil_n=256,fft_n=1024,max_fields_per_chunk=1,
    wavelengths_nm=(550.,),spectral_weights=(1.,),
    target_efl_mm=p["design_spec"]["efl_target_mm"])
 phy=PhysicalHuygensConfig(pupil_n=256,detector_n=128,
    detector_pitch_um=5.,detector_block=192)
 out={"schema":"Qfilm0.4-c-vs-g-same-math-v1","case":"Canon EF50mm f/1.8 patent example",
      "config_fields":2,"pupil":256,"wavelength_nm":[550.],
      "training_eligible":False}
 for backend in ("g","c"):
  t=time.perf_counter()
  try:
   with torch.inference_mode():r=evaluate(p,cfg,phy,backend=backend)
   out[backend]={"variant":r["variant"],"Q_image":r["Q_image"].tolist(),
     "quality":r["quality"],"timings":{"wall_seconds":time.perf_counter()-t,
       "reported_seconds":r["timings"]["total_seconds"]},
     "physics_certified":r["physics_certified"],
     "training_eligible":r["training_eligible"]}
  except Exception as e:out[backend]={"error":repr(e),"traceback":traceback.format_exc()[-2400:]}
  print("CPU_GPU_PARITY",backend,json.dumps({k:v for k,v in out[backend].items()
    if k not in ("Q_image","traceback")}),flush=True)
  FILE.write_text(json.dumps(out,indent=2)+"\n")
 if "error" in out["g"] or "error" in out["c"]:
  print("CPU_GPU_PARITY_FAILED",flush=True);raise RuntimeError("CPU/GPU solver execution error")
 a=np.asarray(out["g"]["Q_image"]);b=np.asarray(out["c"]["Q_image"])
 mtf_diff=float(np.max(np.abs(a[:,10:30]-b[:,10:30])))
 jg=out["g"]["quality"]["J_task"];jc=out["c"]["quality"]["J_task"]
 out["parity"]={"max_MTF_abs_difference":mtf_diff,"J_abs_difference":abs(jg-jc) if jg is not None and jc is not None else None,
                "valid_identical":out["g"]["quality"]["valid"]==out["c"]["quality"]["valid"]}
 FILE.write_text(json.dumps(out,indent=2)+"\n")
 assert mtf_diff<2e-6,out["parity"]
 assert out["parity"]["valid_identical"]
 print("CPU_GPU_PARITY_PASS",json.dumps(out["parity"]),flush=True)
if __name__=="__main__":main()
