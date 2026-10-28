#!/usr/bin/env python3
"""Pilot all 11 REAL product-named camera patent examples with Qfilm0.4-g.

Stage A: strict importing geometry, stop, material dispersion, aspheric layout.
Stage B: one physical 550nm fixed-film verification at h=0 and h=1.
A negative result is reported as failure, not invented MTF ground truth.
"""
from __future__ import annotations
import sys,os,json,time,traceback,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from validation.qfilm_deeplens_same_prescription import BK7  # pinned DeepLens loader shim
from deeplens import GeoLens
from evaluator.qfilm02 import Config,_surfaces_at
from evaluator.optical_primitives import cardinal
from evaluator.qfilm03_physical_mode import PhysicalHuygensConfig
from evaluator.qfilm04 import evaluate
from validation.qfilm04_real_prescription_import import CAMERA_SAMPLES,import_geolens_lens,lens_metadata,IncompatibleLens
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm04_real_catalog"
OUT.mkdir(parents=True,exist_ok=True);FILE=OUT/"11_product_prescriptions_gpu_pilot.json"
DEEP_ROOT=Path("/var/lib/cots-lamcts/deeplens_upstream/datasets/lenses/camera")
def main():
 d={"schema":"Qfilm0.4-11-real-product-related-prescriptions-pilot-v1",
    "source_repo":"vccimaging/DeepLens","source_sha":"911bb9d334fce7e20a4407c4506ac683b52ff246",
    "cases":[],"check":"single green 550nm, field 0 and 21.63mm",
    "real_production_MTF_measured":False,"training_eligible":False}
 for name in CAMERA_SAMPLES:
  o={"name":name};d["cases"].append(o);start=time.perf_counter()
  try:
   path=DEEP_ROOT/(name+".json")
   if not path.exists():raise FileNotFoundError(path)
   raw=json.loads(path.read_text());o.update(lens_metadata(raw,name))
   lens=GeoLens(filename=str(path),device="cuda:0",dtype=torch.float64)
   p=import_geolens_lens(lens,raw,name)
   o.update(import_ok=True,efl_qfilm_green=float(cardinal(_surfaces_at(p["surfaces"],550.))[0]),
       efl_deeplens_green=float(lens.foclen),stop_z_mm=p["stop_z_mm"],
       film_z_mm=p["image_z_mm"],stop_diameter_mm=p["design_spec"]["stop_diameter_mm"])
   cfg=Config(fields=2,pupil_n=256,fft_n=1024,max_fields_per_chunk=1,
      wavelengths_nm=(550.,),spectral_weights=(1.,),
      target_efl_mm=p["design_spec"]["efl_target_mm"])
   phy=PhysicalHuygensConfig(pupil_n=256,detector_n=128,detector_pitch_um=5.)
   with torch.inference_mode():r=evaluate(p,cfg,phy,backend="g")
   o.update(solver_ok=True,J_task=r["quality"]["J_task"],quality_valid=r["quality"]["valid"],
     Qfields=r["Q_image"][:,[0,1,2,10,11,12,20,21,22,30]].tolist(),
     no_training_certification=not r["training_eligible"],
     solver_seconds=r["timings"]["total_seconds"])
  except Exception as e:
   o["solver_ok"]=False;o["error"]=repr(e);o["traceback"]=traceback.format_exc()[-1900:]
  o["elapsed_seconds"]=time.perf_counter()-start
  print("REAL_LENS_PILOT",name,
   json.dumps({k:v for k,v in o.items() if k not in ("traceback","Qfields")}),flush=True)
  FILE.write_text(json.dumps(d,indent=2)+"\n")
 print("PILOT_COMPLETE",json.dumps({"n":len(d["cases"]),
   "import_ok":sum(x.get("import_ok",False) for x in d["cases"]),
   "solver_ok":sum(x.get("solver_ok",False) for x in d["cases"]),
   "quality_valid":sum(x.get("quality_valid",False) for x in d["cases"])}),flush=True)
if __name__=="__main__":main()
