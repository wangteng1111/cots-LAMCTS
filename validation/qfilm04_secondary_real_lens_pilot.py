#!/usr/bin/env python3
"""Independent seven public patent camera-lens examples: strict import & Qfilm0.4-g.
Source Zemax files downloaded from attributed author at runtime only.
Never assert patent design is exactly the commercial production lens.
"""
from __future__ import annotations
import os,sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from validation.qfilm_deeplens_same_prescription import BK7
from deeplens import GeoLens
from evaluator.qfilm02 import Config,_surfaces_at
from evaluator.optical_primitives import cardinal
from evaluator.qfilm03_physical_mode import PhysicalHuygensConfig
from evaluator.qfilm04 import evaluate
from validation.qfilm04_real_prescription_import import (
 import_geolens_lens,extract_full_precision_lens)
from validation.qfilm04_secondary_patent_sources import ZEMAX_SAMPLES,load_remote_zmx
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm04_independent_patents"
OUT.mkdir(parents=True,exist_ok=True)
PATH=OUT/"7_independent_camera_patents_gpu_pilot.json"
def main():
 report={"schema":"Qfilm04-independent-seven-patent-real-product-named-v1",
     "source_repo":"Amarthgul/ZemaxSims",
     "patent_example_not_equivalent_to_production":True,
     "actual_manufactured_mtf_reference":False,
     "cases":[]}
 for name in ZEMAX_SAMPLES:
  c={"identifier":name};report["cases"].append(c);t=time.perf_counter()
  try:
   zmx,provenance=load_remote_zmx(name,OUT/"input_zmx")
   c["provenance"]=provenance
   lens=GeoLens(filename=zmx,device="cuda:0",dtype=torch.float64)
   raw=extract_full_precision_lens(lens,provenance["url"])
   p=import_geolens_lens(lens,raw,name)
   c.update(import_ok=True,n_surfaces=len(p["surfaces"]),
     z_stop=p["stop_z_mm"],z_film=p["image_z_mm"],
     EFL_deeplens_550=float(lens.foclen),
     EFL_Qfilm_550=float(cardinal(_surfaces_at(p["surfaces"],550.))[0]))
   cfg=Config(fields=2,pupil_n=256,fft_n=1024,max_fields_per_chunk=1,
      wavelengths_nm=(550.,),spectral_weights=(1.,),
      target_efl_mm=p["design_spec"]["efl_target_mm"])
   with torch.inference_mode():r=evaluate(p,cfg,PhysicalHuygensConfig(
      pupil_n=256,detector_n=128,detector_pitch_um=5.),backend="g")
   c.update(solver_ok=True,quality_valid=r["quality"]["valid"],
     J_task=r["quality"]["J_task"],seconds=r["timings"]["total_seconds"],
     Qfields=r["Q_image"][:,[0,1,2,10,11,20,21,30]].tolist())
  except Exception as e:
   c["solver_ok"]=False;c["error"]=repr(e);c["traceback"]=traceback.format_exc()[-1800:]
  c["elapsed_sec"]=time.perf_counter()-t
  print("INDEPENDENT_PATENT",name,json.dumps({
   k:v for k,v in c.items() if k not in ("traceback","Qfields")}),flush=True)
  PATH.write_text(json.dumps(report,indent=2)+"\n")
 print("INDEPENDENT_PATENT_SUMMARY",json.dumps({"candidates":len(report["cases"]),
   "import_ok":sum(x.get("import_ok",False) for x in report["cases"]),
   "solver_ok":sum(x.get("solver_ok",False) for x in report["cases"])}),flush=True)
if __name__=="__main__":main()
