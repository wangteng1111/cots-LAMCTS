#!/usr/bin/env python3
"""Qfilm0.3 verify-mode regression: no stochastic chief; 550 EFL parity;
multi-color total J remains final but DeepLens reference uses true monochromatic PSF.
"""
from __future__ import annotations
import os,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from evaluator.qfilm02 import Config
from evaluator.qfilm03 import evaluate,ValidationConfig
from validation.qfilm_deeplens_same_prescription import thin_lens,internal_stop_positive_lens,BK7
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_fixed_chief_verify"
OUT.mkdir(parents=True,exist_ok=True)
FILE=OUT/"corrected_reference_chief_and_mono_mtf.json"
def main():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 d={"schema":"Qfilm03-verify-deterministic-chief-mono-test-v1","cases":[]}
 for name,p in (("singlet",p),("internal_stop_asphere",internal_stop_positive_lens())):
  cfg=Config(fields=11,pupil_n=256,fft_n=1024,max_fields_per_chunk=3,
      wavelengths_nm=(550.,670.),spectral_weights=(.5,.5),
      frame_width_mm=36.,frame_height_mm=24.,
      target_efl_mm=p["design_spec"]["efl_target_mm"])
  saved=None
  for mode in ("native_stochastic","deterministic_physical_chief"):
   refcfg=ValidationConfig(mode="verify",fields=(0.,.1),wavelength_nm=670.,
       sensor_resolution=9600,physical_sensor_width_mm=48.,psf_kernel=128,
       huygens_spp=1000000,base_backend="torch",reference_centering=mode)
   r=evaluate(p,cfg,device="cuda:0",validation=refcfg)
   rr=r["reference"]
   if rr["status"]!="checked_subset":raise AssertionError(str(rr))
   if saved is None:saved=r["quality"]["J_task"]
   assert abs(saved-r["quality"]["J_task"])<1e-12
   assert rr["reference_Q_image_source"]=="single_wavelength_actual_film"
   out={"lens":name,"mode":mode,"reference_status":rr["status"],
       "spectral_J_unchanged":r["quality"]["J_task"],"efl_delta_mm":rr["efl_abs_difference_mm"],
       "max_abs_mono_MTF_vs_fft":rr["mtf_max_abs"],"reference_fields":rr["fields"]}
   d["cases"].append(out)
   print("CENTER_VERIFY",name,mode,
       "maximum_diff",rr["mtf_max_abs"],"efl_error",rr["efl_abs_difference_mm"],flush=True)
   FILE.write_text(json.dumps(d,indent=2)+"\n")
 print("CORRECTED_VERIFY_PASS",len(d["cases"]),flush=True)
if __name__=="__main__":main()
