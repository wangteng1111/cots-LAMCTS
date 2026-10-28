#!/usr/bin/env python3
"""Validate new Qfilm0.3 optional complete physical-Huygens Final-J evaluator
against previously acquired DeepLens full-frame 7-color PSF ground comparison.
"""
from __future__ import annotations
import os,sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from evaluator.qfilm02 import Config,WAVES,WEIGHTS,evaluate as q02
from evaluator.qfilm03 import evaluate_physical_huygens
from evaluator.qfilm03_physical_mode import PhysicalHuygensConfig
from validation.qfilm_deeplens_same_prescription import thin_lens,internal_stop_positive_lens,BK7
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_physical_mode_validation"
OUT.mkdir(parents=True,exist_ok=True)
FILE=OUT/"fullframe_physical_finalJ_regression.json"
REF=json.loads((Path(__file__).resolve().parents[1]/"validation/fixtures/qfilm03_fullframe_deeplens_poly.json").read_text())
def get_lenses():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 return {"singlet":p,"internal_stop_asphere":internal_stop_positive_lens()}
def main():
 result={"schema":"Qfilm03-physical-finalJ-v1","cases":{},"certified":False}
 for name,p in get_lenses().items():
  config=Config(fields=3,pupil_n=64,fft_n=256,max_fields_per_chunk=3,
        wavelengths_nm=WAVES,spectral_weights=WEIGHTS,
        frame_width_mm=36.,frame_height_mm=24.,
        target_efl_mm=p["design_spec"]["efl_target_mm"])
  start=time.perf_counter()
  with torch.inference_mode():old=q02(p,config,device="cuda:0")
  phy=evaluate_physical_huygens(p,config,device="cuda:0",
           physical=PhysicalHuygensConfig(pupil_n=128,detector_n=128,
             detector_pitch_um=5.,detector_block=384,
             amplitude_mode="sqrt_exit_jac",record_all_mtf=True))
  c={"seconds":time.perf_counter()-start,"old_J":old["quality"]["J_task"],
     "physical_J":phy["quality"]["J_task"],
     "old_schema":old["schema"],"physical_schema":phy["schema"],
     "physical_certified":phy["physics_certified"],"training_eligible":phy["training_eligible"],
     "per_field":[]}
  for i,(base,full) in enumerate(zip(old["Q_image"],phy["Q_image"])):
   target=REF["cases"][name][i]
   q=np.asarray(list(base[10:20])+list(base[20:30]))
   n=np.asarray(list(full[10:20])+list(full[20:30]))
   gt=np.asarray(target["MTF_T"]+target["MTF_S"])
   a=np.abs(q-gt);b=np.abs(n-gt)
   c["per_field"].append({"h":target["h"],
       "baseline_max_abs_vs_deeplens":float(a.max()),
       "physical_max_abs_vs_deeplens":float(b.max()),
       "baseline_mean_abs_vs_deeplens":float(a.mean()),
       "physical_mean_abs_vs_deeplens":float(b.mean()),
       "max_physical_vs_baseline":float(np.max(np.abs(n-q))),
       "quality_improvement_in_max_abs":float(a.max()-b.max())})
  assert phy["training_eligible"] is False and phy["physics_certified"] is False
  assert phy["quality"]["valid"] and np.isfinite(phy["quality"]["J_task"])
  assert phy["physical_metadata"]["field_wavelength_evaluations"]==21
  result["cases"][name]=c
  print("PHYSICAL_FINALJ",name,json.dumps(c),flush=True)
  FILE.write_text(json.dumps(result,indent=2)+"\n")
 print("ALL_PASS",json.dumps({"cases":list(result["cases"])}),flush=True)
if __name__=="__main__":main()
