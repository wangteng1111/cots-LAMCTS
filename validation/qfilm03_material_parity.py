#!/usr/bin/env python3
"""Qfilm0.3 physical-material parity; conservatively reject mislabeled glass."""
from __future__ import annotations
import sys,json,os,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluator.qfilm02 import Config
from evaluator.qfilm03 import evaluate,ValidationConfig
from validation.qfilm_deeplens_same_prescription import thin_lens,BK7
cfg=Config(fields=3,pupil_n=64,fft_n=256,max_fields_per_chunk=3,
           wavelengths_nm=(510.,550.,630.),spectral_weights=(.25,.5,.25),
           frame_width_mm=8,frame_height_mm=6,target_efl_mm=50)
v=ValidationConfig(mode="verify",wavelength_nm=550.,fields=(0.,),
    huygens_spp=16384,base_backend="torch")
p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7;p["surfaces"][0]["glass_name"]="N-BK7"
good=evaluate(p,cfg,device="cuda:0",validation=v)
assert good["reference"]["status"]=="checked_subset",good["reference"]
assert good["reference"]["material_parity"][0]["max_index_error"]<2e-6
wrong=copy.deepcopy(p)
wrong["surfaces"][0]["glass_name"]="N-SF6"
bad=evaluate(wrong,cfg,device="cuda:0",validation=v)
assert bad["reference"]["status"]=="material_mismatch",bad["reference"]
assert not good["training_eligible"] and not bad["training_eligible"]
result={"schema":"qfilm03-supplier-glass-parity-v1",
        "matching":{"status":good["reference"]["status"],"material":good["reference"]["material_parity"]},
        "rejected":{"status":bad["reference"]["status"],"reason":bad["reference"]["reason"]}}
path=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_materials.json"
path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+"\n")
print("MATERIAL_PARITY",json.dumps(result),flush=True)
