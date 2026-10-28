#!/usr/bin/env python3
"""Common-incident-ray Qfilm/DeepLens audit on 18 real-model-associated designs.

CPU, 32x32 pupil, center/half/edge at 550 and 670nm. Measures geometry, OPL,
fixed-film OPD, phase, and Jacobian on valid common interior stencils. This
does not certify PSF integration, unknown glass dispersion, or retail lenses.
"""
from __future__ import annotations
import argparse,json,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from validation.qfilm_deeplens_same_prescription import BK7
from validation.qfilm03_matched_ray_wavefront_audit import run_case
from validation.qfilm04_real_prescription_import import (
    CAMERA_SAMPLES,import_geolens_lens,extract_full_precision_lens)
from validation.qfilm04_secondary_patent_sources import ZEMAX_SAMPLES,SOURCE_SHA
from deeplens import GeoLens

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deeplens-root",type=Path,required=True)
    ap.add_argument("--zmx-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    d={"schema":"Qfilm04-18-real-source-common-ray-CPU-audit-v1",
       "deeplens_sha":"911bb9d334fce7e20a4407c4506ac683b52ff246",
       "zmx_sha":SOURCE_SHA,"pupil_n":32,"certified":False,
       "training_eligible":False,"cases":[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for kind,name in [("camera",n) for n in CAMERA_SAMPLES]+[("zmx",n) for n in ZEMAX_SAMPLES]:
        c={"name":name,"source_kind":kind,"rays":[]};d["cases"].append(c)
        try:
            path=(args.deeplens_root/"datasets/lenses/camera"/(name+".json")
                  if kind=="camera" else args.zmx_root/ZEMAX_SAMPLES[name])
            lens=GeoLens(filename=str(path),device="cpu",dtype=torch.float64)
            raw=json.loads(path.read_text()) if kind=="camera" else extract_full_precision_lens(lens,str(path.name))
            p=import_geolens_lens(lens,raw,name)
            for h in (0.,.5,1.):
                for w in (550.,670.):
                    try:
                        r=run_case(lens,p,w,h,n=32,device="cpu")
                    except Exception as exc: r={"lambda_nm":w,"h":h,"error":repr(exc)}
                    c["rays"].append(r)
        except Exception as exc: c.update(error=repr(exc),traceback=traceback.format_exc()[-1800:])
        args.output.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")
        print("REAL_WAVEFRONT_AUDIT",name,len(c["rays"]),
              sum("error" not in r for r in c["rays"]),c.get("error"),flush=True)
    d["summary"]={"lenses":len(d["cases"]),"attempted_ray_cases":sum(len(c["rays"]) for c in d["cases"]),
       "completed_ray_cases":sum("error" not in r for c in d["cases"] for r in c["rays"])}
    args.output.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")
    print("REAL_WAVEFRONT_SUMMARY",d["summary"],flush=True)

if __name__=="__main__":main()
