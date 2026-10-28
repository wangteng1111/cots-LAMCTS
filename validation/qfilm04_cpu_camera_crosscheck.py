#!/usr/bin/env python3
"""CPU center-field screening vs independent native DeepLens Huygens.

11 public camera prescriptions, green and red, unchanged scalar propagators.
Low-SPP reference is a discrepancy detector, NOT a convergence certificate.
PSF pitch 5um is a legacy comparison setting, NOT certified fine sampling
for wide-aperture photographic lenses. Both errors must remain visible.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time, types, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from validation.qfilm_deeplens_same_prescription import BK7
from validation.qfilm03_huygens_safechunk import patch_huygens
from validation.qfilm04_real_prescription_import import CAMERA_SAMPLES, import_geolens_lens
from evaluator.qfilm03_physical_huygens import Propagation, eval_fixed_film, coherent_otf_axis
from deeplens import GeoLens

F=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deeplens-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--spp",type=int,default=16384)
    ap.add_argument("--names",nargs="*",default=list(CAMERA_SAMPLES))
    args=ap.parse_args()
    torch.set_num_threads(4);torch.set_default_dtype(torch.float64)
    _,chunked,patch=patch_huygens(256)
    GeoLens.psf_huygens=chunked
    report={"schema":"Qfilm04-eleven-camera-CPU-monochromatic-screening-v1",
        "upstream_sha":"911bb9d334fce7e20a4407c4506ac683b52ff246",
        "reference_spp":args.spp,"batch_only_patch":patch,
        "field_h":0.,"wavelength_nm":[550.,670.],"frequency_lpmm":F,
        "pupil_n":256,"detector_n":128,"detector_pitch_um":5.,
        "reference_converged":False,"detector_sampling_certified":False,
        "manufactured_lens_empirical_validation":False,
        "training_eligible":False,"cases":[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save(): args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for name in args.names:
        c={"name":name,"wavelengths":[]};report["cases"].append(c)
        try:
            path=args.deeplens_root/"datasets/lenses/camera"/(name+".json")
            raw=json.loads(path.read_text())
            lens=GeoLens(filename=str(path),device="cpu",dtype=torch.float64)
            p=import_geolens_lens(lens,raw,name)
            c.update(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     reference_description=raw.get("info"),f_number=float(lens.fnum))
            # Square native detector, EXACT same physical pitch, no rounding.
            lens.set_sensor(sensor_size=(40.,40.),sensor_res=(8000,8000))
            if abs(float(lens.pixel_size)-.005)>1e-12:raise ValueError("wrong pixel pitch")
            def exact_center(self,points_obj,method="chief_ray",ray=None):
                return torch.zeros((1,2),device="cpu",dtype=torch.float64)
            lens.psf_center=types.MethodType(exact_center,lens)
            for w in (550.,670.):
                t=time.perf_counter();entry={"wavelength_nm":w}
                c["wavelengths"].append(entry)
                try:
                    with torch.inference_mode():
                        q=eval_fixed_film(p,w,0.,Propagation(pupil_n=256,pixel_n=128,
                            pixel_pitch_um=5.,pixel_block=64,weight_mode="sqrt_exit_jac"),device="cpu")
                        torch.manual_seed(20261028+int(w))
                        dpsf=lens.psf_huygens(points=[0.,0.,-1e6],ks=128,
                                             wvln=w*.001,spp=args.spp,recenter=True)
                    qa=np.abs(coherent_otf_axis(q["psf"],.005,F))
                    da=np.abs(coherent_otf_axis(dpsf,.005,F))
                    if not np.all(np.isfinite(qa)) or not np.all(np.isfinite(da)):
                        raise ValueError("nonfinite PSF/MTF")
                    delta=np.abs(qa-da)
                    edge=np.ones((128,128),bool);edge[4:-4,4:-4]=False
                    entry.update(qfilm_mtf_ts=qa.tolist(),deeplens_mtf_ts=da.tolist(),
                        max_abs_mtf_difference=float(delta.max()),
                        mean_abs_mtf_difference=float(delta.mean()),
                        qfilm_border_flux_fraction=float(q["psf"].numpy()[edge].sum()),
                        deeplens_border_flux_fraction=float(dpsf.numpy()[edge].sum()),
                        qfilm_valid_rays=q["valid_rays"],
                        qfilm_physical_exit_z_mm=q["exit_z_mm"],
                        native_exit_pupil_z_mm=float(lens.get_exit_pupil()[0]),
                        bad_jacobian_fraction=q["bad_jacobian_fraction"],
                        approximate_optical_nyquist_pitch_um=w*.001*float(lens.fnum)/2,
                        elapsed_seconds=time.perf_counter()-t)
                except Exception as exc:
                    entry.update(error=repr(exc),traceback=traceback.format_exc()[-2000:])
                save()
                print("CPU_CROSSCHECK",name,w,json.dumps({k:v for k,v in entry.items()
                    if k not in ("qfilm_mtf_ts","deeplens_mtf_ts","traceback")}),flush=True)
        except Exception as exc: c.update(error=repr(exc),traceback=traceback.format_exc()[-2000:])
        save()
    report["summary"]={"lenses":len(report["cases"]),"comparisons":sum(
        "max_abs_mtf_difference" in e for c in report["cases"] for e in c["wavelengths"])}
    save();print("CPU_CROSSCHECK_COMPLETE",report["summary"],flush=True)

if __name__=="__main__":main()
