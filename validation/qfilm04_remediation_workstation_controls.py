#!/usr/bin/env python3
"""Workstation controls: new-kernel CPU/CUDA arithmetic and edge-ray handling.

    The 8 um integration tiles are arithmetic tests, not full PSF certification.
"""
from pathlib import Path
import sys,argparse,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from evaluator.qfilm02 import FREQS
from evaluator.qfilm04_wavefront import (fixed_exit_plane,prepare_wavefront_bundle,
    integrate_fixed_film,WavefrontError)
from evaluator.qfilm04_io import atomic_json
from validation.qfilm04_fine_11_cpu_gpu import import_case,DEEP_SHA


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deeplens-root",type=Path,default=Path("/var/lib/cots-lamcts/deeplens_upstream"))
    ap.add_argument("--output-dir",type=Path,default=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts/qfilm04_controls")))
    ap.add_argument("--cpu-only",action="store_true")
    args=ap.parse_args();torch.set_num_threads(2);args.output_dir.mkdir(parents=True,exist_ok=True)
    if subprocess.check_output(["git","-C",str(args.deeplens_root),"rev-parse","HEAD"],text=True).strip()!=DEEP_SHA:
        raise ValueError("DeepLens source commit mismatch")
    report={"schema":"Qfilm04RemediationWorkstationControls-v1","cuda_available":torch.cuda.is_available(),
        "full_psf_validation":False,"training_eligible":False,"controls":[],"edges":[]}
    subprocess.run([sys.executable,str(ROOT/"validation/qfilm04_remediation_tests.py")],check=True)
    for name,wave in (("ef40mm_f2.8",430.),("ef40mm_f2.8",550.),("rf50mm_f1.8",670.)):
        p,sha=import_case(args.deeplens_root,name);exit_z=fixed_exit_plane(p)
        b=prepare_wavefront_bundle(p,wave,0.,256,exit_z)
        center=b.center_mm
        c=integrate_fixed_film(b,32,.25,FREQS,pixel_block=16,retain_intensity=True)
        entry={"name":name,"wavelength_nm":wave,"source_sha256":sha,"pupil_n":256,
            "support_mm":.008,"purpose":"arithmetic_tile_only","cpu_integral":c["raw_intensity_integral"]}
        if torch.cuda.is_available() and not args.cpu_only:
            gb=prepare_wavefront_bundle(p,wave,0.,256,exit_z,center,"cuda:0")
            g=integrate_fixed_film(gb,32,.25,FREQS,pixel_block=16,retain_intensity=True)
            otf=float(np.max(abs(c["otf"]-g["otf"])))
            intensity=float(np.max(abs(c["raw_intensity"]-g["raw_intensity"]))/np.max(c["raw_intensity"]))
            entry.update(gpu_actual_device="cuda:0",gpu_integral=g["raw_intensity_integral"],
                complex_otf_max_abs=otf,raw_intensity_relative_max_abs=intensity,
                active_nodes_equal=b.metadata["active_nodes"]==gb.metadata["active_nodes"],
                passed=otf<=1e-6 and intensity<=1e-6 and b.metadata["active_nodes"]==gb.metadata["active_nodes"])
        else:entry.update(state="CPU_ONLY" if args.cpu_only else "CUDA_UNAVAILABLE",passed=None)
        report["controls"].append(entry)
        atomic_json(args.output_dir/"controls-checkpoint.json",report)
        print("ARITHMETIC_CONTROL",entry,flush=True)
    for name in ("rf35mm_f1.8","ef35mm_f2.0","rf16mm_f2.8","rf24mm_f1.8"):
        try:
            p,sha=import_case(args.deeplens_root,name)
            b=prepare_wavefront_bundle(p,550.,1.,256,fixed_exit_plane(p))
            row={"name":name,"state":"wavefront_prepared_not_integrated","center_mm":b.center_mm,
                "legacy_fill_constraint_passed":b.metadata["geometric_pupil_fill"]>.2,**b.metadata}
        except WavefrontError as exc:row={"name":name,"state":exc.code,"error":str(exc)}
        report["edges"].append(row)
        atomic_json(args.output_dir/"controls-checkpoint.json",report)
        print("EDGE_CONTROL",row,flush=True)
    report.update(status="completed",completed_unix=time.time(),
        arithmetic_gpu_passed=all(c.get("passed") is True for c in report["controls"]))
    atomic_json(args.output_dir/"controls-terminal.json",report,immutable=True)
    return 0 if args.cpu_only or report["arithmetic_gpu_passed"] else 2


if __name__=="__main__":raise SystemExit(main())
