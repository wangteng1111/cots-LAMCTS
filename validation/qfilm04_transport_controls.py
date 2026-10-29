#!/usr/bin/env python3
"""Limited lossless parallel-ray input-area control; not general certification."""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,numpy as np,torch
from evaluator.qfilm04_wavefront import prepare_wavefront_bundle,fixed_exit_plane
from evaluator.qfilm04_io import atomic_json


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(1)
    p={"surfaces":[{"radius":math.inf,"thickness":1.,"n_after":1.,"v_after":0.,"clear_aperture":30.},
        {"radius":math.inf,"thickness":19.,"n_after":1.,"v_after":0.,"clear_aperture":30.}],
        "stop_z_mm":0.,"image_z_mm":20.,"design_spec":{"stop_diameter_mm":2.,"efl_target_mm":100.}}
    report={"schema":"Qfilm04LosslessInputAreaControls-v1","general_transport_validated":False,
        "scope":"neutral planar surfaces; parallel rays; geometrical flux only; no diffraction closure", "cases":[]}
    for h in (0.,1.):
        alpha=math.atan(h*math.hypot(36,24)/200)
        exact=math.pi*math.cos(alpha)
        for n in (256,512,1024):
            b=prepare_wavefront_bundle(p,550.,h,n,fixed_exit_plane(p),transport_model="input_ray_tube_lossless")
            actual=b.metadata["transmitted_ray_tube_power"]
            report["cases"].append({"field_h":h,"pupil_n":n,"expected_power":exact,
                "active_quadrature_power":actual,"relative_error":abs(actual-exact)/exact,
                "orientation":b.metadata["exit_mapping_orientation"]})
    report["fine_grid_control_passed"]=all(c["relative_error"]<=.005 for c in report["cases"] if c["pupil_n"]==1024)
    atomic_json(args.output,report)
    if not report["fine_grid_control_passed"]:raise AssertionError(report)
    print(report)


if __name__=="__main__":main()
