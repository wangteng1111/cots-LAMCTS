#!/usr/bin/env python3
"""Audit the UNWRAPPED phase of the actual Qfilm physical Huygens integrand.

This traces real pinned prescriptions and tests adjacent active pupil nodes
at selected detector points. It does not compute a full PSF or certify MTF.
Half-cycle steps are a sampling diagnostic, not a rigorous quadrature bound.
"""
from __future__ import annotations
import argparse,hashlib,json,math,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from evaluator.qfilm02 import _surfaces_at,pupil_grid,trace_pupil
from evaluator.qfilm02_huygens import exit_plane_from_rays,exit_area_weights
from validation.qfilm04_fine_11_cpu_gpu import import_case,atomic_json,DEEP_SHA

def audit(p,wavelength_nm,n,offsets_um):
    uv,inside=pupil_grid(n,"cpu");angles=torch.zeros((1,1),dtype=torch.float64)
    stop=p["stop_z_mm"];film=p["image_z_mm"];rs=p["design_spec"]["stop_diameter_mm"]/2
    surfs=_surfaces_at(p["surfaces"],wavelength_nm)
    tr=trace_pupil(surfs,angles,uv,stop,rs,film)
    chief=trace_pupil(surfs,angles,torch.zeros((1,2),dtype=torch.float64),stop,rs,film)
    if not bool(chief[6].all()):raise ValueError("invalid chief")
    mask=(tr[6]&inside[None,:]).reshape(n,n)
    exit_z=float(tr[0].reshape(n,n)[mask].max())+.1
    if exit_z>=film:raise ValueError("exit plane beyond film")
    args=[x.reshape(n,n) for x in (*tr[:6],tr[6]&inside[None,:])]
    xe,ye,opl,ok=exit_plane_from_rays(*args,film,exit_z)
    area,bad,jac=exit_area_weights(xe,ye,ok,2*rs/n)
    active=(area>0)&torch.isfinite(opl)
    cx,cy=float(chief[4].item()),float(chief[5].item())
    ref=opl[active][0]+film-exit_z
    rows=[]
    for ox,oy in offsets_um:
        distance=torch.sqrt((film-exit_z)**2+(cx+ox*.001-xe)**2+(cy+oy*.001-ye)**2)
        # Crucial: inspect before the remainder(..., 1) in the real kernel.
        cycles=(opl+distance-ref)/(wavelength_nm*1e-6)
        differences=[];weights=[]
        for dim in (0,1):
            if dim==0:
                valid=active[1:,:]&active[:-1,:]
                a=(area[1:,:]+area[:-1,:])*.5
            else:
                valid=active[:,1:]&active[:,:-1]
                a=(area[:,1:]+area[:,:-1])*.5
            differences.append(torch.diff(cycles,dim=dim).abs()[valid])
            weights.append(a[valid])
        d=torch.cat(differences);weight=torch.cat(weights)
        rows.append({"detector_offset_um":[ox,oy],"adjacent_pairs":len(d),
            "max_abs_step_cycles":float(d.max()),"p95_abs_step_cycles":float(torch.quantile(d,.95)),
            "p99_abs_step_cycles":float(torch.quantile(d,.99)),
            "fraction_over_half_cycle":float((d>.5).double().mean()),
            "quadrature_weight_fraction_over_half_cycle":float(weight[d>.5].sum()/weight.sum()),
            "fraction_over_quarter_cycle":float((d>.25).double().mean())})
    return {"wavelength_nm":wavelength_nm,"pupil_n":n,"active_nodes":int(active.sum()),
        "exit_z_mm":exit_z,"chief_xy_mm":[cx,cy],"points":rows}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deeplens-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--pupils",type=int,nargs="+",default=[256])
    args=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    actual=subprocess.check_output(["git","-C",str(args.deeplens_root),"rev-parse","HEAD"],text=True).strip()
    if actual!=DEEP_SHA:raise ValueError("DeepLens source commit mismatch")
    report={"schema":"Qfilm04-actual-integrand-phase-sampling-v1","source_commit":DEEP_SHA,
        "torch_version":str(torch.__version__),"device":"cpu","threads":1,
        "physical_kernel_sha256":hashlib.sha256((ROOT/"evaluator/qfilm03_physical_huygens.py").read_bytes()).hexdigest(),
        "full_psf_computed":False,"physics_certified":False,
        "interpretation":"adjacent unwrapped phase increments diagnose sampled oscillatory quadrature; they are not a rigorous error bound",
        "detector_window_half_width_um":320,"cases":[]}
    for name,wave in (("ef40mm_f2.8",430.),("ef40mm_f2.8",550.),("rf50mm_f1.8",670.)):
        p,sha=import_case(args.deeplens_root,name)
        case={"name":name,"source_sha256":sha,"wavelength_nm":wave,"grids":[]};report["cases"].append(case)
        for n in args.pupils:
            t=time.perf_counter()
            with torch.inference_mode():
                row=audit(p,wave,n,[(0.,0.),(310.,0.),(0.,310.),(310.,310.)])
            row["seconds"]=time.perf_counter()-t;case["grids"].append(row)
            atomic_json(args.output,report)
            print("PHASE_AUDIT",name,wave,n,json.dumps(row),flush=True)

if __name__=="__main__":main()
