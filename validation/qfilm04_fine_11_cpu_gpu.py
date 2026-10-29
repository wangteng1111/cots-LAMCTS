#!/usr/bin/env python3
"""Seven-color, fixed-film 0.25um Qfilm0.4 runs for the eleven camera files.

No pupil or support reduction is permitted in solve mode. Checkpoints contain
each completed mono complex OTF; an invalid field never supplies a final J.
Preflight traces and timing tiles are explicitly NOT completed PSF results.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from evaluator.qfilm02 import Config, WAVES, WEIGHTS, FREQS, evaluate as baseline, score_image
from evaluator.qfilm02 import _surfaces_at, pupil_grid, trace_pupil
from evaluator.qfilm03_physical_huygens import Propagation, eval_fixed_film, coherent_otf_axis
from validation.qfilm04_real_prescription_import import CAMERA_SAMPLES, import_geolens_lens

DEEP_SHA="911bb9d334fce7e20a4407c4506ac683b52ff246"

def encode(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)

def atomic_json(path,data):
    # Invalid legacy geometric metrics can contain NaN even when a field is
    # already rejected. Preserve that rejection and write explicit nulls,
    # rather than losing all completed fine-grid OTFs to a JSON exception.
    nonfinite=[]
    def clean(value,where="$"):
        if isinstance(value,np.ndarray): value=value.tolist()
        if isinstance(value,np.generic): value=value.item()
        if isinstance(value,float) and not math.isfinite(value):
            nonfinite.append(where);return None
        if isinstance(value,dict): return {k:clean(v,where+"."+str(k)) for k,v in value.items()}
        if isinstance(value,(tuple,list)): return [clean(v,where+f"[{i}]") for i,v in enumerate(value)]
        return value
    payload=clean(data)
    if nonfinite:
        payload["serialization_diagnostics"]={"nonfinite_values_written_as_null":nonfinite,
            "meaning":"missing or invalid numerical metrics; never an accepted finite result"}
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,default=encode,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,path)

def import_case(root,name):
    # The existing shim only disables optional plotting/NN imports, not optics.
    sys.path.insert(0,str(root))
    from validation.qfilm_deeplens_same_prescription import BK7
    from deeplens import GeoLens
    file=root/"datasets/lenses/camera"/(name+".json")
    raw=json.loads(file.read_text())
    lens=GeoLens(filename=str(file),device="cpu",dtype=torch.float64)
    return import_geolens_lens(lens,raw,name),hashlib.sha256(file.read_bytes()).hexdigest()

def preflight(p,h,w,device):
    uv,inside=pupil_grid(256,device)
    a=math.atan(h*math.hypot(36.,24.)/2/p["design_spec"]["efl_target_mm"])
    angle=torch.tensor([[a]],device=device,dtype=torch.float64)
    params=(_surfaces_at(p["surfaces"],w),angle)
    tr=trace_pupil(*params,uv,p["stop_z_mm"],p["design_spec"]["stop_diameter_mm"]/2,p["image_z_mm"])
    ch=trace_pupil(*params,torch.zeros((1,2),device=device,dtype=torch.float64),
        p["stop_z_mm"],p["design_spec"]["stop_diameter_mm"]/2,p["image_z_mm"])
    count=int((tr[6]&inside.unsqueeze(0)).sum())
    good=bool(ch[6].all()) and count>=32
    return {"trace_ok":good,"transmitted_pupil_rays":count,"chief_valid":bool(ch[6].all())}

def poly_otf(samples):
    ref=np.asarray(next(s["chief_xy_mm"] for s in samples if s["wavelength_nm"]==550.))
    out=np.zeros((len(FREQS),2),dtype=np.complex128)
    weights=np.asarray(WEIGHTS)/sum(WEIGHTS)
    for w,weight in zip(WAVES,weights):
        s=next(s for s in samples if s["wavelength_nm"]==w)
        otf=np.asarray(s["otf_real"])+1j*np.asarray(s["otf_imag"])
        shift=np.asarray(s["chief_xy_mm"])-ref
        out+=weight*otf*np.exp(-2j*np.pi*np.asarray(FREQS)[:,None]*shift[None,:])
    return np.clip(abs(out),0.,1.)

def solve_case(p,case,args,device,save):
    cfg=Config(fields=args.fields,pupil_n=256,fft_n=1024,max_fields_per_chunk=1,
        target_efl_mm=p["design_spec"]["efl_target_mm"])
    q=None
    try:
        with torch.inference_mode(): b=baseline(p,cfg,device=device)
        q=np.array(b["Q_image"],copy=True)
        case["baseline_quality_provisional"]=b["quality"]
        case["baseline_area_weights"]=b["area_weights"]
    except Exception as exc:
        case["baseline_error"]=repr(exc)
    setup=Propagation(pupil_n=256,pixel_n=2560,pixel_pitch_um=.25,
        pixel_block=args.block,weight_mode="sqrt_exit_jac")
    for fi,h in enumerate(np.linspace(0.,1.,args.fields)):
        field=next((f for f in case["fields"] if f["index"]==fi),None)
        if field is None:
            field={"index":fi,"field_h":float(h),"samples":[]}
            case["fields"].append(field)
        for w in WAVES:
            existing=next((s for s in field["samples"] if s["wavelength_nm"]==w),None)
            if existing is not None and existing.get("state") in ("completed","invalid"):
                continue
            if existing is not None: field["samples"].remove(existing)
            sample={"wavelength_nm":w,"state":"running"}
            field["samples"].append(sample)
            last=[time.monotonic()]
            t0=time.perf_counter()
            def progress(done,total):
                if time.monotonic()-last[0]>=30 or done==total:
                    last[0]=time.monotonic()
                    sample["pixels_completed"]=done
                    sample["pixels_total"]=total
                    sample["elapsed_seconds"]=time.perf_counter()-t0
                    save()
                    print("PSF_PROGRESS",case["name"],h,w,done,total,flush=True)
            save()
            try:
                with torch.inference_mode():
                    r=eval_fixed_film(p,w,float(h),setup,device=device,progress_callback=progress)
                if r["bad_jacobian_fraction"]>.02:
                    raise ValueError("exit-pupil Jacobian exceeds 0.02 threshold")
                arr=r["psf"].detach().cpu().numpy()
                # Border measured over the SAME physical 20um band as legacy 4px.
                k=80
                edge=float(arr[:k].sum()+arr[-k:].sum()+arr[k:-k,:k].sum()+arr[k:-k,-k:].sum())
                otf=coherent_otf_axis(r["psf"],.00025,FREQS)
                if not np.all(np.isfinite(otf)):
                    raise ValueError("nonfinite fine-grid OTF")
                sample.update(state="completed",otf_real=otf.real,otf_imag=otf.imag,
                    mtf=abs(otf),chief_xy_mm=r["center_film_mm"],
                    valid_rays=r["valid_rays"],bad_jacobian_fraction=r["bad_jacobian_fraction"],
                    exit_z_mm=r["exit_z_mm"],border_20um_flux_fraction=edge,
                    normalized_psf_sum=float(arr.sum()),
                    elapsed_seconds=time.perf_counter()-t0)
            except Exception as exc:
                sample.update(state="invalid",error=repr(exc),
                    elapsed_seconds=time.perf_counter()-t0,traceback=traceback.format_exc()[-1500:])
            save()
            print("FINE_SAMPLE",case["name"],h,w,sample["state"],sample.get("error",""),flush=True)
        good=len(field["samples"])==len(WAVES) and all(s["state"]=="completed" for s in field["samples"])
        field["complete"]=good
        if good:
            vals=poly_otf(field["samples"])
            field["polychromatic_mtf_ts"]=vals
            if q is not None: q[fi,10:20]=vals[:,0];q[fi,20:30]=vals[:,1]
        elif q is not None:
            q[fi,30]=0.
            q[fi,10:30]=0.  # Never expose legacy FFT MTF as a fine-grid result.
        save()
    case["propagation_complete"]=all(f.get("complete",False) for f in case["fields"])
    case["all_conditions_attempted"]=True
    if q is not None:
        bad_metrics=np.where(~np.isfinite(q).all(axis=1))[0]
        if len(bad_metrics):
            q[bad_metrics,30]=0.
            case["nonfinite_baseline_metric_fields"]=bad_metrics.tolist()
        case["Q_image"]=q
        case["quality"]=score_image(q,b["area_weights"])
    else:
        case["quality"]={"valid":False,"J_task":None,"reason":"baseline failed"}
    case["state"]="completed" if case["propagation_complete"] else "completed_with_invalid_samples"
    completed=[s for f in case["fields"] for s in f["samples"] if s["state"]=="completed"]
    case["sampling_assessment"]={"sampling_converged":False,
        "max_border_20um_flux_fraction":max((s["border_20um_flux_fraction"] for s in completed),default=None),
        "boundary_energy_flag_threshold":.01,
        "boundary_energy_flagged_samples":sum(s["border_20um_flux_fraction"]>.01 for s in completed),
        "interpretation":"boundary energy requires independent support and pupil convergence; program validity is not physical certification"}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deeplens-root",type=Path,default=Path("/var/lib/cots-lamcts/deeplens_upstream"))
    ap.add_argument("--backend",choices=("c","g"),required=True)
    ap.add_argument("--mode",choices=("solve","preflight","benchmark"),default="solve")
    ap.add_argument("--names",nargs="+",choices=CAMERA_SAMPLES,default=list(CAMERA_SAMPLES))
    ap.add_argument("--fields",type=int,default=3)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--block",type=int,default=64)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    if args.fields<2: ap.error("at least two full-frame radial fields required")
    if args.threads<1 or args.block<1: ap.error("threads and block must be positive")
    args.output=args.output or Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/f"fine_11_{args.backend}_{args.mode}.json"
    torch.set_num_threads(args.threads);torch.set_default_dtype(torch.float64)
    device="cpu" if args.backend=="c" else "cuda:0"
    protocol={"source_commit":DEEP_SHA,"names":args.names,"fields_h":np.linspace(0,1,args.fields).tolist(),
        "wavelengths_nm":WAVES,"spectral_weights":WEIGHTS,"frequencies_lpmm":FREQS,
        "frame_mm":[36.,24.],"pupil_n":256,"detector_n":2560,"detector_pitch_um":.25,
        "support_mm":.64,"weight_mode":"sqrt_exit_jac"}
    identity=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    report={"schema":"Qfilm04-fine-eleven-camera-v1","protocol":protocol,"protocol_hash":identity,
        "variant":"Qfilm0.4-"+args.backend,"mode":args.mode,"device":device,
        "torch_version":str(torch.__version__),"threads":args.threads,"block":args.block,
        "status":"running","training_eligible":False,"physics_certified":False,
        "sampling_converged":False,"manufactured_lens_empirical_validation":False,
        "cases":[]}
    if args.output.exists():
        old=json.loads(args.output.read_text())
        if old["protocol_hash"]!=identity or old["mode"]!=args.mode or old["variant"]!=report["variant"]:
            raise ValueError("checkpoint protocol or backend mismatch")
        report=old;report["status"]="running"
        report.update(threads=args.threads,block=args.block)
    def save():
        report["updated_unix"]=time.time();atomic_json(args.output,report)
    if args.backend=="g" and not torch.cuda.is_available():
        report.update(status="unavailable",error="CUDA unavailable; no CPU fallback")
        save();print(json.dumps({"status":report["status"],"error":report["error"]}));return 2
    actual=subprocess.check_output(["git","-C",str(args.deeplens_root),"rev-parse","HEAD"],text=True).strip()
    if actual!=DEEP_SHA: raise ValueError("DeepLens source commit mismatch")
    save()
    for name in args.names:
        case=next((c for c in report["cases"] if c["name"]==name),None)
        if case is not None and case.get("all_conditions_attempted"): continue
        if case is None:
            case={"name":name,"fields":[],"state":"running"};report["cases"].append(case)
        try:
            p,sha=import_case(args.deeplens_root,name)
            if case.get("source_sha256",sha)!=sha: raise ValueError("source bytes changed")
            case.update(source_sha256=sha,provenance=p["source_provenance"])
            if args.mode=="solve":
                solve_case(p,case,args,device,save)
            elif args.mode=="preflight":
                case["fields"]=[]
                for h in np.linspace(0.,1.,args.fields):
                    rows=[]
                    for w in WAVES:
                        try:
                            with torch.inference_mode(): row=preflight(p,float(h),w,device)
                            rows.append({"wavelength_nm":w,**row})
                        except Exception as exc: rows.append({"wavelength_nm":w,"trace_ok":False,"error":repr(exc)})
                    case["fields"].append({"field_h":float(h),"samples":rows});save()
                case["state"]="trace_only_complete";case["all_conditions_attempted"]=True
            else:
                # This small tile has NO window convergence or full-J meaning.
                with torch.inference_mode():
                    t0=time.perf_counter()
                    split=[]
                    def tick(done,total):
                        if args.backend=="g": torch.cuda.synchronize()
                        split.append((done,time.perf_counter()))
                    r=eval_fixed_film(p,550.,0.,Propagation(pupil_n=256,pixel_n=16,
                        pixel_pitch_um=.25,pixel_block=args.block,weight_mode="sqrt_exit_jac"),
                        device=device,progress_callback=tick)
                    if args.backend=="g": torch.cuda.synchronize()
                    seconds=time.perf_counter()-t0
                propagation=split[-1][1]-split[0][1]
                case.update(state="timing_tile_only",tile_pixels=256,tile_seconds=seconds,
                    tile_propagation_seconds=propagation,
                    linear_full_psf_seconds_estimate=propagation*(2560/16)**2,
                    estimate_includes_trace_overhead=False,valid_rays=r["valid_rays"],
                    all_conditions_attempted=True)
        except Exception as exc:
            case.update(state="failed",error=repr(exc),traceback=traceback.format_exc()[-2000:])
        save();print("FINE_CASE",name,case["state"],flush=True)
    report["status"]="finished" if all(c.get("all_conditions_attempted") for c in report["cases"]) else "incomplete"
    report["summary"]={"lenses":len(report["cases"]),
        "completed_psfs":sum(s.get("state")=="completed" for c in report["cases"] for f in c["fields"] for s in f["samples"]),
        "valid_final_scores":sum(c.get("quality",{}).get("valid",False) for c in report["cases"])}
    save();print("FINE_COMPLETE",json.dumps(report["summary"]),flush=True)
    return 0 if report["status"]=="finished" else 1

if __name__=="__main__": raise SystemExit(main())
