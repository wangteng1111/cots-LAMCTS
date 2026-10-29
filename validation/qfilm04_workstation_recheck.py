#!/usr/bin/env python3
"""Read-only workstation status/recovery plus an explicitly limited CPU/GPU tile test.

Copies only this project's known optical job reports; never reads credentials,
signals processes, changes services, or replays a claimed job.
The numerical tile is NOT a complete image-plane MTF or Final-J validation.
"""
from __future__ import annotations
import argparse,copy,json,os,socket,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from validation.qfilm04_fine_11_cpu_gpu import atomic_json,import_case,solve_case,DEEP_SHA
from evaluator.qfilm03_physical_huygens import Propagation,eval_fixed_film

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deeplens-root",type=Path,default=Path("/var/lib/cots-lamcts/deeplens_upstream"))
    a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    out=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts/qfilm04_recheck"));out.mkdir(parents=True,exist_ok=True)
    jobroot=Path(os.environ.get("COTS_JOB_DIR","/var/lib/cots-lamcts/jobs/local-probe")).parent
    report={"schema":"Qfilm04-workstation-recheck-v1","observed_unix":time.time(),
        "hostname":socket.gethostname(),"torch_version":str(torch.__version__),
        "cuda_available":torch.cuda.is_available(),"cpu_count":os.cpu_count(),
        "job_reports":[],"known_optical_processes":[],"bridge_processes":[],
        "physics_certified":False,"training_eligible":False}
    def save(): atomic_json(out/"workstation_recheck.json",report)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit(): continue
        try:
            cmd=(proc/"cmdline").read_bytes().decode(errors="replace")
            if "qfilm04_fine_11_cpu_gpu.py" in cmd:
                report["known_optical_processes"].append({"pid":int(proc.name),"argv":cmd.replace("\0"," ")[:1500]})
            if "cots_workstation_bridge" in cmd:
                report["bridge_processes"].append({"pid":int(proc.name)})
        except (OSError,PermissionError): pass
    if torch.cuda.is_available():
        report["gpu_devices"]=[{"index":i,"name":torch.cuda.get_device_name(i),
            "total_memory_bytes":torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    save()
    for qfile in sorted((ROOT/"remote/queue").glob("000-qfilm04-fine-*.json")):
        req=json.loads(qfile.read_text());jid=req["job_id"]
        backend=req["args"][req["args"].index("--backend")+1]
        job=jobroot/jid;file=job/"artifacts"/f"fine_11_{backend}_solve.json"
        row={"job_id":jid,"backend":backend,"local_checkpoint_exists":file.is_file()}
        if file.is_file():
            try:
                data=json.loads(file.read_text());target=out/"recovered"/(jid+".json")
                atomic_json(target,data)
                samples=[s for c in data["cases"] for f in c["fields"] for s in f["samples"]]
                row.update(checkpoint_mtime=file.stat().st_mtime,report_status=data.get("status"),
                    completed_psfs=sum(s.get("state")=="completed" for s in samples),
                    invalid_samples=sum(s.get("state")=="invalid" for s in samples),
                    active_samples=[{k:s.get(k) for k in ("wavelength_nm","pixels_completed","pixels_total","elapsed_seconds")} for s in samples if s.get("state")=="running"],
                    recovered_artifact="recovered/"+target.name)
            except Exception as exc: row["read_error"]=repr(exc)
        receipt=jobroot.parent/"state/receipts_v05"/(jid+".json")
        if receipt.is_file():
            try:
                rec=json.loads(receipt.read_text());row["local_receipt_state"]=rec.get("state")
                row["receipt_error"]=rec.get("error")
            except Exception as exc: row["receipt_read_error"]=repr(exc)
        report["job_reports"].append(row);save()
    actual=subprocess.check_output(["git","-C",str(a.deeplens_root),"rev-parse","HEAD"],text=True).strip()
    if actual!=DEEP_SHA: raise ValueError("DeepLens commit mismatch")
    p,sha=import_case(a.deeplens_root,"ef40mm_f2.8")
    setup=Propagation(pupil_n=256,pixel_n=16,pixel_pitch_um=.25,pixel_block=64,weight_mode="sqrt_exit_jac")
    probe={"lens":"ef40mm_f2.8","source_sha256":sha,"wavelength_nm":550.,"field_h":0.,
        "pupil_n":256,"pixel_n":16,"pixel_pitch_um":.25,"support_mm":.004,
        "scope":"identical ray/phase/integration arithmetic on a small detector tile only",
        "full_psf_mtf_validation":False,"devices":{}}
    results={}
    for backend,device in (("c","cpu"),("g","cuda:0")):
        if backend=="g" and not torch.cuda.is_available():
            probe["devices"][backend]={"status":"unavailable"};continue
        t=time.perf_counter()
        with torch.inference_mode(): r=eval_fixed_film(p,550.,0.,setup,device=device)
        if backend=="g":torch.cuda.synchronize()
        arr=r["psf"].detach().cpu().numpy();results[backend]=arr
        probe["devices"][backend]={"status":"completed","seconds":time.perf_counter()-t,
            "psf_sum":float(arr.sum()),"valid_rays":r["valid_rays"],"device":device}
    if "c" in results and "g" in results:
        delta=abs(results["c"]-results["g"])
        probe["max_abs_normalized_tile_psf_difference"]=float(delta.max())
        probe["arithmetic_parity_passed"]=bool(delta.max()<2e-6)
    report["arithmetic_tile_probe"]=probe;save()
    # Recover the failed RF24 report without repeating its completed fine PSFs.
    src=ROOT/"artifacts/qfilm04_rf24_gpu_checkpoint_20261029.json"
    if src.is_file() and torch.cuda.is_available():
        recovered=json.loads(src.read_text());case=recovered["cases"][0]
        samples=[s for f in case["fields"] for s in f["samples"]]
        if len(samples)!=21 or not all(s["state"] in ("completed","invalid") for s in samples):
            raise ValueError("RF24 recovery requires 21 already-attempted samples")
        rp,rsha=import_case(a.deeplens_root,"rf24mm_f1.8")
        if rsha!=case["source_sha256"]: raise ValueError("RF24 source checksum mismatch")
        dest=out/"rf24_gpu_report_recovered.json"
        solve_case(rp,case,SimpleNamespace(fields=3,block=64),"cuda:0",lambda:atomic_json(dest,recovered))
        recovered["status"]="finished";recovered["recovery_note"]="Reused seven completed mono PSFs and fourteen recorded invalid samples; recomputed baseline and report only."
        atomic_json(dest,recovered)
        report["rf24_report_recovery"]={"status":"completed","quality":case["quality"],
            "nonfinite_baseline_metric_fields":case.get("nonfinite_baseline_metric_fields",[]),
            "fine_psfs_recomputed":0};save()
    print("WORKSTATION_RECHECK",json.dumps({"cuda":report["cuda_available"],
        "live_optical_processes":len(report["known_optical_processes"]),
        "recovered_reports":sum(x["local_checkpoint_exists"] for x in report["job_reports"]),
        "tile_probe":probe},allow_nan=False),flush=True)
    if probe.get("arithmetic_parity_passed") is False: raise RuntimeError("CPU/GPU arithmetic tile parity failed")

if __name__=="__main__":main()
