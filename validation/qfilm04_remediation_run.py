#!/usr/bin/env python3
"""Pinned real-prescription audit/solve with one frozen plan for CPU and GPU.

Audit does not integrate PSFs. Solve records incomplete convergence explicitly.
Large .npy controls remain on the workstation with a text manifest + SHA256.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from evaluator.qfilm02 import FREQS
from evaluator.qfilm04_io import atomic_json
from evaluator.qfilm04_sampling import ConvergencePolicy,make_plan,solve_condition
from validation.qfilm04_fine_11_cpu_gpu import import_case,DEEP_SHA

CASES={"ef40-blue":("ef40mm_f2.8",430.),"ef40-green":("ef40mm_f2.8",550.),
       "rf50-red":("rf50mm_f1.8",670.)}


def compare_executions(a,b):
    def key(r):return r["pupil_n"],r["support_index"],r["pitch_um"]
    ra={key(r):r for r in a["runs"]};rb={key(r):r for r in b["runs"]}
    matched=[]
    for k in sorted(set(ra)&set(rb)):
        x,y=ra[k],rb[k]
        matched.append({"grid":k,"complex_otf_max_abs":float(np.max(abs(x["otf"]-y["otf"]))),
            "integral_relative_difference":abs(x["raw_intensity_integral"]-y["raw_intensity_integral"])/max(abs(y["raw_intensity_integral"]),1e-300)})
    complete=(a["plan_hash"]==b["plan_hash"] and a["actual_device"]!=b["actual_device"] and
        a["execution_id"]!=b["execution_id"] and a["state"]==b["state"]=="numerically_converged"
        and set(ra)==set(rb) and len(matched)>0)
    return {"matched_runs":matched,"full_condition_pair_verified":bool(complete and
        all(v["complex_otf_max_abs"]<=1e-6 and v["integral_relative_difference"]<=1e-6 for v in matched)),
        "both_conditions_numerically_converged":complete,
        "interpretation":"matched partial runs alone do not verify the complete condition"}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deeplens-root",type=Path,default=Path("/var/lib/cots-lamcts/deeplens_upstream"))
    ap.add_argument("--output-dir",type=Path,default=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts/qfilm04_remediation")))
    ap.add_argument("--case",choices=list(CASES)+["all"],default="all")
    ap.add_argument("--mode",choices=("audit","solve"),default="audit")
    ap.add_argument("--pupils",type=int,nargs="+",default=[256,512,1024,2048,4096])
    ap.add_argument("--supports-mm",type=float,nargs="+",default=[.64,.96,1.28])
    ap.add_argument("--backends",choices=("c","g"),nargs="+",default=["g","c"])
    ap.add_argument("--threads",type=int,default=4)
    ap.add_argument("--block",type=int,default=32)
    ap.add_argument("--max-pair-evaluations",type=int,default=150_000_000_000_000)
    ap.add_argument("--save-full-psf",action="store_true")
    args=ap.parse_args();torch.set_num_threads(args.threads)
    actual=subprocess.check_output(["git","-C",str(args.deeplens_root),"rev-parse","HEAD"],text=True).strip()
    if actual!=DEEP_SHA:raise ValueError("DeepLens source commit mismatch")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.mode=="solve":
        # Each independently claimed Bridge job checks the actual pinned code.
        # A separate job's mutable status is never treated as a passed gate.
        subprocess.run([sys.executable,str(ROOT/"validation/qfilm04_remediation_tests.py")],check=True)
    policy=ConvergencePolicy(pupil_candidates=tuple(args.pupils),support_candidates_mm=tuple(args.supports_mm),
        max_pair_evaluations=args.max_pair_evaluations)
    policy.validate()
    report={"schema":"Qfilm04RemediationValidation-v1","mode":args.mode,"source_commit":DEEP_SHA,
        "training_eligible":False,"physics_certified":False,"cases":[],"status":"running"}
    try:report["code_commit"]=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    except Exception:report["code_commit"]=None
    report["code_files_sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/"evaluator").glob("qfilm04*.py"))}
    sequence=0;last=[0.]
    def status(message):
        nonlocal sequence
        if time.monotonic()-last[0]>=30 or isinstance(message,dict):
            last[0]=time.monotonic();sequence+=1
            atomic_json(args.output_dir/"progress.json",{"sequence":sequence,"updated_unix":time.time(),
                "state":"running","message":message})
            print("REMEDIATION_PROGRESS",sequence,json.dumps(message),flush=True)
    selected=list(CASES) if args.case=="all" else [args.case]
    for identifier in selected:
        name,wave=CASES[identifier];case={"id":identifier,"name":name,"wavelength_nm":wave,"field_h":0.}
        report["cases"].append(case);start=time.perf_counter()
        try:
            p,sha=import_case(args.deeplens_root,name);case["source_sha256"]=sha
            plan=make_plan(p,wave,0.,FREQS,policy,progress_callback=status)
            atomic_json(args.output_dir/(identifier+"-plan.json"),plan,immutable=True)
            case["phase_plan_hash"]=plan["plan_hash"];case["plan_file"]=identifier+"-plan.json"
            if args.mode=="audit":
                case["state"]="phase_audit_complete_no_psf_computed"
            else:
                results={};case["executions"]={}
                for backend in args.backends:
                    if backend=="g" and not torch.cuda.is_available():
                        case["executions"][backend]={"state":"CUDA_UNAVAILABLE","fallback_used":False};continue
                    prefix=identifier+"-"+backend
                    def checkpoint(value):atomic_json(args.output_dir/(prefix+"-checkpoint.json"),value)
                    counter=[0]
                    def save_raw(image,metadata):
                        counter[0]+=1;filename=prefix+f"-raw-{counter[0]:03d}.npy"
                        path=args.output_dir/filename;np.save(path,image,allow_pickle=False)
                        h=hashlib.sha256()
                        with path.open("rb") as stream:
                            for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
                        return {"file":filename,"sha256":h.hexdigest(),"bytes":path.stat().st_size,
                            "shape":list(image.shape),"dtype":str(image.dtype),**metadata}
                    result=solve_condition(p,plan,policy,"cpu" if backend=="c" else "cuda:0",args.block,
                        checkpoint,lambda done,total:status({"case":identifier,"backend":backend,
                            "pixels_done":done,"pixels_total":total}) if done==total else
                            status([identifier,backend,done,total]),save_raw if args.save_full_psf else None)
                    plan=result["resolved_sampling_plan"]
                    case["plan_hash"]=plan["plan_hash"]
                    # The second backend replays the first backend's actual
                    # integration sequence, not a separate adaptive search.
                    atomic_json(args.output_dir/(identifier+"-executed-plan.json"),plan)
                    atomic_json(args.output_dir/(prefix+"-terminal.json"),result,immutable=True)
                    results[backend]=result
                    case["executions"][backend]={"state":result["state"],"flags":result["flags"],
                        "execution_id":result["execution_id"],"actual_device":result["actual_device"],
                        "pair_evaluations":result["pair_evaluations"],"terminal_file":prefix+"-terminal.json"}
                if "c" in results and "g" in results:case["cpu_gpu_comparison"]=compare_executions(results["c"],results["g"])
                case["state"]="attempted"
        except Exception as exc:
            case.update(state="error",error=repr(exc),traceback=traceback.format_exc())
        case["elapsed_seconds"]=time.perf_counter()-start
        atomic_json(args.output_dir/"summary-checkpoint.json",report)
    report["status"]="completed_with_errors" if any(c["state"]=="error" for c in report["cases"]) else "completed"
    report["finished_unix"]=time.time()
    atomic_json(args.output_dir/"summary-terminal.json",report,immutable=True)
    print("REMEDIATION_TERMINAL",json.dumps(report),flush=True)
    return 2 if report["status"]=="completed_with_errors" else 0


if __name__=="__main__":raise SystemExit(main())
