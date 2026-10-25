#!/usr/bin/env python3
"""Workstation CPU/CUDA Qfilm0.2 validation with machine-readable artifacts."""
from __future__ import annotations
import json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR",str(ROOT/"artifacts"/"qfilm02_server")))
OUT.mkdir(parents=True,exist_ok=True)
SMOKE=ROOT/"validation"/"qfilm02_smoke.py"
DEVICE_NAMES=("cpu","cuda:0","cuda:1")

def run_device(device):
    tag=device.replace(":","_")
    target=OUT/f"qfilm02_{tag}_full.json"
    command=[sys.executable,"-u",str(SMOKE),"--device",device,"--full","--out",str(target)]
    env=dict(os.environ)
    env["PYTHONPATH"]=str(ROOT)+os.pathsep+env.get("PYTHONPATH","")
    env["OMP_NUM_THREADS"]="1"
    env["MKL_NUM_THREADS"]="1"
    start=time.perf_counter()
    cp=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=900)
    elapsed=time.perf_counter()-start
    (OUT/f"qfilm02_{tag}.log").write_text(cp.stdout)
    result={"device":device,"elapsed_sec":elapsed,"returncode":cp.returncode}
    if cp.returncode==0 and target.is_file():
        result["report"]=json.loads(target.read_text())
    else:
        result["error_tail"]=cp.stdout[-6000:]
    return result

def main():
    summary={"schema":"Qfilm0.2-server-validation-v1","repo_commit":os.environ.get("COTS_GIT_COMMIT"),"devices":[]}
    for dev in DEVICE_NAMES:
        print(f"QFILM_BEGIN {dev}",flush=True)
        try:r=run_device(dev)
        except Exception as e:r={"device":dev,"error":repr(e),"traceback":traceback.format_exc()}
        summary["devices"].append(r)
        print(f"QFILM_DONE {dev} {json.dumps({k:v for k,v in r.items() if k!='report'})}",flush=True)
        (OUT/"qfilm02_server_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if all("report" in x for x in summary["devices"]):
        cpu=summary["devices"][0]["report"]["real_prescription_smoke"]
        diffs={}
        for r in summary["devices"][1:]:
            gpu=r["report"]["real_prescription_smoke"]
            diffs[r["device"]]={
                "J_task_abs":abs(cpu["J_task"]-gpu["J_task"]),
                "central_mtf_max_abs":float(np.max(np.abs(np.asarray(cpu["central_mtf_t"])-np.asarray(gpu["central_mtf_t"])))),
                "edge_mtf_max_abs":float(np.max(np.abs(np.asarray(cpu["edge_mtf_t"])-np.asarray(gpu["edge_mtf_t"])))),
                "gpu_faster_than_cpu":cpu["seconds"]/gpu["seconds"] if gpu["seconds"]>0 else None,
            }
        summary["comparisons"]=diffs
        summary["passed"]=all(v["J_task_abs"]<0.02 and v["central_mtf_max_abs"]<0.02 and v["edge_mtf_max_abs"]<0.02 for v in diffs.values())
    else:summary["passed"]=False
    (OUT/"qfilm02_server_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("QFILM_SUMMARY "+json.dumps({"passed":summary["passed"],"comparisons":summary.get("comparisons"),"devices":[{"device":x["device"],"returncode":x.get("returncode"),"error":x.get("error")} for x in summary["devices"]]}),flush=True)
    return 0 if summary["passed"] else 2

if __name__=="__main__":sys.exit(main())
