#!/usr/bin/env python3
"""Read-only fresh challenge/response and Qfilm workstation tests.

Does not restart services, signal jobs, read credentials or alter receipts.
The first artifact is written before importing the scientific runtime.
"""
from __future__ import annotations
import argparse,hashlib,json,os,platform,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as stream:
        json.dump(data,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def command(argv,timeout=20):
    try:
        result=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              timeout=timeout,check=False,cwd=ROOT)
        return {"returncode":result.returncode,"output":result.stdout[-12000:]}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {"error":type(exc).__name__}


def state_snapshot(bridge_root):
    state=bridge_root/"state";pending=state/"pending_publish"
    files=sorted(pending.glob("*.json")) if pending.is_dir() else []
    counts={};times=[]
    # Metadata only; deliberately never expose stored API payloads or errors.
    for path in files[:4096]:
        try:
            item=json.loads(path.read_text());target=item.get("path","")
            category="heartbeat" if target=="remote/agent_status.json" else target.split("/")[1] if target.startswith("remote/") else "other"
            counts[category]=counts.get(category,0)+1
            if isinstance(item.get("queued_at"),(float,int)):times.append(item["queued_at"])
        except (OSError,ValueError):pass
    jobs=[]
    for request in sorted((ROOT/"remote/queue").glob("*qfilm04*.json"))[:128]:
        jid=request.stem;job=bridge_root/"jobs"/jid
        row={"job_id":jid,"job_directory_exists":job.is_dir()}
        receipt=state/"receipts_v05"/(jid+".json")
        if receipt.is_file():
            try:
                data=json.loads(receipt.read_text())
                row["receipt"]={k:data[k] for k in ("state","code_commit","claimed_at","finished_at","returncode") if k in data}
            except (OSError,ValueError):row["receipt_unreadable"]=True
        artifacts=job/"artifacts"
        row["checkpoint_files"]=[{"name":str(p.relative_to(artifacts)),"bytes":p.stat().st_size,
            "mtime":p.stat().st_mtime} for p in sorted(artifacts.rglob("*.json"))[:100] if p.is_file()] if artifacts.is_dir() else []
        jobs.append(row)
    processes=[]
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():continue
        try:
            argv=(path/"cmdline").read_bytes().decode(errors="replace").split("\0")
            scripts=[Path(a).name for a in argv if a.endswith(".py") and
                     ("qfilm" in Path(a).name or "cots_workstation_bridge" in Path(a).name)]
            if not scripts:continue
            values={}
            for line in (path/"status").read_text().splitlines():
                k,_,v=line.partition(":")
                if k in ("State","PPid","Threads","VmRSS"):values[k]=v.strip()
            processes.append({"pid":int(path.name),"scripts":scripts,"status":values})
        except (OSError,PermissionError):pass
    return {"pending_publications":len(files),"pending_categories":counts,
        "pending_metadata_truncated":len(files)>4096,"oldest_queued_unix":min(times) if times else None,
        "newest_queued_unix":max(times) if times else None,"qfilm_jobs":jobs,"processes":processes}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nonce",required=True)
    ap.add_argument("--output-dir",type=Path,default=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts/qfilm04_reconnect")))
    ap.add_argument("--snapshot-only",action="store_true")
    args=ap.parse_args()
    if not args.nonce.isalnum() or not 16<=len(args.nonce)<=80:raise ValueError("invalid challenge nonce")
    out=args.output_dir
    report={"schema":"Qfilm04FreshReconnect-v1","nonce":args.nonce,"started_unix":time.time(),
        "hostname":socket.gethostname(),"pid":os.getpid(),"python":sys.executable,
        "cpu_count":os.cpu_count(),"job_id":os.environ.get("COTS_JOB_ID"),
        "code_commit_env":os.environ.get("COTS_GIT_COMMIT"),"state":"fresh_challenge_received",
        "training_eligible":False,"physics_certified":False}
    write(out/"fresh-handshake.json",report)
    print("FRESH_HANDSHAKE",json.dumps(report),flush=True)
    report["git_head"]=command(["git","rev-parse","HEAD"])
    report["code_files_sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/"evaluator").glob("qfilm04*.py"))}
    report["systemd"]=command(["systemctl","show","cots-lamcts-bridge.service","--no-pager",
        "--property=ActiveState,SubState,MainPID,NRestarts,ExecMainStartTimestamp"])
    report["nvidia_smi"]=command(["nvidia-smi","--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                                 "--format=csv,noheader,nounits"])
    report["workstation_state"]=state_snapshot(Path(os.environ.get("COTS_BRIDGE_ROOT","/var/lib/cots-lamcts")))
    write(out/"reconnect-checkpoint.json",report)
    if not args.snapshot_only:
        report["state"]="testing"
        write(out/"reconnect-checkpoint.json",report)
        report["qfilm_unit_tests"]=command([sys.executable,"validation/qfilm04_remediation_tests.py"],timeout=180)
        write(out/"reconnect-checkpoint.json",report)
        if report["qfilm_unit_tests"].get("returncode")==0:
            # This requires actual CUDA; no --cpu-only substitution is made.
            report["qfilm_workstation_controls"]=command([sys.executable,
                "validation/qfilm04_remediation_workstation_controls.py","--output-dir",str(out/"controls")],timeout=1200)
    report["finished_unix"]=time.time()
    tests=report.get("qfilm_unit_tests",{}).get("returncode")==0
    controls=report.get("qfilm_workstation_controls",{}).get("returncode")==0
    report["state"]="snapshot_complete" if args.snapshot_only else "tests_passed" if tests and controls else "tests_incomplete_or_failed"
    write(out/"reconnect-terminal.json",report)
    print("FRESH_RECONNECT_TERMINAL",json.dumps({k:report[k] for k in ("nonce","hostname","state","finished_unix")}),flush=True)
    return 0 if args.snapshot_only or tests and controls else 2


if __name__=="__main__":raise SystemExit(main())
