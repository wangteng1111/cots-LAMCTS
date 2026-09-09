#!/usr/bin/env python3
import json, os, platform, shutil, subprocess, sys
from pathlib import Path

def cmd(args):
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20, check=False)
        return {"returncode": p.returncode, "stdout": p.stdout[-20000:]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

out = {
    "schema": 1,
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python": sys.executable,
    "python_version": sys.version,
    "cpu_count": os.cpu_count(),
    "cwd": os.getcwd(),
    "git_commit_env": os.environ.get("COTS_GIT_COMMIT"),
    "job_id": os.environ.get("COTS_JOB_ID"),
    "disk": shutil.disk_usage(os.getcwd())._asdict(),
    "commands": {
        "git_head": cmd(["git", "rev-parse", "HEAD"]),
        "git_status": cmd(["git", "status", "--short"]),
        "lscpu": cmd(["lscpu"]),
        "nvidia_smi": cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]),
    },
}
for mod in ["numpy", "scipy", "torch", "numba"]:
    try:
        m = __import__(mod)
        out[f"{mod}_version"] = getattr(m, "__version__", "unknown")
    except Exception as e:
        out[f"{mod}_error"] = f"{type(e).__name__}: {e}"

try:
    import torch
    out["torch_cuda_available"] = torch.cuda.is_available()
    out["torch_cuda_version"] = torch.version.cuda
    out["torch_device_count"] = torch.cuda.device_count()
    out["torch_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception as e:
    out["torch_cuda_error"] = f"{type(e).__name__}: {e}"

artifact_dir = Path(os.environ["COTS_JOB_ARTIFACT_DIR"])
artifact_dir.mkdir(parents=True, exist_ok=True)
(artifact_dir / "workstation_probe.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(out, indent=2, sort_keys=True))
