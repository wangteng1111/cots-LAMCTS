#!/usr/bin/env python3
"""Deep workstation runtime inventory for recovering the previously working CUDA/PyTorch environment."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

roots = [Path("/opt"), Path("/usr/local"), Path("/home"), Path("/root"), Path("/var/lib"), Path("/mnt"), Path("/workspace")]
cands=set()
for root in roots:
    if not root.exists(): continue
    try:
        for p in root.rglob("python"):
            s=str(p)
            if any(k in s.lower() for k in ("/bin/python","venv","conda","miniconda","anaconda",".venv","env")) and p.is_file():
                cands.add(s)
            if len(cands)>300: break
    except Exception: pass
for p in ["/usr/bin/python3","/usr/local/bin/python3","/opt/venv/bin/python","/opt/cots-lamcts/.venv/bin/python"]:
    if Path(p).exists(): cands.add(p)

def run(cmd, timeout=20):
    try:
        x=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
        return {"rc":x.returncode,"out":x.stdout[-8000:]}
    except Exception as e:return {"rc":-1,"out":repr(e)}

rows=[]
probe='import json,sys; d={"python":sys.executable,"version":sys.version};\ntry:\n import torch; d.update(torch= torch.__version__, cuda=torch.cuda.is_available(), cuda_version=torch.version.cuda, devices=torch.cuda.device_count(), names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\nexcept Exception as e:d["torch_error"]=repr(e)\nprint(json.dumps(d))'
for p in sorted(cands):
    r=run([p,"-c",probe])
    rows.append({"candidate":p,**r})

extra={}
for name,cmd in {
 "which_python":["bash","-lc","which -a python python3 || true"],
 "conda":["bash","-lc","find /opt /home /root /usr/local /var/lib -maxdepth 5 -type f -path '*/bin/conda' 2>/dev/null | head -100"],
 "torch_dirs":["bash","-lc","find /opt /home /root /usr/local /var/lib /mnt -maxdepth 8 -type d -name 'torch' 2>/dev/null | head -100"],
 "systemd":["bash","-lc","systemctl cat cots-workstation-bridge.service 2>&1 || systemctl cat cots-lamcts-bridge.service 2>&1 || true"],
 "process_history":["bash","-lc","ps auxww | grep -E 'python|conda|venv' | grep -v grep | head -100"],
}.items(): extra[name]=run(cmd,60)

out={"bridge_python":sys.executable,"env_bridge_python":os.environ.get("COTS_BRIDGE_PYTHON"),"candidates":rows,"extra":extra}
adir=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","."))
adir.mkdir(parents=True,exist_ok=True)
(adir/"runtime_inventory.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
