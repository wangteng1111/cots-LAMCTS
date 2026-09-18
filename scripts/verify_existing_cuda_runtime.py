#!/usr/bin/env python3
import json, subprocess, os
py="/var/lib/cots-lamcts/venv/bin/python"
checks=[
 [py,"-c","import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i),(torch.ones(1024,1024,device=f'cuda:{i}')@torch.ones(1024,1024,device=f'cuda:{i}')).sum().item()) for i in range(torch.cuda.device_count())]"],
 [py,"-c","import scipy,numpy; print('scipy',scipy.__version__,'numpy',numpy.__version__)"],
]
out=[]
for c in checks:
 p=subprocess.run(c,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
 out.append({"cmd":c,"rc":p.returncode,"out":p.stdout})
print(json.dumps(out,indent=2))
if any(x["rc"] for x in out): raise SystemExit(1)
