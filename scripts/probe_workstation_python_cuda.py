#!/usr/bin/env python3
"""Probe common workstation Python runtimes for torch/CUDA without requiring torch in bridge Python."""
import json,os,glob,subprocess,sys
from pathlib import Path
cands=[sys.executable,'/usr/bin/python3','/usr/local/bin/python3']
patterns=['/opt/*/bin/python*','/opt/*/*/bin/python*','/home/*/.venv/bin/python*','/home/*/venv/bin/python*','/home/*/miniconda3/bin/python*','/home/*/miniconda3/envs/*/bin/python*','/home/*/anaconda3/bin/python*','/home/*/anaconda3/envs/*/bin/python*']
for p in patterns:cands.extend(glob.glob(p))
seen=[];rows=[]
code="import json,sys; r={'python':sys.executable,'version':sys.version.split()[0]};\ntry:\n import torch; r.update(torch=__import__('torch').__version__,cuda=bool(torch.cuda.is_available()),devices=torch.cuda.device_count(),names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\nexcept Exception as e:r.update(torch_error=repr(e));\nprint(json.dumps(r))"
for p in cands:
 p=os.path.realpath(p)
 if p in seen or not os.path.isfile(p) or not os.access(p,os.X_OK):continue
 seen.append(p)
 try:
  q=subprocess.run([p,'-c',code],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
  rows.append({'candidate':p,'rc':q.returncode,'output':q.stdout[-4000:]})
 except Exception as e:rows.append({'candidate':p,'error':repr(e)})
out={'bridge_python':sys.executable,'candidates':rows}
print(json.dumps(out,indent=2),flush=True)
ad=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','.'));ad.mkdir(parents=True,exist_ok=True);(ad/'python_cuda_probe.json').write_text(json.dumps(out,indent=2))
