#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path('/var/lib/cots-lamcts')
VENV = ROOT / 'venv'
ART = Path(os.environ.get('COTS_JOB_ARTIFACT_DIR', '.'))
ART.mkdir(parents=True, exist_ok=True)

PACKAGES = [
    'pip>=24.0',
    'setuptools>=70',
    'wheel>=0.43',
    'numpy>=1.26,<2.3',
    'scipy>=1.11,<1.16',
    'numba>=0.61,<0.63',
    'scikit-learn>=1.4,<1.8',
    'torch',
]

def run(cmd, timeout=3600, check=True):
    print('+', ' '.join(map(str, cmd)), flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout, check=False)
    print(p.stdout, flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f'command failed {p.returncode}: {cmd}')
    return p

summary = {
    'schema': 1,
    'started_at': time.time(),
    'host_python': sys.executable,
    'host_python_version': sys.version,
    'platform': platform.platform(),
    'venv': str(VENV),
}

ROOT.mkdir(parents=True, exist_ok=True)
py = VENV / 'bin' / 'python'

# Ubuntu may omit python3-venv/ensurepip. A no-pip venv still works, and pip can
# be bootstrapped without root privileges using PyPA's official helper.
if not py.exists():
    if VENV.exists():
        shutil.rmtree(VENV)
    run([sys.executable, '-m', 'venv', '--without-pip', str(VENV)], timeout=300)

pip_check = run([str(py), '-m', 'pip', '--version'], timeout=60, check=False)
if pip_check.returncode != 0:
    get_pip = ROOT / 'get-pip.py'
    print('+ download https://bootstrap.pypa.io/get-pip.py', flush=True)
    urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', get_pip)
    run([str(py), str(get_pip)], timeout=900)

run([str(py), '-m', 'pip', 'install', '--upgrade', *PACKAGES], timeout=7200)

verify = r'''
import json, platform, sys
out = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform()}
import numpy, scipy, numba, sklearn
out.update(numpy=numpy.__version__, scipy=scipy.__version__, numba=numba.__version__, sklearn=sklearn.__version__)
import torch
out['torch'] = torch.__version__
out['cuda_available'] = bool(torch.cuda.is_available())
out['cuda_version'] = torch.version.cuda
out['cuda_device_count'] = torch.cuda.device_count()
out['cuda_devices'] = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    out['cuda_devices'].append({'index': i, 'name': p.name, 'total_memory': p.total_memory, 'major': p.major, 'minor': p.minor})
if torch.cuda.is_available():
    x = torch.randn((2048, 2048), device='cuda:0')
    y = x @ x
    torch.cuda.synchronize()
    out['cuda_smoke_mean'] = float(y.mean().cpu())
print(json.dumps(out, indent=2, sort_keys=True))
'''
p = run([str(py), '-c', verify], timeout=600)
try:
    verification = json.loads(p.stdout[p.stdout.find('{'):])
except Exception:
    verification = {'raw': p.stdout}
summary['verification'] = verification
summary['finished_at'] = time.time()
summary['ok'] = bool(verification.get('cuda_available')) and verification.get('cuda_device_count', 0) >= 2
(ART / 'workstation_env.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
if not summary['ok']:
    raise SystemExit(2)
