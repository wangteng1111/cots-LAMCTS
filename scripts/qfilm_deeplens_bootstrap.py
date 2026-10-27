#!/usr/bin/env python3
"""Fetch pinned upstream DeepLens into job-local files; do not mutate Torch env."""
from __future__ import annotations
import os, sys, json, subprocess, importlib.util, pathlib, traceback, time
SHA = '911bb9d334fce7e20a4407c4506ac683b52ff246'
ROOT = pathlib.Path(os.environ.get('COTS_JOB_ARTIFACT_DIR', '/tmp/qfilm_deeplens'))
ROOT.mkdir(parents=True,exist_ok=True)
UP=pathlib.Path('/var/lib/cots-lamcts/deeplens_upstream')
if not UP.parent.exists() or not os.access(UP.parent,os.W_OK):UP=ROOT/'deeplens_upstream'
log={'schema':'qfilm-deeplens-bootstrap-v1','source':'https://github.com/vccimaging/DeepLens','sha':SHA,'upstream_path':str(UP)}
def run(args, timeout=240):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 print('CMD',args,'returncode',p.returncode,'stdout_tail',p.stdout[-1100:],'stderr_tail',p.stderr[-1800:],flush=True)
 if p.returncode!=0:raise RuntimeError('command failure '+repr(args)+': '+p.stderr[-1000:])
 return p.stdout.strip()
try:
 if not (UP/'.git').is_dir():
  run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/vccimaging/DeepLens.git',str(UP)],timeout=180)
 current=run(['git','-C',str(UP),'rev-parse','HEAD'])
 if current!=SHA:
  run(['git','-C',str(UP),'fetch','--depth','1','origin',SHA],timeout=180)
 if current!=SHA or not (UP/'deeplens'/'__init__.py').is_file():
  run(['git','-C',str(UP),'checkout','--force',SHA],timeout=180)
 log['checked_out']=run(['git','-C',str(UP),'rev-parse','HEAD'])
 sys.path.insert(0,'/var/lib/cots-lamcts/deeplens_python_deps')
 sys.path.insert(0,str(UP))
 for dep in ['torch','torchvision','cv2','matplotlib','einops','scipy','skimage','deeplens']:
  try:
   mod=__import__(dep)
   log[dep]={'available':True,'version':getattr(mod,'__version__','unknown')}
  except Exception as e:
   log[dep]={'available':False,'error':str(e)[:900]}
 print('BOOTSTRAP',json.dumps(log,ensure_ascii=False),flush=True)
except Exception as e:
 log['error']=repr(e);log['traceback']=traceback.format_exc();print('BOOTSTRAP_FAIL',json.dumps(log,ensure_ascii=False),flush=True)
finally:
 (ROOT/'deeplens_bootstrap.json').write_text(json.dumps(log,ensure_ascii=False,indent=2)+'\n')
if 'error' in log or not log.get('deeplens',{}).get('available'):sys.exit(2)