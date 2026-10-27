#!/usr/bin/env python3
"""Install pinned DeepLens optional Python dependencies into isolated folder only."""
import os, sys, subprocess, json, pathlib, traceback, time
root=pathlib.Path('/var/lib/cots-lamcts/deeplens_python_deps')
try:
 root.mkdir(parents=True,exist_ok=True)
except Exception:
 root=pathlib.Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','/tmp'))/'deeplens_python_deps';root.mkdir(parents=True,exist_ok=True)
up=pathlib.Path('/var/lib/cots-lamcts/deeplens_upstream')
if not up.exists():raise SystemExit('DeepLens source not cloned')
pack=['matplotlib==3.10.6','opencv-python-headless==4.12.0.88','einops==0.8.1','scikit-image==0.25.2','pillow==11.3.0','contourpy==1.3.3','cycler==0.12.1','fonttools==4.60.1','kiwisolver==1.4.9','pyparsing==3.2.5','packaging==25.0','imageio==2.37.0','tifffile==2025.10.4','lazy-loader==0.4','networkx==3.5','tqdm==4.67.1','pyyaml==6.0.3']
cmd=[sys.executable,'-m','pip','install','--target',str(root),'--no-deps','--disable-pip-version-check','--retries','2','--timeout','35',*pack]
print('INSTALL_TARGET',str(root),flush=True)
p=subprocess.run(cmd,capture_output=True,text=True,timeout=550)
print('PIP',p.returncode,p.stdout[-5000:],p.stderr[-5000:],flush=True)
sys.path.insert(0,str(root));sys.path.insert(0,str(up))
result={'source_sha':'911bb9d334fce7e20a4407c4506ac683b52ff246','pip_returncode':p.returncode,'root':str(root),'torch_before':None,'import_check':{}}
for name in ['torch','matplotlib','cv2','einops','skimage','deeplens']:
 try:
  m=__import__(name);result['import_check'][name]={'ok':True,'version':str(getattr(m,'__version__','unknown'))}
 except Exception as e:result['import_check'][name]={'ok':False,'error':str(e)[:700],'traceback':traceback.format_exc()[-2000:]}
print('IMPORT_CHECK',json.dumps(result,ensure_ascii=False),flush=True)
out=pathlib.Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','/tmp'))/'deeplens_deps.json';out.write_text(json.dumps(result,indent=2)+'\n')
if not result['import_check']['deeplens']['ok']:sys.exit(2)