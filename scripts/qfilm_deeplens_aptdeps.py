#!/usr/bin/env python3
"""Provision DeepLens rendering dependencies via allowed Ubuntu apt mirror (no root/pip)."""
from __future__ import annotations
import os,sys,subprocess,json,pathlib,traceback
root=pathlib.Path('/var/lib/cots-lamcts/deeplens_apt_deps')
root.mkdir(parents=True,exist_ok=True)
cache=root/'packages';cache.mkdir(exist_ok=True)
pkgs=['python3-matplotlib','python3-contourpy','python3-cycler','python3-dateutil','python3-fonttools',
'python3-kiwisolver','python3-packaging','python3-pil','python3-pyparsing','python3-six',
'python3-einops','python3-opencv','python3-skimage','python3-imageio','python3-tifffile','python3-networkx']
report={'schema':'deeplens-apt-isolated-deps-v1','installed':[],'failed':[]}
for name in pkgs:
 existing=list(cache.glob(name+'_*.deb'))
 if not existing:
  try:
   p=subprocess.run(['apt','download',name],cwd=cache,capture_output=True,text=True,timeout=85)
   if p.returncode!=0:
    report['failed'].append({'name':name,'message':p.stderr[-650:]});print('APT_FAIL',name,p.stderr[-400:],flush=True);continue
  except Exception as e:report['failed'].append({'name':name,'error':repr(e)});continue
 existing=list(cache.glob(name+'_*.deb'))
 for filename in existing:
  try:
   p=subprocess.run(['dpkg-deb','-x',str(filename),str(root)],capture_output=True,text=True,timeout=45)
   if p.returncode:raise RuntimeError(p.stderr)
   report['installed'].append(name);print('APT_OK',name,flush=True)
  except Exception as e:report['failed'].append({'name':name,'error':repr(e)})
sys.path.insert(0,str(root/'usr/lib/python3/dist-packages'))
up=pathlib.Path('/var/lib/cots-lamcts/deeplens_upstream');sys.path.insert(0,str(up))
for name in ['matplotlib','cv2','einops','skimage']:
 try:
  m=__import__(name);report[name]={'ok':True,'version':str(getattr(m,'__version__','?'))}
 except Exception as e:report[name]={'ok':False,'error':str(e)[:500]}
print('DEPS_RESULT',json.dumps(report),flush=True)
out=pathlib.Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','/tmp'))/'deeplens_aptdeps.json';out.write_text(json.dumps(report,indent=2)+'\n')