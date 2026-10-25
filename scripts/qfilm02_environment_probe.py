#!/usr/bin/env python3
from __future__ import annotations
import os,sys,sysconfig,shutil,glob,json,subprocess,platform
def gl(pattern):return glob.glob(pattern)[:30]
data={
 'python':sys.executable,'python_version':sys.version,'sysconfig_include':sysconfig.get_paths().get('include'),
 'env_cache':{k:v for k,v in os.environ.items() if any(t in k for t in ('TRITON','CUDA','PYTHON','TORCH'))},
 'nvcc':shutil.which('nvcc'),'gcc':shutil.which('gcc'),
 'common_dev_headers':gl('/usr/include/python*/Python.h')+gl('/usr/local/include/python*/Python.h')+gl('/opt/**/include/python*/Python.h'),
 'cuda_headers':gl('/usr/local/cuda*/include/cuda_runtime.h')+gl('/usr/local/cuda*/bin/nvcc'),
 'python_h_locations':[],
}
for path in ['/usr/include','/usr/local/include','/var/lib/cots-lamcts/venv/include','/opt/conda/include']:
 try:
  x=subprocess.run(['find',path,'-name','Python.h','-o','-name','cuda_runtime.h'],capture_output=True,text=True,timeout=8)
  data['python_h_locations']+=x.stdout.splitlines()[:30]
 except Exception as e:data['python_h_locations']+=['ERR '+repr(e)]
print(json.dumps(data,indent=2))
