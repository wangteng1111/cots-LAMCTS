#!/usr/bin/env python3
"""Non-root portable install of Python 3.12 development headers for Triton JIT."""
from __future__ import annotations
import subprocess,os,sys,shutil,json,tarfile,glob
from pathlib import Path
root=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","/tmp"))/"python_headers"
root.mkdir(parents=True,exist_ok=True)
def run(argv,timeout=100):
 p=subprocess.run(argv,cwd=root,capture_output=True,text=True,timeout=timeout)
 print("COMMAND",argv,"RC",p.returncode,"stdout",p.stdout[-2000:],"stderr",p.stderr[-2000:],flush=True)
 return p
print("APT_SHOW")
run(["apt-cache","policy","libpython3.12-dev"])
p=run(["apt","download","libpython3.12-dev"],timeout=120)
files=list(root.glob("*.deb"))
if p.returncode or not files:raise SystemExit("apt download failed")
dest=root/"unpacked";dest.mkdir(exist_ok=True)
run(["dpkg-deb","-x",str(files[0]),str(dest)],timeout=60)
inc=dest/"usr"/"include"/"python3.12"
assert (inc/"Python.h").is_file(),"no Python.h in package"
os.environ["C_INCLUDE_PATH"]=str(inc)+os.pathsep+os.environ.get("C_INCLUDE_PATH","")
os.environ["CPATH"]=str(inc)+os.pathsep+os.environ.get("CPATH","")
print("AVAILABLE_PYTHON_HEADER",str(inc),flush=True)
# Python header follows expected ABI; now execute suite without spawning Python.
from runpy import run_path
run_path(str(Path(__file__).parents[1]/"validation"/"qfilm02_accelerated.py"),run_name="__main__")
