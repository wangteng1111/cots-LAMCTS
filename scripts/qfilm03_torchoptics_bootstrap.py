#!/usr/bin/env python3
"""Provision pinned TorchOptics source on workstation without touching the existing Torch venv."""
import os, subprocess, pathlib, json
SRC=pathlib.Path("/var/lib/cots-lamcts/torchoptics_upstream")
SHA="34fe9c40f874e53db225e703c5580befe8c523b5"
URL="https://github.com/MatthewFilipovich/torchoptics.git"
SRC.parent.mkdir(parents=True,exist_ok=True)
def run(cmd):
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=180)
 print("CMD",cmd,"RC",r.returncode,"LOG",(r.stdout+"\n"+r.stderr)[-2000:],flush=True)
 if r.returncode:raise RuntimeError("bootstrap command failed")
if not (SRC/".git").exists():
 run(["git","clone","--filter=blob:none","--no-checkout",URL,str(SRC)])
run(["git","-C",str(SRC),"fetch","--depth","1","origin",SHA])
run(["git","-C",str(SRC),"checkout","--force",SHA])
r=subprocess.check_output(["git","-C",str(SRC),"rev-parse","HEAD"],text=True).strip()
assert r==SHA,r
print("TORCHOPTICS_BOOTSTRAP",json.dumps({"source":URL,"sha":r,"path":str(SRC),"has_field":(SRC/"src/torchoptics/fields.py").exists()}),flush=True)
