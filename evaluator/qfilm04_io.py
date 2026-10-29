"""Strict atomic JSON and immutable terminal records for numerical runs."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import numpy as np


def json_safe(value,missing=None,path="$"):
    if missing is None:missing=[]
    if isinstance(value,np.ndarray):value=value.tolist()
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,complex):return {"real":json_safe(value.real,missing,path+".real"),
                                       "imag":json_safe(value.imag,missing,path+".imag")}
    if isinstance(value,float) and not math.isfinite(value):
        missing.append(path);return None
    if isinstance(value,dict):return {str(k):json_safe(v,missing,path+"."+str(k)) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [json_safe(v,missing,path+f"[{i}]") for i,v in enumerate(value)]
    return value


def atomic_json(path,value,immutable=False):
    path=Path(path);missing=[];data=json_safe(value,missing)
    if missing and isinstance(data,dict):
        data["serialization_diagnostics"]={"nonfinite_values_written_as_null":missing}
    payload=json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n"
    path.parent.mkdir(parents=True,exist_ok=True)
    if immutable:
        # Link a fully flushed temporary inode exclusively into its final name.
        # Readers cannot observe a half-written terminal and old publications
        # cannot replace it. The temporary file is on the same filesystem.
        import uuid
        temp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
        with temp.open("w") as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        try:
            os.link(temp,path)
        except FileExistsError:
            if path.read_text()!=payload:raise ValueError("immutable terminal record already exists")
        finally:temp.unlink(missing_ok=True)
        return
    temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("w") as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
