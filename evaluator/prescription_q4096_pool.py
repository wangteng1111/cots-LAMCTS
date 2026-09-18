"""Persistent multi-GPU pool for arbitrary Alpha Lense prescriptions.

One long-lived process is bound to each GPU.  Each worker evaluates prescriptions
serially, so the legacy evaluator's temporary module-global configuration is
isolated per process.  Physics results are cached by prescription optical hash
and evaluator/DesignSpec signature under /var/lib/cots-lamcts.
"""
from __future__ import annotations
import concurrent.futures as cf
import hashlib,json,multiprocessing as mp,os
from pathlib import Path
from typing import Iterable
from alpha_lense.stage_pipeline_v01 import Prescription

DEFAULT_CACHE=Path("/var/lib/cots-lamcts/q4096_prescription_v3")
_WORKER_GPU=None
_WORKER_EVAL=None

def _init_worker(gpu:int):
    global _WORKER_GPU,_WORKER_EVAL
    _WORKER_GPU=int(gpu)
    os.environ["COTS_EVAL_CUDA_DEVICE"]=str(_WORKER_GPU)
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
        os.environ.setdefault(k,"1")
    from evaluator import prescription_q4096 as P
    P.CUDA.set_cuda_device(_WORKER_GPU)
    _WORKER_EVAL=P

def _worker_eval(p:Prescription)->dict:
    if _WORKER_EVAL is None:raise RuntimeError("prescription CUDA worker not initialized")
    r=_WORKER_EVAL.evaluate_prescription(p,device=int(_WORKER_GPU))
    _WORKER_EVAL.CUDA.synchronize()
    return r

def _signature(p:Prescription)->str:
    from evaluator.prescription_q4096 import evaluator_signature,ADAPTER_VERSION
    s=evaluator_signature(p)
    raw=f"{ADAPTER_VERSION}|{p.optical_hash()}|{s}"
    return hashlib.sha256(raw.encode()).hexdigest()

def _atomic(path:Path,obj:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj,separators=(",",":")))
    os.replace(tmp,path)

def _read(path:Path):
    try:
        if not path.is_file():return None
        x=json.loads(path.read_text())
        if not isinstance(x,dict) or "J" not in x or not x.get("config_hash") or x.get("config_hash")=="invalid":return None
        return x
    except Exception:return None

class PrescriptionQ4096Pool:
    def __init__(self,gpus=(0,1),workers_per_gpu:int=1,cache_dir:str|Path=DEFAULT_CACHE,write_cache:bool=True):
        self.gpus=tuple(map(int,gpus))
        if not self.gpus:raise ValueError("at least one GPU required")
        self.workers_per_gpu=max(1,int(workers_per_gpu))
        self.cache_dir=Path(cache_dir);self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.write_cache=bool(write_cache);self._rr=0
        ctx=mp.get_context("spawn")
        self.executors=[cf.ProcessPoolExecutor(max_workers=self.workers_per_gpu,mp_context=ctx,initializer=_init_worker,initargs=(g,)) for g in self.gpus]
    @property
    def capacity(self):return len(self.gpus)*self.workers_per_gpu
    def _submit(self,p):
        ex=self.executors[self._rr%len(self.executors)];self._rr+=1
        return ex.submit(_worker_eval,p)
    def evaluate(self,prescriptions:Iterable[Prescription],bypass_cache:bool=False)->list[dict]:
        ps=list(prescriptions);out=[None]*len(ps);groups={}
        for i,p in enumerate(ps):
            k=_signature(p);groups.setdefault(k,{"p":p,"pos":[]})["pos"].append(i)
        futs={}
        for k,g in groups.items():
            path=self.cache_dir/f"{k}.json"
            cached=None if bypass_cache else _read(path)
            if cached is not None:
                for i in g["pos"]:out[i]=dict(cached)
            else:
                fut=self._submit(g["p"]);futs[fut]=(k,g["pos"],path)
        for fut in cf.as_completed(futs):
            k,pos,path=futs[fut];r=fut.result()
            if self.write_cache and r.get("config_hash") not in (None,"invalid"):_atomic(path,r)
            for i in pos:out[i]=dict(r)
        if any(x is None for x in out):raise RuntimeError("missing prescription pool result")
        return out
    def close(self):
        for ex in self.executors:ex.shutdown(wait=True,cancel_futures=False)
    def __enter__(self):return self
    def __exit__(self,*exc):self.close();return False
