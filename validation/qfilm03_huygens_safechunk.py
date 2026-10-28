#!/usr/bin/env python3
"""Memory-safe exact DeepLens Huygens integration by changing ONLY ray batch size.

Upstream function body is dynamically loaded from the pinned repository;
the sole substitution is 10000 -> 1024 rays per accumulation block.
Upstream git checkout is NOT modified and optical equations stay identical.
Cross-validation: seed-matched 96x96 PSFs from original and chunked method.
"""
from __future__ import annotations
import inspect,textwrap,hashlib,torch,json,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from validation.qfilm_deeplens_same_prescription import thin_lens,conversion,BK7
from deeplens import GeoLens
def patch_huygens(batch=1024):
 old=GeoLens.psf_huygens
 source=textwrap.dedent(inspect.getsource(old))
 needle="batch_size = min(num_valid, 10_000)"
 assert source.count(needle)==1,"upstream DeepLens Huygens method changed unexpectedly"
 altered=source.replace(needle,f"batch_size = min(num_valid, {batch:,})")
 ns={}
 exec(compile(altered,old.__code__.co_filename,"exec"),old.__globals__,ns)
 new=ns["psf_huygens"]
 return old,new,{"original_sha256":hashlib.sha256(source.encode()).hexdigest(),
                  "modified_sha256":hashlib.sha256(altered.encode()).hexdigest(),
                  "only_change":"Huygens ray-batch size 10000 -> "+str(batch)}
def parity(original,chunked):
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 d,_=conversion(p,sensor_size=12.8,sensor_res=3072,material_mode="bk7")
 out=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))
 out.mkdir(parents=True,exist_ok=True)
 file=out/"chunk_parity_lens.json";file.write_text(json.dumps(d))
 lens=GeoLens(filename=str(file),device="cuda:0",dtype=torch.float64)
 point=[0.,0.,-1e6];seed=314159
 torch.manual_seed(seed)
 with torch.inference_mode():a=original(lens,points=point,ks=96,wvln=.55,spp=65536,recenter=True)
 torch.manual_seed(seed)
 with torch.inference_mode():b=chunked(lens,points=point,ks=96,wvln=.55,spp=65536,recenter=True)
 diff=(a-b).abs()
 result={"PSF_max_abs":float(diff.max()),"PSF_mean_abs":float(diff.mean()),
         "PSF_sum_a":float(a.sum()),"PSF_sum_b":float(b.sum())}
 assert result["PSF_max_abs"]<1e-8,result
 return result
def main():
 original,chunked,meta=patch_huygens(1024)
 test=parity(original,chunked)
 GeoLens.psf_huygens=chunked
 out=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))
 report={"schema":"Qfilm03-batch-invariance-v1","upstream_SHA":"911bb9d334fce7e20a4407c4506ac683b52ff246",
         **meta,"parity":test,"optical_physics_changed":False}
 (out/"huygens_safechunk_parity.json").write_text(json.dumps(report,indent=2)+"\n")
 print("Huygens_BATCH_INVARIANCE",json.dumps(report),flush=True)
 from validation.qfilm03_huygens_reference_convergence import main as run_tests
 run_tests()
if __name__=="__main__":main()
