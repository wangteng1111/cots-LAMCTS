#!/usr/bin/env python3
"""Red-field PSF root-cause: same Qfilm eikonal, FFT vs physical direct Huygens,
with and without true exit pupil area/Jacobian energy weighting; external DeepLens.
"""
from __future__ import annotations
import os,sys,json,math,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from evaluator.qfilm02 import Config,evaluate as qeval
from evaluator.qfilm03_physical_huygens import Propagation,eval_fixed_film,mtf_from_psf_exact
from validation.qfilm_deeplens_same_prescription import thin_lens,internal_stop_positive_lens,conversion,mtf_from_psf,BK7
from deeplens import GeoLens
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_propagation_root_cause"
OUT.mkdir(parents=True,exist_ok=True);TARGET=OUT/"matched_physical_huygens_vs_fft.json"
F=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)
CASES=[("singlet",550.,0.),("singlet",630.,0.),("singlet",670.,0.),
       ("singlet",670.,.1),("internal_stop_asphere",550.,0.),
       ("internal_stop_asphere",630.,0.),("internal_stop_asphere",670.,0.),
       ("internal_stop_asphere",670.,.1)]
def mtd(aa,bb):
 arr=np.abs(np.asarray(aa)-np.asarray(bb))
 return {"max":float(arr.max()),"mean":float(arr.mean()),"rms":float(np.sqrt(np.mean(arr**2))),
   "freq_max":arr.max(axis=-1).tolist()}
def main():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 systems={"singlet":p,"internal_stop_asphere":internal_stop_positive_lens()}
 result={"schema":"qfilm03-fixedfilm-photonic-propagation-source-identification-v1",
  "wavelengths":[550.,630.,670.],"reference":"DeepLens native Huygens 1M rays",
  "physical_frame":[36.,24.],"psf_pitch_um":5.,"psf_n":128,
  "explicit_exit_jacobian":True,"uniform_stop_diagnostic":True,
  "not_certified":True,"results":[]}
 for name,p in systems.items():
  data,_=conversion(p,sensor_size=48.,sensor_res=9600,material_mode="bk7")
  path=OUT/(name+".json");path.write_text(json.dumps(data))
  lens=GeoLens(filename=str(path),device="cuda:0",dtype=torch.float64)
  for cn,w,h in CASES:
   if cn!=name:continue
   rr={"case":name,"lambda_nm":w,"h":h,"methods":{}}
   result["results"].append(rr)
   try:
    cfg=Config(fields=11,pupil_n=128,fft_n=512,max_fields_per_chunk=11,
       wavelengths_nm=(w,),spectral_weights=(1.,),
       frame_width_mm=36.,frame_height_mm=24.,
       target_efl_mm=p["design_spec"]["efl_target_mm"])
    t=time.perf_counter()
    with torch.inference_mode():q=qeval(p,cfg,device="cuda:0")
    qb=np.stack((q["Q_image"][round(h*10),10:20],q["Q_image"][round(h*10),20:30]),axis=-1)
    rr["methods"]["qfilm_fft128"]={"seconds":time.perf_counter()-t,"MTF":qb.tolist()}
    alpha=math.atan(h*math.hypot(36,24)/2/p["design_spec"]["efl_target_mm"])
    point=[0.,math.tan(alpha)*float(lens.foclen)/(float(lens.sensor_size[1])/2),-1e6]
    torch.manual_seed(20261028+int(w)*17+int(h*10)*100)
    with torch.inference_mode():ref=lens.psf_huygens(points=point,ks=128,wvln=w*.001,spp=1000000,recenter=True)
    tx,ty=mtf_from_psf(ref.cpu().numpy(),.005,F)
    deep=np.column_stack((tx,ty)) if h==0 else np.column_stack((ty,tx))
    rr["methods"]["deeplens_random_huygens_1M"]={"MTF":deep.tolist()}
    rr["errors"]={"fft_vs_deeplens":mtd(qb,deep)}
    for n in (64,128):
     for weight in ("uniform_stop","sqrt_exit_jac"):
      start=time.perf_counter()
      try:
       with torch.inference_mode():
        computed=eval_fixed_film(p,w,h,Propagation(pupil_n=n,pixel_n=128,
                             pixel_pitch_um=5.,pixel_block=32,weight_mode=weight))
       direct=mtf_from_psf_exact(computed["psf"],.005,F)
       label=f"qfilm_huygens_{n}_{weight}"
       rr["methods"][label]={"MTF":direct.tolist(),"seconds":time.perf_counter()-start,
        "bad_jacobian_fraction":computed["bad_jacobian_fraction"],
        "jac_min":computed["jac_min"],"jac_p95":computed["jac_p95"],
        "active_rays":computed["valid_rays"]}
       rr["errors"][label+"_vs_deeplens"]=mtd(direct,deep)
       rr["errors"][label+"_vs_fft"]=mtd(direct,qb)
      except Exception as exc:
       rr["methods"][f"qfilm_huygens_{n}_{weight}"]={"error":repr(exc),"traceback":traceback.format_exc()[-2400:]}
    print("PROPAGATION",name,w,h,json.dumps({"errors":rr["errors"],
       "status":{k:v.get("error") for k,v in rr["methods"].items() if "error" in v}}),flush=True)
   except Exception as e:rr["error"]=repr(e);rr["traceback"]=traceback.format_exc()[-4000:]
   TARGET.write_text(json.dumps(result,indent=2)+"\n")
 print("PROPAGATION_SUMMARY",json.dumps([{"case":x["case"],"w":x["lambda_nm"],"h":x["h"],
  "errors":{k:v["max"] for k,v in x.get("errors",{}).items() if k.endswith("_vs_deeplens")}} for x in result["results"]]),flush=True)
if __name__=="__main__":main()
