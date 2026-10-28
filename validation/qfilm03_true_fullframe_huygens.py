#!/usr/bin/env python3
"""TRUE 36x24mm full-frame fixed-film comparison (not legacy 8x6mm sample field).

DeepLens physical sensor metadata is deliberately made 48x48 mm with 9600²
nominal pitch 5um (Nyquist 100 lp/mm).  The Huygens kernel is only 128², so
the large sensor metadata is used for field-angle/physical pixel matching,
not a 9600² ASM FFT.
"""
from __future__ import annotations
import sys,os,json,time,math,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from evaluator.qfilm02 import Config
from evaluator.qfilm03 import evaluate,ValidationConfig
from validation.qfilm_deeplens_same_prescription import (thin_lens,internal_stop_positive_lens,
                                                         BK7,conversion,mtf_from_psf)
from deeplens import GeoLens
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_true_fullframe"
OUT.mkdir(parents=True,exist_ok=True);TARGET=OUT/"fullframe_huygens.json"
FREQS=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)
WAVES=(430.,550.,670.)
FIELDS=(0,1,2,4,6,8,10)
SENSOR_WIDTH=48.;SENSOR_N=9600;KS=128
def main():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 cases=[("singlet",p),("internal_stop_asphere",internal_stop_positive_lens())]
 d={"schema":"qfilm03-true-36x24-huygens-v1",
    "frame_width_mm":36.,"frame_height_mm":24.,
    "radial_image_height_mm":math.hypot(36,24)/2,
    "deep_sensor_width_mm":SENSOR_WIDTH,"deep_sensor_res":SENSOR_N,
    "PSF_patch_pixels":KS,"PSF_patch_mm":KS*SENSOR_WIDTH/SENSOR_N,
    "frequency_lpmm":FREQS,"wavelength_nm":WAVES,
    "reference":"DeepLens upstream pinned Huygens, 1 million rays",
    "not_certified":True,"cases":{}}
 for name,p in cases:
  r={"points":[]};d["cases"][name]=r
  try:
   data,_=conversion(p,sensor_size=SENSOR_WIDTH,sensor_res=SENSOR_N,material_mode="bk7")
   path=OUT/(name+".json");path.write_text(json.dumps(data))
   lens=GeoLens(filename=str(path),device="cuda:0",dtype=torch.float64)
   r["DeepLens_focal_mm"]=float(lens.foclen)
   r["pixel_pitch_mm"]=float(lens.pixel_size)
   for w in WAVES:
    cfg=Config(fields=11,pupil_n=256,fft_n=1024,max_fields_per_chunk=2,
               wavelengths_nm=(w,),spectral_weights=(1.,),
               frame_width_mm=36.,frame_height_mm=24.,
               target_efl_mm=p["design_spec"]["efl_target_mm"])
    with torch.inference_mode():
     qr=evaluate(p,cfg,device="cuda:0",
           validation=ValidationConfig(mode="fast",base_backend="torch"))
    q=qr["Q_image"]
    for fi in FIELDS:
     h=fi/10;alpha=math.atan(h*math.hypot(36,24)/2/p["design_spec"]["efl_target_mm"])
     point=[0.,math.tan(alpha)*float(lens.foclen)/(float(lens.sensor_size[1])/2),-1e6]
     torch.manual_seed(20261028+int(w)*17+fi*100)
     begin=time.perf_counter()
     try:
      with torch.inference_mode():
       psf=lens.psf_huygens(points=point,ks=KS,wvln=w*.001,spp=1000000,recenter=True)
      torch.cuda.synchronize()
      x,y=mtf_from_psf(psf.detach().cpu().numpy(),float(lens.pixel_size),FREQS)
      t,s=(x,y) if fi==0 else (y,x)
      e=np.abs(np.r_[t,s]-q[fi,10:30])
      result={"lambda_nm":w,"field_h":h,
           "image_height_mm":h*math.hypot(36,24)/2,
           "ray_angle_deg":math.degrees(alpha),
           "Qfilm_field_valid":bool(q[fi,30]>.5),
           "max_MTF_error":float(e.max()),"mean_MTF_error":float(e.mean()),
           "MTF_T_DeepLens":t,"MTF_S_DeepLens":s,
           "MTF_T_Qfilm":q[fi,10:20].tolist(),"MTF_S_Qfilm":q[fi,20:30].tolist(),
           "huygens_s":time.perf_counter()-begin,"PSF_sum":float(psf.sum())}
     except Exception as exc:result={"lambda_nm":w,"field_h":h,"error":repr(exc)}
     r["points"].append(result)
     print("FULLFRAME",name,w,h,
       "MTFmax",result.get("max_MTF_error"),"error",result.get("error"),flush=True)
     TARGET.write_text(json.dumps(d,indent=2)+"\n")
   errors=[p["max_MTF_error"] for p in r["points"] if "max_MTF_error" in p]
   r["max_MTF_error"]=max(errors) if errors else None
   r["n_valid_comparisons"]=len(errors)
   r["n_errors"]=len(r["points"])-len(errors)
  except Exception as e:
   r["error"]=repr(e);r["traceback"]=traceback.format_exc()[-3500:]
   print("CASE_FAILED",name,repr(e),flush=True)
  TARGET.write_text(json.dumps(d,indent=2)+"\n")
 print("FULLFRAME_SUMMARY",json.dumps({k:{a:v for a,v in r.items() if a!='points'} for k,r in d["cases"].items()}),flush=True)
if __name__=="__main__":main()
