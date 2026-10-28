#!/usr/bin/env python3
"""Qfilm0.3 reference Huygens convergence: ray sampling, PSF window and fixed-film focus.

Runs 250K vs 1M Huygens rays (same deterministic random state) and same
1M rays with a doubled PSF window. DeepLens normalizes EACH cropped PSF:
this tests window sensitivity, not photometric power conservation.
"""
from __future__ import annotations
import sys,os,json,time,copy,math,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from validation.qfilm_deeplens_same_prescription import (thin_lens,internal_stop_positive_lens,
                                                         conversion,mtf_from_psf,BK7)
from deeplens import GeoLens
from evaluator.qfilm02 import Config
from evaluator.qfilm03 import evaluate,ValidationConfig
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_huygens_quality"
OUT.mkdir(parents=True,exist_ok=True)
TARGET=OUT/"reference_sampling_and_focus.json"
WAVES=(430.,550.,670.);FREQS=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)

def maken(name,p):
 d,_=conversion(p,sensor_size=12.8,sensor_res=3072,material_mode="bk7")
 path=OUT/(name+".json");path.write_text(json.dumps(d))
 return GeoLens(filename=str(path),device="cuda:0",dtype=torch.float64)
def psf_metrics(lens,point,fi,w,spp,ks,seed):
 torch.manual_seed(seed)
 start=time.perf_counter()
 with torch.inference_mode():
  psf=lens.psf_huygens(points=point,ks=ks,wvln=w*.001,spp=spp,recenter=True)
 torch.cuda.synchronize()
 tm=time.perf_counter()-start
 x,y=mtf_from_psf(psf.detach().cpu().numpy(),float(lens.pixel_size),FREQS)
 t,s=(x,y) if fi==0 else (y,x)
 return np.r_[t,s],tm
def case(name,p,rec):
 lens=maken(name,p)
 summary={"method":"DeepLens Huygens","source":"official upstream 911bb9d",
     "name":name,"entries":[],"PSF_window_96_mm":96*float(lens.pixel_size),
     "PSF_window_192_mm":192*float(lens.pixel_size)}
 for w in WAVES:
  for fi in (0,5,10):
   h=fi/10;ang=math.atan(h*5./float(p["design_spec"]["efl_target_mm"]))
   y=math.tan(ang)*float(lens.foclen)/(float(lens.sensor_size[1])/2)
   point=[0.,y,-1.e6]
   seed=20261028+fi*131+int(w)*17
   a,ta=psf_metrics(lens,point,fi,w,262144,96,seed)
   b,tb=psf_metrics(lens,point,fi,w,1000000,96,seed)
   c,tc=psf_metrics(lens,point,fi,w,1000000,192,seed)
   u=np.abs(a-b);v=np.abs(b-c)
   e={"field_h":h,"wavelength_nm":w,
      "spp_262144_to_1M_mtf_max":float(u.max()),
      "spp_262144_to_1M_mtf_mean":float(u.mean()),
      "window_96_to_192_mtf_max":float(v.max()),
      "window_96_to_192_mtf_mean":float(v.mean()),
      "low_ray_time_s":ta,"million96_s":tb,"million192_s":tc}
   summary["entries"].append(e)
   print("REF_CONV",name,w,h,"rays",round(float(u.max()),5),
         "window",round(float(v.max()),5),"time",round(tb+tc,2),flush=True)
   rec["cases"][name]=summary;TARGET.write_text(json.dumps(rec,indent=2)+"\n")
 summary["maximum_ray_sampling_MTF_change"]=max(e["spp_262144_to_1M_mtf_max"] for e in summary["entries"])
 summary["maximum_window_MTF_change"]=max(e["window_96_to_192_mtf_max"] for e in summary["entries"])
 summary["mean_window_MTF_change"]=sum(e["window_96_to_192_mtf_mean"] for e in summary["entries"])/len(summary["entries"])
 return summary
def focus_sensitivity(name,p,rec):
 # Verified identical fixed film shifts; tests whether 10um / 50um focus errors
 # dominate claimed PSF discrepancies, not a focus optimization.
 focus={}
 for delta in (-.05,-.01,0.,.01,.05):
  p2=copy.deepcopy(p);p2["image_z_mm"]=float(p2["image_z_mm"])+delta
  lens=maken(name+"_focus_"+str(delta),p2)
  config=Config(fields=11,pupil_n=256,fft_n=1024,max_fields_per_chunk=2,
                wavelengths_nm=(550.,),spectral_weights=(1.,),
                frame_width_mm=8.,frame_height_mm=6.,
                target_efl_mm=float(p2["design_spec"]["efl_target_mm"]))
  with torch.inference_mode():
   qr=evaluate(p2,config,device="cuda:0",validation=ValidationConfig(mode="fast",base_backend="torch"))
  rows={}
  for fi in (0,5,10):
   ang=math.atan((fi/10)*5./float(p2["design_spec"]["efl_target_mm"]))
   y=math.tan(ang)*float(lens.foclen)/(float(lens.sensor_size[1])/2)
   ref,t=psf_metrics(lens,[0.,y,-1.e6],fi,550.,1000000,96,
                        20261028+fi*107+550*17)
   q=qr["Q_image"][fi]
   rows[str(fi)]={"Huygens_MTF":ref.tolist(),"Qfilm_MTF":q[10:30].tolist(),
                  "max_abs_error":float(np.max(np.abs(ref-q[10:30]))),"huygens_s":t}
  focus[str(delta)]={"image_z_mm":p2["image_z_mm"],"fields":rows}
  rec["focus_sweep"][name]=focus
  TARGET.write_text(json.dumps(rec,indent=2)+"\n")
  print("FOCUS",name,delta,{i:round(x["max_abs_error"],4) for i,x in rows.items()},flush=True)
 return focus
def main():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 lenses=[("singlet",p),("internal_stop_asphere",internal_stop_positive_lens())]
 rec={"schema":"qfilm03-Huygens-reference-convergence-v1","cases":{},"focus_sweep":{},
      "physics_certified":False,"warning":"96/192 crop individually normalized; no flux certification"}
 for name,p in lenses:
  try:rec["cases"][name]=case(name,p,rec)
  except Exception as e:
   rec["cases"][name]={"error":repr(e),"traceback":traceback.format_exc()[-4500:]}
   print("REFERENCE_ERROR",name,repr(e),flush=True)
  TARGET.write_text(json.dumps(rec,indent=2)+"\n")
 # Focus sensitivity only on first lens to keep workload finite
 try:focus_sensitivity("singlet",lenses[0][1],rec)
 except Exception as e:rec["focus_sweep"]["error"]={"error":repr(e),"traceback":traceback.format_exc()[-4500:]}
 TARGET.write_text(json.dumps(rec,indent=2)+"\n")
 print("SUMMARY",json.dumps({k:{x:v for x,v in d.items() if x.startswith("maximum")} for k,d in rec["cases"].items()}),flush=True)
if __name__=="__main__":main()
