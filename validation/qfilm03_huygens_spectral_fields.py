#!/usr/bin/env python3
"""DeepLens native Huygens vs Qfilm0.3: multiple wavelengths and full field envelope.

Runs exact matching N-BK7 prescription, same mechanical sensor/focus, same
physical field angle and true lp/mm. Reference Huygens PSFs are monochromatic,
recentered at monochromatic chief only for PSF shape/MTF. This does NOT verify
polychromatic centroid, color MTF, chief displacement, throughput or transmission.
Huygens itself must be convergence-qualified by separate window / sampling tests.
"""
from __future__ import annotations
import sys,os,json,time,math,copy,traceback,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from evaluator.qfilm02 import Config,WAVES
from evaluator.qfilm03 import evaluate,ValidationConfig
from validation.qfilm_deeplens_same_prescription import (thin_lens,internal_stop_positive_lens,
                                                            conversion,mtf_from_psf,BK7)
from deeplens import GeoLens

OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_huygens"
OUT.mkdir(parents=True,exist_ok=True)
TARGET=OUT/"huygens_spectral_fields.json"
WAVES_L=(430.,470.,510.,550.,590.,630.,670.)
FREQS=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)

def case_list():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 q=copy.deepcopy(p);q["design_spec"]["stop_diameter_mm"]*=1.25
 return [("singlet",p),("internal_stop_asphere",internal_stop_positive_lens()),
         ("singlet_large_stop",q)]
def qfilm_values(p):
 values={}
 for n in (256,512):
  for w in WAVES_L:
   cfg=Config(fields=11,pupil_n=n,fft_n=4*n,
       max_fields_per_chunk=max(1,min(11,int(11*(128/n)**2))),
       wavelengths_nm=(w,),spectral_weights=(1.,),
       frame_width_mm=8.,frame_height_mm=6.,
       target_efl_mm=float(p["design_spec"]["efl_target_mm"]))
   start=time.perf_counter()
   with torch.inference_mode():
    r=evaluate(p,cfg,device="cuda:0",
               validation=ValidationConfig(mode="fast",base_backend="torch"))
   torch.cuda.synchronize()
   values[(n,w)]=r["Q_image"]
   print("QMONO",n,w,round(time.perf_counter()-start,3),flush=True)
 return values
def run():
 rec={"schema":"qfilm03-spectral-field-huygens-v1",
      "method":"upstream DeepLens GeoLens.psf_huygens vs Qfilm 0.3 monochromatic",
      "source_SHA":"911bb9d334fce7e20a4407c4506ac683b52ff246",
      "physics_assumption":"monochromatic recentered PSF shape, NOT absolute film position",
      "focus_protocol":"fixed mechanical sensor plane from shared prescription",
      "MTF_freq_lpmm":FREQS,"wavelengths_nm":WAVES_L,
      "deep_spp":1000000,"ks":96,"sensor_resolution":3072,
      "case_results":{},"label_certified":False}
 for case,p in case_list():
  # Third candidate is a larger stop perturbation of the same commercial singlet.
  fields=range(11) if case!="singlet_large_stop" else (0,2,5,8,10)
  wavelengths=WAVES_L if case!="singlet_large_stop" else (430.,550.,670.)
  cr={"fields_run":list(fields),"wavelengths_nm":list(wavelengths),"records":[],
      "errors_by_wavelength":{},"errors_by_field":{}}
  rec["case_results"][case]=cr
  try:
   q=qfilm_values(p)
   optical,_=conversion(p,sensor_size=12.8,sensor_res=3072,material_mode="bk7")
   lp=OUT/f"{case}_native_deeplens.json";lp.write_text(json.dumps(optical,indent=2)+"\n")
   lens=GeoLens(filename=str(lp),device="cuda:0",dtype=torch.float64)
   cr["EFL_deep_mm"]=float(lens.foclen)
   cr["EFL_qfilm_mm"]=float(evaluate(p,Config(fields=2,pupil_n=64,fft_n=256,
       wavelengths_nm=(550.,),spectral_weights=(1.,),target_efl_mm=p["design_spec"]["efl_target_mm"]),
       device="cuda:0",validation=ValidationConfig(mode="fast",base_backend="torch"))["efl_mm"])
   cr["EFL_difference_mm"]=abs(cr["EFL_deep_mm"]-cr["EFL_qfilm_mm"])
   cr["fixed_film_z_mm"]=float(p["image_z_mm"])
   cr["sensor_pixel_pitch_mm"]=float(lens.pixel_size)
   cr["PSF_window_mm"]=96*float(lens.pixel_size)
   for w in wavelengths:
    errs=[]
    for fi in fields:
     h=fi/10.0;alpha=math.atan(h*5./float(p["design_spec"]["efl_target_mm"]))
     normalized_y=math.tan(alpha)*float(lens.foclen)/(float(lens.sensor_size[1])/2)
     point=[0.,normalized_y,-1.e6]
     seed=int(20261028+fi*107+w*17)
     torch.manual_seed(seed)
     t=time.perf_counter()
     with torch.inference_mode():
      psf=lens.psf_huygens(points=point,ks=96,wvln=w*.001,
                           spp=1000000,recenter=True)
     torch.cuda.synchronize()
     rt=time.perf_counter()-t
     px,py=mtf_from_psf(psf.detach().cpu().numpy(),float(lens.pixel_size),FREQS)
     # Qfilm field is +X, DeepLens source field is +Y.
     ht,hs=(px,py) if fi==0 else (py,px)
     qa=q[(256,w)][fi];qb=q[(512,w)][fi]
     hmtf=np.r_[ht,hs]
     da=np.abs(hmtf-qa[10:30]);db=np.abs(hmtf-qb[10:30])
     resolution=np.abs(qa[10:30]-qb[10:30])
     entry={"h":h,"field_index":fi,"wavelength_nm":w,
          "PSF_sum":float(psf.sum()),"deep_seconds":rt,
          "DeepLens_MTF_t":ht,"DeepLens_MTF_s":hs,
          "Qfilm256_MTF_t":qa[10:20].tolist(),"Qfilm256_MTF_s":qa[20:30].tolist(),
          "Qfilm512_MTF_t":qb[10:20].tolist(),"Qfilm512_MTF_s":qb[20:30].tolist(),
          "mtf_max_diff_Q256_vs_Deep":float(da.max()),
          "mtf_max_diff_Q512_vs_Deep":float(db.max()),
          "mtf_mean_diff_Q512_vs_Deep":float(db.mean()),
          "mtf_p95_diff_Q512_vs_Deep":float(np.quantile(db,.95)),
          "mtf_max_diff_Q256_vs_Q512":float(resolution.max()),
          "mtf_rms_diff_Q512_vs_Deep":float(np.sqrt(np.mean(db**2))),
          "Qfilm512_valid":bool(qb[30]>.5),"DeepLens_seed":seed}
     cr["records"].append(entry);errs.append(float(db.max()))
     print("HUYGENS",case,w,h,"err",round(float(db.max()),5),
           "Qresolution",round(float(resolution.max()),5),"secs",round(rt,3),flush=True)
     TARGET.write_text(json.dumps(rec,indent=2)+"\n")
    cr["errors_by_wavelength"][str(w)]={"max":max(errs),"mean":statistics.mean(errs)}
   for fi in fields:
    e=[x["mtf_max_diff_Q512_vs_Deep"] for x in cr["records"] if x["field_index"]==fi]
    cr["errors_by_field"][str(fi)]={"max":max(e),"mean":statistics.mean(e)}
   cr["summary"]={
    "n_huygens_PSFs":len(cr["records"]),
    "max_MTF_disagreement":max(x["mtf_max_diff_Q512_vs_Deep"] for x in cr["records"]),
    "p95_max_per_PSF":float(np.quantile([x["mtf_max_diff_Q512_vs_Deep"] for x in cr["records"]],.95)),
    "mean_max_per_PSF":statistics.mean(x["mtf_max_diff_Q512_vs_Deep"] for x in cr["records"]),
    "mean_huygens_seconds":statistics.mean(x["deep_seconds"] for x in cr["records"]),
    "max_Q256_to_Q512":max(x["mtf_max_diff_Q256_vs_Q512"] for x in cr["records"])}
   print("CASE_SUMMARY",case,json.dumps(cr["summary"]),flush=True)
  except Exception as e:
   cr["error"]=repr(e);cr["traceback"]=traceback.format_exc()[-6000:]
   print("CASE_ERROR",case,repr(e),flush=True)
  TARGET.write_text(json.dumps(rec,indent=2)+"\n")
 print("FINAL_STATUS",json.dumps({c:k.get("summary",k.get("error")) for c,k in rec["case_results"].items()}),flush=True)
if __name__=="__main__":run()
