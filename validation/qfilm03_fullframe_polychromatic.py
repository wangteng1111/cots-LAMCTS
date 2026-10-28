#!/usr/bin/env python3
"""Full-frame polychromatic Qfilm512 versus DeepLens Huygens fixed-film PSF MTF.
Incoherent sum of same physical 5um-grid Huygens monochromatic PSFs. No
per-wavelength recentering in post-processing. DeepLens 'recenter=True'
chooses chief at primary (550nm) on each call, common global patch origin.
Per-color normalization intentionally follows Qfilm's unit-PSF convention;
absolute irradiance and real film spectral response remain uncertified.
"""
from __future__ import annotations
import sys,os,json,time,math,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from evaluator.qfilm02 import Config,WAVES,WEIGHTS,evaluate as qeval
from validation.qfilm_deeplens_same_prescription import thin_lens,internal_stop_positive_lens,conversion,mtf_from_psf,BK7
from deeplens import GeoLens
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_poly_fullframe"
OUT.mkdir(parents=True,exist_ok=True)
PATH=OUT/"polychromatic_huygens_36x24.json"
FREQS=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)
HS=(0.,.5,1.)
WEIGHTS_N=np.asarray(WEIGHTS,dtype=float)/sum(WEIGHTS)
def main():
 p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
 results={"schema":"qfilm03-36x24-polychromatic-mtf-v1",
   "physical_frame":[36.,24.],"lambda_nm":list(WAVES),"weights":WEIGHTS_N.tolist(),
   "PSF_origin":"DeepLens primary-wavelength (550nm) chief image coordinate",
   "deep_sensor_n":9600,"sensor_width_mm":48.,"pixel_pitch_mm":.005,
   "PSF_crop":128,"spp_per_lambda":1000000,
   "photometric_flux_certified":False,"cases":{}}
 for name,p in (("singlet",p),("internal_stop_asphere",internal_stop_positive_lens())):
  c={"fields":[]};results["cases"][name]=c
  try:
   lens_data,_=conversion(p,sensor_size=48.,sensor_res=9600,material_mode="bk7")
   j=OUT/(name+".json");j.write_text(json.dumps(lens_data))
   lens=GeoLens(filename=str(j),device="cuda:0",dtype=torch.float64)
   cfg=Config(fields=3,pupil_n=512,fft_n=2048,max_fields_per_chunk=1,
      wavelengths_nm=WAVES,spectral_weights=WEIGHTS,
      frame_width_mm=36.,frame_height_mm=24.,
      target_efl_mm=float(p["design_spec"]["efl_target_mm"]))
   with torch.inference_mode():qr=qeval(p,cfg,device="cuda:0")
   for fi,h in enumerate(HS):
    angle=math.atan(h*math.hypot(36.,24.)/2/p["design_spec"]["efl_target_mm"])
    point=[0.,math.tan(angle)*float(lens.foclen)/(float(lens.sensor_size[1])/2),-1e6]
    samples=[];polys=[]
    for wi,(w,weight) in enumerate(zip(WAVES,WEIGHTS_N)):
     torch.manual_seed(20261028+int(w)*19+int(h*10)*79)
     t=time.perf_counter()
     with torch.inference_mode():
      psf=lens.psf_huygens(points=point,ks=128,wvln=w*.001,spp=1000000,recenter=True)
     torch.cuda.synchronize()
     psf=psf.cpu().numpy().astype(np.float64)
     psf=psf/psf.sum()
     y,x=np.indices(psf.shape)
     centroid_x=float(((x+0.5-64)*.005*psf).sum())
     centroid_y=float(((y+0.5-64)*.005*psf).sum())
     samples.append({"wavelength_nm":w,"seconds":time.perf_counter()-t,
       "window_relative_centroid_x_um":centroid_x*1000,
       "window_relative_centroid_y_um":centroid_y*1000,
       "PSF_peak_normalized":float(psf.max())})
     polys.append(float(weight)*psf)
    mixed=np.sum(polys,axis=0);mixed/=mixed.sum()
    x,y=mtf_from_psf(mixed,.005,FREQS)
    mt,ms=(x,y) if h==0 else (y,x)
    q=qr["Q_image"][fi,10:30]
    errors=np.abs(np.r_[mt,ms]-q)
    field={"h":h,"angle_deg":math.degrees(angle),
       "DeepLens_poly_MTF_T":mt,"DeepLens_poly_MTF_S":ms,
       "Qfilm_poly_MTF_T":q[:10].tolist(),"Qfilm_poly_MTF_S":q[10:].tolist(),
       "max_abs_poly_MTF_difference":float(errors.max()),
       "mean_abs_poly_MTF_difference":float(errors.mean()),
       "p95_abs_poly_MTF_difference":float(np.quantile(errors,.95)),
       "per_frequency_max_abs_difference":np.maximum(errors[:10],errors[10:]).tolist(),
       "chromatic_PSF_samples":samples,
       "MTF_nyquist_lpmm":100.}
    c["fields"].append(field)
    print("POLY",name,h,field["max_abs_poly_MTF_difference"],flush=True)
    PATH.write_text(json.dumps(results,indent=2)+"\n")
  except Exception as e:
   c["error"]=repr(e);c["traceback"]=traceback.format_exc()[-4000:]
   print("POLY_FAIL",name,repr(e),flush=True)
  PATH.write_text(json.dumps(results,indent=2)+"\n")
 print("POLY_COMPLETE",json.dumps({name:{"n":len(c["fields"]),"max":max((v["max_abs_poly_MTF_difference"] for v in c["fields"]),default=None),"error":c.get("error")} for name,c in results["cases"].items()}),flush=True)
if __name__=="__main__":main()
