#!/usr/bin/env python3
"""Qfilm0.4-g 18 real-product/patent optical designs, fixed-film 7-color MTF.

Three shards of 6 cases; source files from pinned DeepLens / attributed
independent patent Zemax reference. The evaluator is NOT compared to an actual
manufactured lens MTF because those measurements are not supplied.

Compares EVERY physical 7-wave PSF with native, 250k-ray DeepLens Huygens on
same optical model, fixed film, physical 5um pixels, exact common 550nm chief.
A successful import or same-code physics comparison is NEVER called an
empirical production-lens validation.
"""
from __future__ import annotations
import sys,os,json,math,time,argparse,types,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from validation.qfilm_deeplens_same_prescription import BK7,mtf_from_psf
from deeplens import GeoLens
from evaluator.qfilm02 import Config,_surfaces_at,trace_pupil,WAVES,WEIGHTS
from evaluator.qfilm04 import evaluate as qfilm04
from evaluator.qfilm03_physical_mode import PhysicalHuygensConfig
from validation.qfilm04_real_prescription_import import (
    CAMERA_SAMPLES,import_geolens_lens,lens_metadata,extract_full_precision_lens)
from validation.qfilm04_secondary_patent_sources import ZEMAX_SAMPLES,load_remote_zmx
F=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)
R=math.hypot(36.,24.)/2
ALL=[("DeepLens_camera",x) for x in CAMERA_SAMPLES]+[("PatentZMX",x) for x in ZEMAX_SAMPLES]
WEIGHT=np.asarray(WEIGHTS,dtype=float)/sum(WEIGHTS)
ROOT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm04_18_real_published"
ROOT.mkdir(parents=True,exist_ok=True)
DEEP=Path("/var/lib/cots-lamcts/deeplens_upstream/datasets/lenses/camera")
def open_case(kind,name):
 if kind=="DeepLens_camera":
  path=DEEP/(name+".json");raw=json.loads(path.read_text())
  provenance={"source":"pinned-vccimaging/DeepLens","version":"911bb9d334fce7e20a4407c4506ac683b52ff246",
    "url":f"https://github.com/vccimaging/DeepLens/blob/911bb9d334fce7e20a4407c4506ac683b52ff246/datasets/lenses/camera/{name}.json",
    "metadata":lens_metadata(raw,name),
    "manufactured_prescription_unverified":True}
 else:
  path,zsource=load_remote_zmx(name,ROOT/"zmx_cache")
  provenance=zsource
  raw=None
 lens=GeoLens(filename=str(path),device="cuda:0",dtype=torch.float64)
 if raw is None:raw=extract_full_precision_lens(lens,provenance["url"])
 p=import_geolens_lens(lens,raw,name,wavelengths=WAVES)
 return lens,p,provenance
def setup_pitch(lens):
 wh=list(map(float,lens.sensor_size))
 if len(wh)!=2 or min(wh)<=0:raise ValueError("invalid reference sensor size")
 resolution=(int(round(wh[0]/.005)),int(round(wh[1]/.005)))
 lens.set_sensor_res(sensor_res=resolution)
 if abs(float(lens.pixel_size)-.005)>1e-6:
  raise ValueError("physical detector pixel pitch mismatch")
 return resolution
def independent_poly(lens,p,h,spp=250000):
 alpha=math.atan(h*R/p["design_spec"]["efl_target_mm"])
 a=torch.tensor([[alpha]],device="cuda:0",dtype=torch.float64)
 stop_r=p["design_spec"]["stop_diameter_mm"]/2
 with torch.inference_mode():
  chief=trace_pupil(_surfaces_at(p["surfaces"],550.),a,
     torch.zeros((1,2),dtype=torch.float64,device="cuda:0"),
     p["stop_z_mm"],stop_r,p["image_z_mm"])
 if not bool(chief[6].all()):raise ValueError("invalid green chief")
 physical_chief_x=float(chief[4].item())
 def fixed_green(self,points_obj,method="chief_ray",ray=None):
  return torch.tensor([[0.,physical_chief_x]],device="cuda:0",dtype=torch.float64)
 old=lens.psf_center;lens.psf_center=types.MethodType(fixed_green,lens)
 pnt=[0.,math.tan(alpha)*float(lens.foclen)/(float(lens.sensor_size[1])/2),-1e6]
 accum=None;edge=[];times=[]
 try:
  for i,w in enumerate(WAVES):
   torch.manual_seed(20261028+int(w)*19+int(h*10)*79)
   t=time.perf_counter()
   with torch.inference_mode():
    psf=lens.psf_huygens(points=pnt,ks=128,wvln=w*.001,
                             spp=spp,recenter=True)
   torch.cuda.synchronize()
   arr=psf.detach().cpu().numpy().astype(np.float64)
   arr/=max(arr.sum(),1e-200)
   if not np.all(np.isfinite(arr)):raise ValueError("nonfinite PSF")
   border=np.ones(arr.shape,bool);border[4:-4,4:-4]=False
   edge.append(float(arr[border].sum()))
   times.append(time.perf_counter()-t)
   if accum is None:accum=WEIGHT[i]*arr
   else:accum+=WEIGHT[i]*arr
 finally:
  lens.psf_center=old
 accum/=accum.sum()
 tx,ty=mtf_from_psf(accum,.005,F)
 return np.asarray(list(tx)+list(ty) if abs(h)<1e-12 else list(ty)+list(tx)),
         {"fixed_green_chief_film_mm":physical_chief_x,"max_4px_border_flux_fraction":max(edge),
          "wavelength_4px_border_flux_fraction":edge,"huygens_seconds":sum(times),
          "reference_quality":"window-risk" if max(edge)>0.01 else "finite-spp-unconverged"}
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--shard",type=int,required=True,
       choices=[0,1,2]);parser.add_argument("--spp",type=int,default=250000)
 args=parser.parse_args()
 torch.set_num_threads(8)
 selected=ALL[args.shard*6:(args.shard+1)*6]
 FILE=ROOT/f"qfilm04_published_18_shard{args.shard}.json"
 report={"schema":"Qfilm04-18-published-optical-examples-v1",
   "shard":args.shard,"backend":"Qfilm0.4-g","source_examples":len(selected),
   "wavelength_nm":list(WAVES),"weights":WEIGHT.tolist(),
   "field_h":[0.,.5,1.],"physical_frame_mm":[36.,24.],
   "PSF_window_mm":.64,"reference_spp_per_wavelength":args.spp,
   "evidence":"two-independently-implemented-scalar-solvers",
   "manufactured_lens_empirical_validation":False,
   "certified":False,"training_eligible":False,"cases":[]}
 for kind,name in selected:
  c={"name":name,"source_kind":kind,"fields":[]};report["cases"].append(c)
  start=time.perf_counter()
  try:
   lens,p,provenance=open_case(kind,name)
   c["provenance"]=provenance
   c["parsed_surfaces"]=len(p["surfaces"])
   c["reference_pixel_resolution"]=setup_pitch(lens)
   c["physical_efl_mm"]=float(lens.foclen)
   c["physical_stop_z_mm"]=p["stop_z_mm"]
   c["physical_film_z_mm"]=p["image_z_mm"]
   cfg=Config(fields=3,pupil_n=256,fft_n=1024,max_fields_per_chunk=1,
    wavelengths_nm=WAVES,spectral_weights=WEIGHTS,frame_width_mm=36.,
    frame_height_mm=24.,target_efl_mm=p["design_spec"]["efl_target_mm"])
   with torch.inference_mode():
    qr=qfilm04(p,cfg,PhysicalHuygensConfig(pupil_n=256,detector_n=128,
           detector_pitch_um=5.,detector_block=384),backend="g")
   c["solver_ok"]=True
   c["quality_valid"]=qr["quality"]["valid"]
   c["final_J"]=qr["quality"]["J_task"]
   c["Qfilm_runtime_seconds"]=qr["timings"]["total_seconds"]
   for i,h in enumerate((0.,.5,1.)):
    try:
     gt,meta=independent_poly(lens,p,h,spp=args.spp)
     q=np.asarray(qr["Q_image"][i,10:30])
     delta=np.abs(gt-q)
     row={"h":h,"qfilm_MTF_T":q[:10].tolist(),"qfilm_MTF_S":q[10:].tolist(),
      "reference_MTF_T":gt[:10].tolist(),"reference_MTF_S":gt[10:].tolist(),
      "mean_abs_mtf_difference":float(delta.mean()),
      "max_abs_mtf_difference":float(delta.max()),
      "p95_abs_mtf_difference":float(np.quantile(delta,.95)),**meta}
     c["fields"].append(row)
     print("BENCH_REAL_FIELD",name,h,"max",row["max_abs_mtf_difference"],
          "mean",row["mean_abs_mtf_difference"],"border",meta["max_4px_border_flux_fraction"],
          flush=True)
    except Exception as e:
     c["fields"].append({"h":h,"error":repr(e)})
     print("BENCH_REAL_FIELD_ERROR",name,h,repr(e),flush=True)
    FILE.write_text(json.dumps(report,indent=2)+"\n")
   valid=[x for x in c["fields"] if "max_abs_mtf_difference" in x]
   c["n_fields_compared"]=len(valid)
   c["max_abs_mtf_difference"]=max((x["max_abs_mtf_difference"] for x in valid),default=None)
   c["mean_abs_mtf_difference"]=np.mean([x["mean_abs_mtf_difference"] for x in valid]).item() if valid else None
  except Exception as e:
   c["solver_ok"]=False;c["error"]=repr(e);c["traceback"]=traceback.format_exc()[-2500:]
  c["elapsed_seconds"]=time.perf_counter()-start
  print("BENCH_REAL_LENS",name,json.dumps({k:v for k,v in c.items()
   if k not in ("fields","traceback","provenance")}),flush=True)
  FILE.write_text(json.dumps(report,indent=2)+"\n")
 print("BENCH_REAL_SHARD_COMPLETE",json.dumps({
   "shard":args.shard,"n":len(selected),"solver_ok":sum(c.get("solver_ok",False) for c in report["cases"]),
   "with_three_field_reference":sum(c.get("n_fields_compared",0)==3 for c in report["cases"])}),flush=True)
if __name__=="__main__":main()
