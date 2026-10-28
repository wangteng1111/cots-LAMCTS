#!/usr/bin/env python3
"""Full-frame Qfilm/DeepLens red & nominal Huygens convergence, unmodified physics.

For each challenging case, compare 1M and 4M rays, 128px (0.64mm)
and 192px (0.96mm) PSF windows at fixed 5um physical detector pitch.
A strict batch-only Huygens override (10k -> 4096 rays) is validated
against pristine upstream; NO wave equation is changed.
"""
from __future__ import annotations
import sys,os,json,time,math,copy,traceback,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from validation.qfilm_deeplens_same_prescription import (thin_lens,internal_stop_positive_lens,conversion,mtf_from_psf,BK7)
from validation.qfilm03_huygens_safechunk import patch_huygens,parity
from evaluator.qfilm02 import evaluate as qev,Config
from deeplens import GeoLens
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_fullframe_reference_convergence"
OUT.mkdir(parents=True,exist_ok=True)
PATH=OUT/"red_reference_ray_window.json"
F=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)
CASES=[("singlet",670.,0.),("singlet",670.,.1),("singlet",550.,0.),
       ("internal_stop_asphere",670.,0.),("internal_stop_asphere",670.,.1),
       ("internal_stop_asphere",670.,.2),("internal_stop_asphere",550.,0.)]
def run_point(lens,p,w,h,mode_spp,ks,seed,depth_mm=1.e6):
    angle=math.atan(h*math.hypot(36.,24.)/2/p["design_spec"]["efl_target_mm"])
    point=[0.,math.tan(angle)*float(lens.foclen)/(float(lens.sensor_size[1])/2),-float(depth_mm)]
    torch.manual_seed(seed)
    start=time.perf_counter()
    with torch.inference_mode():psf=lens.psf_huygens(points=point,ks=ks,wvln=w*.001,spp=mode_spp,recenter=True)
    torch.cuda.synchronize()
    px,py=mtf_from_psf(psf.detach().cpu().numpy(),float(lens.pixel_size),F)
    t,s=(px,py) if abs(h)<1e-12 else (py,px)
    return {"spp":mode_spp,"ks":ks,"object_distance_mm":float(depth_mm),"physical_window_mm":ks*float(lens.pixel_size),
            "elapsed_s":time.perf_counter()-start,"PSF_sum":float(psf.sum()),
            "MTF":list(t)+list(s)}
def main():
    old,new,meta=patch_huygens(batch=4096)
    parity_record=parity(old,new)
    assert parity_record["PSF_max_abs"]<1e-8,parity_record
    GeoLens.psf_huygens=new
    p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
    lenses={"singlet":p,"internal_stop_asphere":internal_stop_positive_lens()}
    d={"schema":"qfilm03-36x24-reference-convergence-v2",
       "deep_revision":"911bb9d334fce7e20a4407c4506ac683b52ff246",
       "optical_physics_changed":False,"batch_parity":parity_record,"patch":meta,
       "sensor_size_mm":48.,"sensor_n":9600,"physical_pitch_mm":.005,
       "image_frame_mm":[36.,24.],"MTF_frequency_lpmm":F,"cases":[]}
    for name,p in lenses.items():
        data,_=conversion(p,sensor_size=48.,sensor_res=9600,material_mode="bk7")
        f=OUT/(name+".json");f.write_text(json.dumps(data,indent=2))
        lens=GeoLens(filename=str(f),device="cuda:0",dtype=torch.float64)
        for n,w,h in CASES:
            if n!=name:continue
            record={"name":name,"lambda_nm":w,"field_h":h,"conditions":[]}
            d["cases"].append(record)
            cfg=Config(fields=11,pupil_n=512,fft_n=2048,max_fields_per_chunk=2,
                wavelengths_nm=(w,),spectral_weights=(1.,),
                frame_width_mm=36.,frame_height_mm=24.,
                target_efl_mm=p["design_spec"]["efl_target_mm"])
            with torch.inference_mode():q=qev(p,cfg,device="cuda:0")
            qa=q["Q_image"][int(round(h*10)),10:30].tolist()
            record["Qfilm512_MTF"]=qa
            seed=20261028+int(w)*73+int(round(h*10))*101
            configurations=[(1000000,128),(1000000,192),(4000000,128)]
            if w==670. and h in (0.,.1):configurations.append((4000000,192))
            for spp,ks in configurations:
                try:
                    r=run_point(lens,p,w,h,spp,ks,seed)
                    err=np.abs(np.asarray(r["MTF"])-np.asarray(qa))
                    r["Qfilm_reference_MTF_max_abs"]=float(err.max())
                    r["Qfilm_reference_MTF_mean_abs"]=float(err.mean())
                    record["conditions"].append(r)
                    print("HCONV",name,w,h,"spp",spp,"ks",ks,
                        "diff",r["Qfilm_reference_MTF_max_abs"],"time",round(r["elapsed_s"],2),flush=True)
                except Exception as e:
                    record["conditions"].append({"spp":spp,"ks":ks,"error":repr(e)})
                    print("HCONV_FAIL",name,w,h,spp,ks,repr(e),flush=True)
                PATH.write_text(json.dumps(d,indent=2)+"\n")
            # Far-field normalization check: DeepLens internally converts the
            # normalized input to actual object-space coordinates via -depth/focal.
            # A 1-km source is not a mathematical plane wave; quantify this error.
            if w==670. and (name=="singlet" and h==0. or name=="internal_stop_asphere" and h==.1):
                for distance in (1.e7,1.e8):
                    try:
                        r=run_point(lens,p,w,h,1000000,128,seed,depth_mm=distance)
                        r["Qfilm_reference_MTF_max_abs"]=float(np.max(np.abs(np.asarray(r["MTF"])-np.asarray(qa))))
                        record["conditions"].append(r)
                        print("DISTANCE",name,w,h,distance,r["Qfilm_reference_MTF_max_abs"],flush=True)
                    except Exception as exc:
                        record["conditions"].append({"spp":1000000,"ks":128,"object_distance_mm":distance,"error":repr(exc)})
                    PATH.write_text(json.dumps(d,indent=2)+"\n")
            arr=[x for x in record["conditions"] if "MTF" in x]
            def get(spp,ks):
                return next((np.asarray(x["MTF"]) for x in arr if x["spp"]==spp and x["ks"]==ks),None)
            a=get(1000000,128);b=get(1000000,192);c=get(4000000,128);e=get(4000000,192)
            record["convergence"]={
                "window_128_to_192_1M_max_MTF":float(np.max(np.abs(a-b))) if a is not None and b is not None else None,
                "rays_1M_to_4M_128_max_MTF":float(np.max(np.abs(a-c))) if a is not None and c is not None else None,
                "window_128_to_192_4M_max_MTF":float(np.max(np.abs(c-e))) if c is not None and e is not None else None,
                "rays_1M_to_4M_192_max_MTF":float(np.max(np.abs(b-e))) if b is not None and e is not None else None}
            PATH.write_text(json.dumps(d,indent=2)+"\n")
    print("REFERENCE_COMPLETE",json.dumps([{"n":x["name"],"w":x["lambda_nm"],"h":x["field_h"],"convergence":x.get("convergence")} for x in d["cases"]]),flush=True)
if __name__=="__main__":main()
