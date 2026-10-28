#!/usr/bin/env python3
"""Full-frame seven-color intensity-PSF integration with deterministic physical
green chief at every wavelength, compared with legacy Qfilm512 and new direct
physical Huygens Final-J. Fixed mechanical film for all wavelengths.
"""
from __future__ import annotations
import sys,os,json,math,types,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from evaluator.qfilm02 import Config,WAVES,WEIGHTS,evaluate as qeval,_surfaces_at,trace_pupil
from evaluator.qfilm03 import evaluate_physical_huygens
from evaluator.qfilm03_physical_mode import PhysicalHuygensConfig
from validation.qfilm_deeplens_same_prescription import (
    thin_lens,internal_stop_positive_lens,conversion,mtf_from_psf,BK7)
from deeplens import GeoLens
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_poly_fixed_chief"
OUT.mkdir(parents=True,exist_ok=True)
FILE=OUT/"polychromatic_huygens_36x24_fixed_chief.json"
F=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)
weights=np.asarray(WEIGHTS,dtype=float)
weights=weights/weights.sum()
HS=(0.,.5,1.)
def main():
    p=thin_lens();p["surfaces"][0]["sellmeier"]=BK7
    result={"schema":"Qfilm03-sevencolor-deterministic-chieffilm-v1",
      "frame_mm":[36.,24.],"sensor_pitch_mm":.005,"psf_n":128,
      "spectral_weights":weights.tolist(),"wavelength_nm":list(WAVES),
      "reference_chief_definition":"same physical green(550nm) chief ray on fixed film",
      "deep_spp_per_wave":1000000,"photometrically_certified":False,
      "cases":{}}
    for name,p in (("singlet",p),("internal_stop_asphere",internal_stop_positive_lens())):
        c={"fields":[]};result["cases"][name]=c
        try:
            data,_=conversion(p,sensor_size=48.,sensor_res=9600,material_mode="bk7")
            fp=OUT/(name+".json")
            fp.write_text(json.dumps(data))
            lens=GeoLens(filename=str(fp),device="cuda:0",dtype=torch.float64)
            cfg=Config(fields=3,pupil_n=512,fft_n=2048,max_fields_per_chunk=1,
                wavelengths_nm=WAVES,spectral_weights=WEIGHTS,
                frame_width_mm=36.,frame_height_mm=24.,
                target_efl_mm=p["design_spec"]["efl_target_mm"])
            with torch.inference_mode():
                baseline=qeval(p,cfg,device="cuda:0")
                physical=evaluate_physical_huygens(p,cfg,device="cuda:0",
                    physical=PhysicalHuygensConfig(
                        pupil_n=128,detector_n=128,detector_pitch_um=5.,
                        detector_block=384,amplitude_mode="sqrt_exit_jac"))
            c["fft_final_J"]=baseline["quality"]["J_task"]
            c["physical_final_J"]=physical["quality"]["J_task"]
            c["physical_training_eligible"]=physical["training_eligible"]
            for idx,h in enumerate(HS):
                angle=math.atan(h*math.hypot(36.,24.)/2/p["design_spec"]["efl_target_mm"])
                ang=torch.tensor([[angle]],device="cuda:0",dtype=torch.float64)
                with torch.inference_mode():
                    chief=trace_pupil(_surfaces_at(p["surfaces"],550.),ang,
                        torch.zeros((1,2),device="cuda:0",dtype=torch.float64),
                        p["stop_z_mm"],p["design_spec"]["stop_diameter_mm"]/2,
                        p["image_z_mm"])
                if not bool(chief[6].all()):raise ValueError("invalid physical chief ray")
                chief_mm=float(chief[4].item())
                def stable_green_chief(self,points_obj,method="chief_ray",ray=None):
                    return torch.tensor([[0.,chief_mm]],device="cuda:0",dtype=torch.float64)
                original=lens.psf_center
                lens.psf_center=types.MethodType(stable_green_chief,lens)
                rows=[]
                point=[0.,math.tan(angle)*float(lens.foclen)/(float(lens.sensor_size[1])/2),-1e6]
                try:
                    for wi,w in enumerate(WAVES):
                        torch.manual_seed(20261028+int(w)*19+int(h*10)*79)
                        t=time.perf_counter()
                        with torch.inference_mode():
                            psf=lens.psf_huygens(points=point,ks=128,wvln=w*.001,
                                spp=1000000,recenter=True)
                        torch.cuda.synchronize()
                        arr=psf.detach().cpu().numpy().astype(np.float64)
                        arr/=arr.sum()
                        rows.append((weights[wi]*arr,{"wavelength_nm":w,
                            "elapsed_s":time.perf_counter()-t,"peak":float(arr.max())}))
                finally:
                    lens.psf_center=original
                mix=np.sum([x[0] for x in rows],axis=0)
                mix/=mix.sum()
                tx,ty=mtf_from_psf(mix,.005,F)
                gt=np.asarray(tx+ty,dtype=float) if h==0 else np.asarray(ty+tx,dtype=float)
                q=baseline["Q_image"][idx,10:30]
                ph=physical["Q_image"][idx,10:30]
                e0=np.abs(gt-q);e1=np.abs(gt-ph)
                entry={"h":h,"chief_550_image_x_mm":chief_mm,
                    "deeplens_polychromatic_mtf":gt.tolist(),
                    "baseline_fft_mtf":q.tolist(),"physical_huygens_mtf":ph.tolist(),
                    "legacy_max_abs":float(e0.max()),"legacy_mean_abs":float(e0.mean()),
                    "new_max_abs":float(e1.max()),"new_mean_abs":float(e1.mean()),
                    "max_abs_improvement":float(e0.max()-e1.max()),
                    "per_lambda":list(x[1] for x in rows)}
                c["fields"].append(entry)
                print("FIXED_GREEN_POLY",name,h,
                    "FFTmax",entry["legacy_max_abs"],
                    "NEWmax",entry["new_max_abs"],
                    "FFTmean",entry["legacy_mean_abs"],
                    "NEWmean",entry["new_mean_abs"],flush=True)
                FILE.write_text(json.dumps(result,indent=2)+"\n")
        except Exception as ex:
            c["error"]=repr(ex);c["traceback"]=traceback.format_exc()[-5000:]
            print("FIXED_GREEN_FAIL",name,repr(ex),flush=True)
        FILE.write_text(json.dumps(result,indent=2)+"\n")
    assert all(len(c["fields"])==3 and "error" not in c for c in result["cases"].values()),result
    print("FIXED_GREEN_POLY_PASS",json.dumps({k:{"fields":len(c["fields"]),
          "fft":c["fft_final_J"],"physical":c["physical_final_J"]}
          for k,c in result["cases"].items()}),flush=True)
if __name__=="__main__":main()
