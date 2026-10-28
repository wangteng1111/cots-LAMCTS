#!/usr/bin/env python3
"""Signed complex-OTF phase / pupil propagated PSF diagnostic for residual
fixed-film seven-color 5lpmm discrepancy. Does NOT fit any model parameters.
"""
from __future__ import annotations
import sys,os,json,math,types,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from evaluator.qfilm02 import WAVES,WEIGHTS,_surfaces_at,trace_pupil
from evaluator.qfilm03_physical_huygens import Propagation,eval_fixed_film,coherent_otf_axis
from validation.qfilm_deeplens_same_prescription import internal_stop_positive_lens,thin_lens,conversion,BK7
from deeplens import GeoLens
OUT=Path(os.environ.get("COTS_JOB_ARTIFACT_DIR","artifacts"))/"qfilm03_signed_otf"
OUT.mkdir(parents=True,exist_ok=True);FILE=OUT/"signed_otf_center_ablation.json"
W=np.asarray(WEIGHTS,dtype=float);W/=W.sum();F=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)
def optof(psf):
    p=np.asarray(psf,dtype=np.float64);p/=p.sum();n=p.shape[0];mid=n//2
    otf=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(p)));otf/=otf[mid,mid]
    f=np.fft.fftshift(np.fft.fftfreq(n,.005))
    def ival(v,fr):
        return complex(np.interp(fr,f[mid:],v[mid:].real),np.interp(fr,f[mid:],v[mid:].imag))
    return np.array([[ival(otf[mid],fr),ival(otf[:,mid],fr)] for fr in F])
def main():
    base=thin_lens();base["surfaces"][0]["sellmeier"]=BK7
    report={"schema":"qfilm03-signed-OTF-center-color-v1","cases":{}}
    for name,p in (("singlet",base),("internal_stop_asphere",internal_stop_positive_lens())):
        c={"wavelengths":[]};report["cases"][name]=c
        deep,_=conversion(p,sensor_size=48.,sensor_res=9600,material_mode="bk7")
        fp=OUT/(name+".json");fp.write_text(json.dumps(deep))
        lens=GeoLens(filename=str(fp),device="cuda:0",dtype=torch.float64)
        angle=torch.tensor([[0.]],dtype=torch.float64,device="cuda:0")
        with torch.inference_mode():
            ch=trace_pupil(_surfaces_at(p["surfaces"],550.),
              angle,torch.zeros((1,2),dtype=torch.float64,device="cuda:0"),
              p["stop_z_mm"],p["design_spec"]["stop_diameter_mm"]/2,p["image_z_mm"])
        center=float(ch[4].item())
        def stable(self,points_obj,method="chief_ray",ray=None):
            return torch.tensor([[0.,center]],device="cuda:0",dtype=torch.float64)
        old=lens.psf_center;lens.psf_center=types.MethodType(stable,lens)
        sumd=np.zeros((len(F),2),dtype=np.complex128)
        sumq=np.zeros((len(F),2),dtype=np.complex128)
        for j,lam in enumerate(WAVES):
            torch.manual_seed(20261028+int(lam)*19)
            with torch.inference_mode():
                psf=lens.psf_huygens(points=[0.,0.,-1e6],
                    ks=128,wvln=lam*.001,spp=1000000,recenter=True)
                phy=eval_fixed_film(p,lam,0.,Propagation(pupil_n=256,
                    pixel_n=128,pixel_pitch_um=5.,pixel_block=384,
                    weight_mode="sqrt_exit_jac"))
            d=psf.detach().cpu().numpy()
            q=phy["psf"].detach().cpu().numpy()
            dop=optof(d);qop=coherent_otf_axis(phy["psf"],.005,F)
            sumd+=W[j]*dop;sumq+=W[j]*qop
            k=F.index(5.)
            rec={"wavelength_nm":lam,"weight":float(W[j]),
                "physical_chief_film_mm":center,
                "Deep_OTF5_real":float(dop[k,0].real),
                "Qfilm_Huygens_OTF5_real":float(qop[k,0].real),
                "Deep_OTF5_imag":float(dop[k,0].imag),
                "Qfilm_Huygens_OTF5_imag":float(qop[k,0].imag),
                "Deep_MTF5":float(abs(dop[k,0])),
                "Qfilm_Huygens_MTF5":float(abs(qop[k,0])),
                "max_mono_MTF_error":float(np.max(np.abs(np.abs(dop)-np.abs(qop)))),
                "PSF_L1":float(np.abs(d-q).sum()),
                "PSF_peak_Deep":float(d.max()),"PSF_peak_Q":float(q.max())}
            c["wavelengths"].append(rec)
            print("SIGNED_COLOR",name,lam,rec,flush=True)
            FILE.write_text(json.dumps(report,indent=2)+"\n")
        lens.psf_center=old
        err=np.abs(np.abs(sumd)-np.abs(sumq))
        c["summary"]={"Deep_OTF5":float(sumd[0,0].real),
            "Qfilm_Huygens_OTF5":float(sumq[0,0].real),
            "Deep_poly_MTF5":float(abs(sumd[0,0])),
            "Qfilm_Huygens_poly_MTF5":float(abs(sumq[0,0])),
            "maximum_polychromatic_abs_MTF_disagreement":float(err.max()),
            "mean_polychromatic_abs_MTF_disagreement":float(err.mean())}
        FILE.write_text(json.dumps(report,indent=2)+"\n")
        print("SIGNED_RESULT",name,c["summary"],flush=True)
    print("SIGNED_COMPLETE",len(report["cases"]),flush=True)
if __name__=="__main__":main()
