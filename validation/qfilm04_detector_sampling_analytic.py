#!/usr/bin/env python3
"""Airy-PSF sampling control, independent of any photographic prescription.

Known circular-pupil MTF vs normalized point samples, same 0.64mm support.
This tests image-grid aliasing, not Qfilm ray-trace accuracy or lens tolerances.
No claim that a real aberrated photographic lens equals an Airy PSF.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import j1

F=np.array([5.,10.,15.,20.,30.,40.,50.,60.,70.,80.])

def sampled(wave_nm,fnumber,pitch_um):
    n=int(round(640./pitch_um));pitch=pitch_um*.001
    x=(np.arange(n)-(n-1)*.5)*pitch
    radius=np.hypot(x[:,None],x[None,:])
    u=np.pi*radius/(wave_nm*1e-6*fnumber)
    psf=np.ones_like(u)
    np.divide(2*j1(u),u,out=psf,where=u!=0)
    psf*=psf
    projection=psf.sum(axis=0);projection/=projection.sum()
    return np.abs(np.exp(-2j*np.pi*F[:,None]*x[None,:])@projection)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--pitch-um",type=float,default=.25)
    ap.add_argument("--compare-legacy",action="store_true",help="also reproduce 5um aliasing")
    args=ap.parse_args()
    if not np.isfinite(args.pitch_um) or args.pitch_um<=0:
        ap.error("pitch must be finite and positive")
    out={"schema":"Qfilm04-analytic-Airy-image-sampling-control-v2",
         "detector_pitch_um":args.pitch_um,"detector_n":int(round(640./args.pitch_um)),
         "frequency_lpmm":F.tolist(),"physical_support_mm":.64,"cases":[]}
    for w in (550.,670.):
        for fno in (1.4,1.8,2.8,5.6):
            u=F*(w*1e-6*fno)
            exact=2/np.pi*(np.arccos(u)-u*np.sqrt(1-u*u))
            fine=sampled(w,fno,args.pitch_um)
            c={"wavelength_nm":w,"fnumber":fno,"analytic_MTF":exact.tolist(),
               "point_sampled_MTF":fine.tolist(),
               "max_abs_error":float(np.max(np.abs(exact-fine)))}
            if args.compare_legacy:
                coarse=sampled(w,fno,5.)
                c.update(point_sampled_MTF_5um=coarse.tolist(),
                         max_abs_error_5um=float(np.max(np.abs(exact-coarse))))
            out["cases"].append(c)
            print("AIRY_SAMPLING",w,fno,args.pitch_um,c["max_abs_error"],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+"\n")
    assert max(c["max_abs_error"] for c in out["cases"])<.01

if __name__=="__main__":main()
