#!/usr/bin/env python3
"""Ideal circular-pupil Fraunhofer quadrature control, NOT a real lens run.

At y=0, sum the 2D circular grid by columns exactly. Uniform pupil quadrature
has an intensity repetition period lambda*Fnumber*N in image coordinates.
This isolates a plausible mechanism; it does not prove that the physical
nonuniform exit-plane Huygens results have this exact period.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import j1

def intensity(n,x_um,wavelength_um,fnumber):
    u=(np.arange(n)+.5)/n-.5
    counts=(u[:,None]**2+u[None,:]**2<=.25).sum(axis=0)
    phase=-2j*np.pi*np.asarray(x_um)[:,None]*u[None,:]/(wavelength_um*fnumber)
    return abs(np.exp(phase)@counts/counts.sum())**2

def airy(x_um,wavelength_um,fnumber):
    t=np.pi*np.abs(np.asarray(x_um))/(wavelength_um*fnumber)
    a=np.ones_like(t);np.divide(2*j1(t),t,out=a,where=t!=0)
    return a*a

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    report={"schema":"Qfilm04-ideal-pupil-quadrature-alias-control-v1",
        "real_lens_validation":False,"qfilm_physical_kernel_modified":False,
        "interpretation":"controlled mechanism only; real exit mapping and aberrations require independent convergence",
        "cases":[]}
    for name,wavelength_um,fnumber in (("EF40-associated-fnumber",.43,2.8090031745543445),
                                      ("RF50-associated-fnumber",.67,1.8415697057496243)):
        period=wavelength_um*fnumber*256
        x=np.array([0.,period])
        exact=airy(x,wavelength_um,fnumber)
        row={"name":name,"wavelength_um":wavelength_um,"fnumber":fnumber,
            "N256_period_um":period,"comparison_x_um":x.tolist(),
            "analytic_airy_intensity_relative_to_peak":exact.tolist(),"grid_results":[]}
        for n in (256,512,1024):
            values=intensity(n,x,wavelength_um,fnumber)
            row["grid_results"].append({"pupil_n":n,"intensity_relative_to_onaxis":values.tolist()})
            if n==256: assert abs(values[1]-1)<1e-12
        assert exact[1]<1e-7
        report["cases"].append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
