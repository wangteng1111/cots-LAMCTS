"""Arbitrary-prescription adapter for the authoritative direct-OPD Q4096 engine.

Reuses the validated Q4096 ray/OPD/OTF machinery but supplies wavelength-specific
surfaces from an Alpha Lense variable-topology prescription. It does not map a
real lens to Mandler catalog IDs.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from alpha_lense.stage_pipeline_v01 import Prescription
from core.mandler_benchmark_core import cauchy_from_ne_ve,Surf,cardinal,front_matrix
from evaluator import market_final_direct_q4096_v4 as E

@dataclass
class PrescriptionOptics:
    p:Prescription
    def surfaces(self,lam:float):
        z=0.;n1=1.;out=[]
        for s in self.p.surfaces:
            n2=1. if s.n_after<=1.000001 else cauchy_from_ne_ve(s.n_after,max(s.v_after,1e-6),lam)
            out.append(Surf(z,s.radius,n1,n2));z+=s.thickness;n1=n2
        return out
    def stop_z(self):
        return sum(s.thickness for s in self.p.surfaces[:max(0,min(self.p.stop_after,len(self.p.surfaces)))])

def evaluate_prescription(p:Prescription,qmc_samples:int=4096)->dict:
    if qmc_samples!=4096:raise ValueError('authoritative labels require Q4096')
    O=PrescriptionOptics(p);stop=O.stop_z();se=O.surfaces(E.WAVES[1]);efl,focus=cardinal(se);fm=front_matrix(se,stop)
    if fm is None or not math.isfinite(efl) or abs(float(fm[0,0]))<1e-12:return {'J':1e9,'merit_J':1e9,'config_hash':E.CONFIG_HASH,'error':'nonfinite_cardinal'}
    # The v4 tracing routines accept arbitrary Surf lists. We reproduce its merit
    # loop with the prescription stop translated to z=0 by shifting surfaces.
    shifted=[]
    for lam in E.WAVES:
        shifted.append([Surf(s.z-stop,s.R,s.n1,s.n2) for s in O.surfaces(lam)])
    se=shifted[1];efl,focus=cardinal(se);fm=front_matrix(se,0.)
    if fm is None or abs(float(fm[0,0]))<1e-12:return {'J':1e9,'merit_J':1e9,'config_hash':E.CONFIG_HASH,'error':'bad_stop'}
    # Match entrance pupil to requested f-number when known; otherwise retain v4 reference stop.
    target_fno=p.design_spec.get('max_f_number') if p.design_spec else None
    stop_r=E.STOP_R if not target_fno else abs(float(fm[0,0]))*(abs(efl)/float(target_fno))/2.
    old=E.STOP_R
    try:
        E.STOP_R=stop_r
        # Temporarily inject a tiny make_surfaces-compatible object so the authoritative
        # scalar implementation remains exactly one source of merit logic.
        old_make=E.M.make_surfaces;old_card=E.M.cardinal;old_front=E.M.front_matrix
        E.M.make_surfaces=lambda design,lam,**kw: [Surf(s.z-stop,s.R,s.n1,s.n2) for s in O.surfaces(lam)]
        E.M.cardinal=cardinal;E.M.front_matrix=front_matrix
        r=E._evaluate_surfaces(None,None,qmc_samples)
    finally:
        E.STOP_R=old;E.M.make_surfaces=old_make;E.M.cardinal=old_card;E.M.front_matrix=old_front
    r['J']=r['merit_J'];r['prescription_adapter']='alpha_lense_v01';return r
