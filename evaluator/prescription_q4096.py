"""Arbitrary-prescription authoritative Q4096 adapter using the CUDA v4 backend."""
from __future__ import annotations
import math,os
from dataclasses import dataclass
from alpha_lense.stage_pipeline_v01 import Prescription
from evaluator.optical_primitives import cauchy_from_ne_ve,Surf,cardinal,front_matrix
os.environ['ALPHA_LENSE_PRESCRIPTION_CORE']='1'
from evaluator import market_final_direct_q4096_v4 as E
from evaluator import market_final_direct_q4096_v4_cuda as CUDA

@dataclass
class PrescriptionOptics:
    p:Prescription
    def surfaces(self,lam:float):
        z=0.;n1=1.;out=[]
        for s in self.p.surfaces:
            n2=1. if s.n_after<=1.000001 else cauchy_from_ne_ve(s.n_after,max(s.v_after,1e-6),lam)
            out.append(Surf(z,s.radius,n1,n2));z+=s.thickness;n1=n2
        return out
    def stop_z(self):return sum(s.thickness for s in self.p.surfaces[:max(0,min(self.p.stop_after,len(self.p.surfaces)))])

def evaluate_prescription(p:Prescription,qmc_samples:int=4096,device:int=0)->dict:
    if qmc_samples!=4096:raise ValueError('authoritative labels require Q4096')
    CUDA.set_cuda_device(device);O=PrescriptionOptics(p);stop=O.stop_z();se=O.surfaces(E.WAVES[1]);efl,_=cardinal(se);fm=front_matrix(se,stop)
    if fm is None or not math.isfinite(efl) or abs(float(fm[0,0]))<1e-12:return {'J':1e9,'merit_J':1e9,'config_hash':E.CONFIG_HASH,'error':'nonfinite_cardinal','backend':CUDA.BACKEND}
    shifted=[Surf(s.z-stop,s.R,s.n1,s.n2) for s in O.surfaces(E.WAVES[1])];efl,_=cardinal(shifted);fm=front_matrix(shifted,0.)
    if fm is None or abs(float(fm[0,0]))<1e-12:return {'J':1e9,'merit_J':1e9,'config_hash':E.CONFIG_HASH,'error':'bad_stop','backend':CUDA.BACKEND}
    target_fno=p.design_spec.get('max_f_number') if p.design_spec else None;stop_r=E.STOP_R if not target_fno else abs(float(fm[0,0]))*(abs(efl)/float(target_fno))/2.
    old=E.STOP_R;old_make=E.M.make_surfaces;old_card=E.M.cardinal;old_front=E.M.front_matrix
    try:
        E.STOP_R=stop_r;E.M.make_surfaces=lambda design,lam,**kw:[Surf(s.z-stop,s.R,s.n1,s.n2) for s in O.surfaces(lam)];E.M.cardinal=cardinal;E.M.front_matrix=front_matrix
        r=E._evaluate_surfaces(None,None,qmc_samples);CUDA.synchronize()
    finally:E.STOP_R=old;E.M.make_surfaces=old_make;E.M.cardinal=old_card;E.M.front_matrix=old_front
    r['J']=r['merit_J'];r['prescription_adapter']='alpha_lense_v02';r['backend']=CUDA.BACKEND;r['device']=device;return r
