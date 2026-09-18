"""Arbitrary-prescription authoritative Q4096 adapter using the CUDA v4 backend.

The legacy Mandler evaluator remains unchanged.  This adapter configures the
same Q4096 physics per prescription/DesignSpec inside one process.  It is
intended to be called serially within a GPU worker process.
"""
from __future__ import annotations
import hashlib,json,math,os
from dataclasses import dataclass
from alpha_lense.stage_pipeline_v01 import Prescription
from evaluator.optical_primitives import cauchy_from_ne_ve,Surf,cardinal,front_matrix
os.environ['ALPHA_LENSE_PRESCRIPTION_CORE']='1'
from evaluator import market_final_direct_q4096_v4 as E
from evaluator import market_final_direct_q4096_v4_cuda as CUDA

ADAPTER_VERSION="alpha_lense_prescription_q4096_v3_designspec"

@dataclass
class PrescriptionOptics:
    p:Prescription
    def surfaces(self,lam:float):
        z=0.;n1=1.;out=[]
        for s in self.p.surfaces:
            n2=1. if s.n_after<=1.000001 else cauchy_from_ne_ve(s.n_after,max(s.v_after,1e-6),lam)
            out.append(Surf(z,s.radius,n1,n2,float(getattr(s,'conic',0.0)),tuple(getattr(s,'asphere',()) or ())))
            z+=s.thickness;n1=n2
        return out
    def stop_z(self):
        z=getattr(self.p,'stop_z_mm',None)
        if z is not None and math.isfinite(float(z)):return float(z)
        return sum(s.thickness for s in self.p.surfaces[:max(0,min(self.p.stop_after,len(self.p.surfaces)))])
    def image_z(self):
        z=getattr(self.p,'image_z_mm',None)
        if z is not None and math.isfinite(float(z)):return float(z)
        ss=self.surfaces(E.WAVES[1]);_,focus=cardinal(ss)
        return float(focus)

def _finite(x):
    try:return math.isfinite(float(x))
    except Exception:return False

def _runtime_config(p:Prescription,O:PrescriptionOptics,shifted,efl,fm):
    spec=p.design_spec or {}
    stop=O.stop_z();image=O.image_z();film=float(image-stop)
    target_efl=spec.get('efl_target_mm')
    if not _finite(target_efl):target_efl=float(efl)
    target_fno=spec.get('max_f_number')
    stop_diam=spec.get('stop_diameter_mm')
    if _finite(stop_diam) and float(stop_diam)>0:
        stop_r=float(stop_diam)/2.0
    elif _finite(target_fno) and float(target_fno)>0:
        stop_r=abs(float(fm[0,0]))*(abs(float(efl))/float(target_fno))/2.0
    else:
        raise ValueError('prescription has neither explicit stop diameter nor F-number')
    actual_fno=abs(float(efl))/(2.0*stop_r/max(abs(float(fm[0,0])),1e-12))
    if not _finite(target_fno) or float(target_fno)<=0:target_fno=actual_fno
    max_field=spec.get('max_field_deg')
    if not _finite(max_field):
        ic=spec.get('image_circle_mm')
        if _finite(ic) and abs(float(target_efl))>1e-9:
            max_field=math.degrees(math.atan((float(ic)/2.0)/abs(float(target_efl))))
        else:max_field=E.MAX_FIELD_DEG
    cfg=dict(E.CONFIG)
    cfg.update({
      'version':'alpha_lense_prescription_q4096_v3',
      'film_z_mm':float(film),'target_efl_mm':float(target_efl),
      'target_fno':float(target_fno),'max_field_deg':float(max_field),
      'stop_radius_mm':float(stop_r),'prescription_adapter':ADAPTER_VERSION,
      'source_config_index':int(getattr(p,'source_config_index',0)),
    })
    h=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return cfg,h,float(film),float(target_efl),float(target_fno),float(max_field),float(stop_r)

def evaluator_signature(p:Prescription)->str:
    O=PrescriptionOptics(p);stop=O.stop_z()
    shifted=[Surf(s.z-stop,s.R,s.n1,s.n2,getattr(s,'conic',0.0),tuple(getattr(s,'asphere',()) or ())) for s in O.surfaces(E.WAVES[1])]
    efl,_=cardinal(shifted);fm=front_matrix(shifted,0.)
    if fm is None or not math.isfinite(efl) or abs(float(fm[0,0]))<1e-12:return 'invalid'
    return _runtime_config(p,O,shifted,efl,fm)[1]

def evaluate_prescription(p:Prescription,qmc_samples:int=4096,device:int=0)->dict:
    if qmc_samples!=4096:raise ValueError('authoritative labels require Q4096')
    CUDA.set_cuda_device(device)
    O=PrescriptionOptics(p);stop=O.stop_z()
    shifted=[Surf(s.z-stop,s.R,s.n1,s.n2,getattr(s,'conic',0.0),tuple(getattr(s,'asphere',()) or ())) for s in O.surfaces(E.WAVES[1])]
    efl,_=cardinal(shifted);fm=front_matrix(shifted,0.)
    if fm is None or not math.isfinite(efl) or abs(float(fm[0,0]))<1e-12:
        return {'J':1e9,'merit_J':1e9,'config_hash':'invalid','error':'nonfinite_cardinal','backend':CUDA.BACKEND,'device':device}
    try:cfg,ch,film,target_efl,target_fno,max_field,stop_r=_runtime_config(p,O,shifted,efl,fm)
    except Exception as exc:
        return {'J':1e9,'merit_J':1e9,'config_hash':'invalid','error':f'bad_design_spec:{exc}','backend':CUDA.BACKEND,'device':device}

    old=(E.STOP_R,E.FILM,E.TARGET_EFL,E.TARGET_FNO,E.MAX_FIELD_DEG,E.M.make_surfaces,E.M.cardinal,E.M.front_matrix)
    try:
        E.STOP_R=stop_r;E.FILM=film;E.TARGET_EFL=target_efl;E.TARGET_FNO=target_fno;E.MAX_FIELD_DEG=max_field
        E.M.make_surfaces=lambda design,lam,**kw:[Surf(s.z-stop,s.R,s.n1,s.n2,getattr(s,'conic',0.0),tuple(getattr(s,'asphere',()) or ())) for s in O.surfaces(lam)]
        E.M.cardinal=cardinal;E.M.front_matrix=front_matrix
        r=E._evaluate_surfaces(None,None,qmc_samples);CUDA.synchronize()
    finally:
        E.STOP_R,E.FILM,E.TARGET_EFL,E.TARGET_FNO,E.MAX_FIELD_DEG,E.M.make_surfaces,E.M.cardinal,E.M.front_matrix=old
    r['J']=r['merit_J'];r['config_hash']=ch;r['evaluator_config']=cfg
    r['prescription_adapter']=ADAPTER_VERSION;r['backend']=CUDA.BACKEND;r['device']=device
    r['film_z_mm']=film;r['stop_radius_mm']=stop_r;r['target_efl_mm']=target_efl;r['target_fno']=target_fno;r['max_field_deg']=max_field
    return r
