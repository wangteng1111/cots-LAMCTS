import math, json, random, time, argparse, hashlib, os
from dataclasses import dataclass, asdict
from typing import Tuple
import numpy as np
from scipy.optimize import root
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ============================================================
# COTS-LA-MCTS v3 Mandler meta benchmark
# ONE evaluator only: evaluate_final(). No L1/L2/L3/proxy score.
# ============================================================

EX = [
[48.88,8.89,1.62286,60.08,182.96,0.38,36.92,15.11,1.58565,46.17,math.inf,2.31,1.67764,31.97,23.06,9.14,13.36,-23.91,1.92,1.57046,42.56,math.inf,7.77,1.64128,55.15,-36.92,0.38,1063.24,6.73,1.62286,60.08,-48.88,59.21],
[59.32,18.56,1.79227,47.15,147.94,0.38,38.70,11.27,1.67133,41.64,math.inf,2.30,1.73430,28.19,26.03,10.07,13.12,-26.03,1.91,1.62408,36.11,math.inf,8.46,1.72055,47.69,-38.70,0.38,-9525.05,6.38,1.72055,47.69,-59.32,59.71],
[51.05,9.06,1.64128,55.15,190.84,0.38,37.54,15.36,1.60889,43.63,math.inf,2.31,1.70444,29.84,23.67,9.16,13.35,-24.78,1.92,1.59911,38.91,math.inf,7.74,1.66152,50.59,-37.54,0.38,767.77,6.73,1.64128,55.15,-51.05,59.04],
[54.30,11.38,1.74795,44.49,149.79,0.39,37.69,12.59,1.67133,41.64,math.inf,2.31,1.73430,28.19,24.56,10.91,13.78,-25.95,1.93,1.63003,35.45,math.inf,6.97,1.72055,47.69,-37.69,0.39,-636.72,8.66,1.72055,47.69,-54.30,60.03],
[84.34,7.76,1.76166,27.37,298.43,0.38,38.93,13.61,1.70587,40.75,math.inf,1.94,1.79190,25.55,28.59,10.54,15.07,-28.59,1.91,1.65284,33.59,math.inf,9.48,1.79227,47.15,-38.93,0.38,math.inf,7.40,1.79227,47.15,-84.34,71.20],
[50.16,7.35,1.66152,50.59,186.96,0.38,37.35,14.33,1.60889,43.63,math.inf,2.31,1.70444,29.84,23.33,9.90,13.12,-25.18,1.92,1.59911,38.91,math.inf,8.00,1.66152,50.59,-37.35,0.38,math.inf,6.73,1.64128,55.15,-50.16,61.84],
[57.94,9.42,1.79227,47.15,190.48,0.38,40.31,14.67,1.62987,38.68,math.inf,3.38,1.76859,26.31,25.75,10.33,13.81,-27.28,1.92,1.63003,35.45,math.inf,6.73,1.74759,44.49,-40.31,0.38,math.inf,7.94,1.72055,47.69,-57.94,57.59],
[82.13,7.67,1.73430,28.19,375.73,0.38,39.43,13.80,1.67133,41.64,math.inf,2.49,1.79190,25.55,28.74,12.08,12.84,-28.74,1.92,1.65222,33.60,math.inf,10.03,1.79227,47.15,-39.43,0.38,math.inf,7.09,1.79227,47.15,-82.13,71.77],
[59.94,9.57,1.79227,47.15,167.31,0.38,40.30,14.35,1.67133,41.64,math.inf,2.87,1.73430,28.19,25.67,10.81,13.39,-27.69,1.91,1.63003,35.45,math.inf,7.65,1.72055,47.69,-40.30,0.38,math.inf,8.61,1.72055,47.69,-59.94,58.88],
]

WAVES=(479.99,546.07,643.85)
WWEIGHTS=(0.25,0.5,0.25)
FIELDS=(0.0,0.25,0.5,0.75,1.0)
FREQS=(10.0,20.0,40.0)
FILM=89.37
TARGET_EFL=100.0
TARGET_FNO=2.0
MAX_FIELD=math.radians(22.5)

# 13-point deterministic 2-D pupil, fixed for every evaluated design.
PUPIL=[(0.0,0.0)]
for a in (0,math.pi/2,math.pi,3*math.pi/2):
    PUPIL.append((0.5*math.cos(a),0.5*math.sin(a)))
for k in range(8):
    a=2*math.pi*k/8
    PUPIL.append((math.cos(a),math.sin(a)))
PUPIL=tuple(PUPIL)

FINAL_EVALUATOR_CONFIG={
    'name':'mandler_final_v3', 'wavelength_nm':WAVES, 'wavelength_weights':WWEIGHTS,
    'fields_fraction':FIELDS, 'pupil_points':PUPIL, 'frequencies_lpmm':FREQS,
    'film_z_mm':FILM, 'target_efl_mm':TARGET_EFL, 'target_fno':TARGET_FNO,
    'max_field_deg':22.5, 'stop_shooting':'wavelength-specific exact',
    'stop_solver':'deterministic_newton_5starts_6iter', 'stop_tolerance_mm':1e-5,
    'score':'legacy core metadata only',
    'no_intermediate_scores':True,
}
CONFIG_HASH=hashlib.sha256(json.dumps(FINAL_EVALUATOR_CONFIG,sort_keys=True).encode()).hexdigest()


def cauchy_from_ne_ve(ne,ve,lam):
    F,E,C=WAVES
    delta=(ne-1.0)/ve
    B=delta/(1/F**2-1/C**2)
    A=ne-B/E**2
    return A+B/lam**2

@dataclass(frozen=True)
class Component:
    label:str
    radii:Tuple[float,...]
    th:Tuple[float,...]
    ne:Tuple[float,...]
    ve:Tuple[float,...]
    source_example:int
    source_group:int
    def oriented(self,flip:bool):
        if not flip:return self
        rr=tuple((-r if math.isfinite(r) else r) for r in reversed(self.radii))
        return Component(self.label,rr,tuple(reversed(self.th)),tuple(reversed(self.ne)),tuple(reversed(self.ve)),self.source_example,self.source_group)

def build_catalog():
    out=[]
    for ei,row in enumerate(EX,1):
        (r1,a1,n1,v1,r2,a2,r3,a3,n2,v2,r4,a4,n3,v3,r5,a5,a6,r7,a7,n4,v4,r8,a8,n5,v5,r9,a9,r10,a10,n6,v6,r11,bfl)=row
        out += [
            Component(f'E{ei}G1',(r1,r2),(a1,),(n1,),(v1,),ei,1),
            Component(f'E{ei}G2',(r3,r4,r5),(a3,a4),(n2,n3),(v2,v3),ei,2),
            Component(f'E{ei}G3',(r7,r8,r9),(a7,a8),(n4,n5),(v4,v5),ei,3),
            Component(f'E{ei}G4',(r10,r11),(a10,),(n6,),(v6,),ei,4),
        ]
    return out
CAT=build_catalog()

@dataclass
class Surf:
    z:float; R:float; n1:float; n2:float

def make_surfaces(design,lam,gaps=(0.38,9.14,13.36,0.38)):
    comps=[CAT[i].oriented(f) for i,f in design]
    g12,g2s,gs3,g34=gaps
    L=[sum(c.th) for c in comps]
    z2=-g2s-L[1]; z1=z2-g12-L[0]; z3=gs3; z4=z3+L[2]+g34
    starts=[z1,z2,z3,z4]
    surfs=[]
    for c,z0 in zip(comps,starts):
        regs=[1.0]+[cauchy_from_ne_ve(ne,ve,lam) for ne,ve in zip(c.ne,c.ve)]+[1.0]
        z=z0
        for si,R in enumerate(c.radii):
            surfs.append(Surf(z,R,regs[si],regs[si+1]))
            if si<len(c.th):z+=c.th[si]
    surfs.sort(key=lambda s:s.z)
    return surfs

def mat_ref(n1,n2,R):
    if math.isinf(R):return np.array([[1.,0.],[0.,n1/n2]])
    return np.array([[1.,0.],[-(n2-n1)/(n2*R),n1/n2]])
def mat_tr(d):return np.array([[1.,d],[0.,1.]])
def cardinal(surfs):
    M=np.eye(2);zp=surfs[0].z
    for k,s in enumerate(surfs):
        if k:M=mat_tr(s.z-zp)@M
        M=mat_ref(s.n1,s.n2,s.R)@M;zp=s.z
    A=float(M[0,0]);C=float(M[1,0])
    if abs(C)<1e-12:return math.inf,math.inf
    return -1/C,zp-A/C
def front_matrix(surfs,stop=0.0):
    fs=[s for s in surfs if s.z<stop]
    if not fs:return None
    M=np.eye(2);zp=fs[0].z
    for k,s in enumerate(fs):
        if k:M=mat_tr(s.z-zp)@M
        M=mat_ref(s.n1,s.n2,s.R)@M;zp=s.z
    return mat_tr(stop-zp)@M

REF_DES=tuple((i,False) for i in (0,1,2,3))
REF_SURF=make_surfaces(REF_DES,WAVES[1]); REF_EFL,REF_FOCUS=cardinal(REF_SURF)
FMREF=front_matrix(REF_SURF,0.0); REF_STOP_RADIUS=abs(float(FMREF[0,0]))*(TARGET_EFL/TARGET_FNO)/2.0

def comp_signature(c):
    rr=list(c.radii)+[math.inf]*(3-len(c.radii)); curv=[0.0 if math.isinf(r) else 50.0/r for r in rr[:3]]
    th=(list(c.th)+[0.0]*(2-len(c.th)))[:2]; th=[x/20.0 for x in th]
    mats=[]
    for j in range(2):
        if j<len(c.ne):
            for lam in WAVES: mats.append(cauchy_from_ne_ve(c.ne[j],c.ve[j],lam)-1.0)
        else: mats += [0.,0.,0.]
    return np.asarray(curv+th+mats,dtype=np.float64)

ORIENTED=[]
for i,c in enumerate(CAT):
    for f in (False,True): ORIENTED.append((i,f,comp_signature(c.oriented(f))))
SIGMAT=np.stack([x[2] for x in ORIENTED]); SIG_MEAN=SIGMAT.mean(0); SIG_STD=SIGMAT.std(0)+1e-6

def audit_description(d):
    return [{'example':CAT[i].source_example,'group':CAT[i].source_group,'flip':bool(f),'label':CAT[i].label} for i,f in d]
