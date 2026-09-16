"""Dependency-light optical primitives for arbitrary prescription evaluation."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
WAVES=(479.99,546.07,643.85)
def cauchy_from_ne_ve(ne,ve,lam):
    F,E,C=WAVES;delta=(ne-1.0)/ve;B=delta/(1/F**2-1/C**2);A=ne-B/E**2;return A+B/lam**2
@dataclass
class Surf:
    z:float;R:float;n1:float;n2:float
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
