"""COTS-LA-MCTS authoritative direct-OPD market evaluator v4 (single-fidelity Q4096).

Key changes from v3:
* authoritative labels are Q4096 only;
* sag-safe first-surface initialization;
* log-domain MTF global + lower-tail merit;
* no duplicate paraxial-focus penalty (fixed-film diffraction MTF already contains defocus);
* circular-pupil fill normalization (not square-Sobol denominator);
* relative illumination from per-wavelength field/on-axis pupil-fill ratio * cos^4;
* max/full-field distortion rather than mean distortion;
* robust pseudo-Huber EFL/F-number market penalties;
* J is the sole authoritative scalar; search score is -J.
"""
import math, json, hashlib, importlib.util, time, warnings, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import qmc
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

_HERE=Path(__file__).resolve()
if os.environ.get('ALPHA_LENSE_PRESCRIPTION_CORE')=='1':
    _CORE_CANDIDATES=[_HERE.parent/'optical_primitives.py']
else:
    _CORE_CANDIDATES=[_HERE.parent.parent/'core'/'mandler_benchmark_core.py',Path('/mnt/data/cots_meta/mandler_benchmark_core.py'),Path('/mnt/data/mandler_meta_v3_final.py')]
_CORE_PATH=next((p for p in _CORE_CANDIDATES if p.exists()),None)
if _CORE_PATH is None: raise FileNotFoundError('optical core not found')
_SPEC=importlib.util.spec_from_file_location('M',str(_CORE_PATH)); M=importlib.util.module_from_spec(_SPEC); sys.modules[_SPEC.name]=M; _SPEC.loader.exec_module(M)

WAVES=(479.99,546.07,643.85); WWEIGHTS=(0.25,0.5,0.25); FIELDS=(0.,.25,.5,.75,1.)
MAX_FIELD_DEG=22.5; FREQS=(10.,20.,40.,60.); FILM=89.37; TARGET_EFL=100.; TARGET_FNO=2.
QMC_SAMPLES=4096; CERT_QMC_SAMPLES=4096; QMC_SEED=20260908; STOP_R=getattr(M,'REF_STOP_RADIUS',25.0); REF_DES=getattr(M,'REF_DES',None)
MTF_EPS=1e-4; W_MTF_LOG_GLOBAL=2.8; W_MTF_LOG_TAIL=0.9; MTF_TAIL_FRACTION=.20
W_ILLUM_LOG=.65; W_ONAXIS_PUPIL_LOG=.80; ILLUM_EPS=1e-4
W_DIST=.40; DIST_SCALE_PCT=1.; W_LCA=.40; LCA_SCALE_MM=.020
EFL_REL_SIGMA=.03; FNO_REL_SIGMA=.08; CARDINAL_HUBER_DELTA=2.; W_EFL=.50; W_FNO=.50
QUALITY_DISPLAY_SCALE=10.
CONFIG={
 'version':'market_direct_opd_v4_q4096_single_fidelity','wavelengths_nm':WAVES,'wavelength_weights':WWEIGHTS,
 'field_fractions':FIELDS,'max_field_deg':MAX_FIELD_DEG,'mtf_lpmm':FREQS,'film_z_mm':FILM,
 'target_efl_mm':TARGET_EFL,'target_fno':TARGET_FNO,'qmc_samples_authoritative':QMC_SAMPLES,
 'single_fidelity_labels':True,'qmc_seed':QMC_SEED,'mtf_engine':'direct_OPD_pupil_autocorrelation',
 'raytrace':'numpy_exact_spherical_sag_safe_v2','fixed_image_plane':True,
 'authoritative_scalar':'J_lower_is_better; search score=-J','quality_score_100':'100*exp(-J/10), display only',
 'mtf_merit':'log-domain all-field/all-frequency/all-axis + lower-tail mean','mtf_eps':MTF_EPS,
 'mtf_tail_fraction':MTF_TAIL_FRACTION,'focus_penalty':'none; fixed-film diffraction MTF already includes defocus',
 'pupil_fill_normalization':'valid rays / Sobol samples inside unit disk',
 'relative_illumination':'per-wavelength field/on-axis pupil-fill ratio * cos(field)^4',
 'distortion':'max absolute full-field percent','lca':'max chief-ray axial color spread mm',
}
CONFIG_HASH=hashlib.sha256(json.dumps(CONFIG,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def _pseudo_huber(z,delta=CARDINAL_HUBER_DELTA):return delta*delta*(math.sqrt(1.+(z/delta)**2)-1.)
def _safe_log(x,eps):return math.log(max(float(x),eps))
def _dir(ax,ay=0.):
 tx,ty=math.tan(ax),math.tan(ay);dz=1/math.sqrt(1+tx*tx+ty*ty);return dz,tx*dz,ty*dz

def _sag_ds(R,r):
 if math.isinf(R):return 0.,0.,True
 q=R*R-r*r
 if q<=0:return math.nan,math.nan,False
 sq=math.sqrt(q);sg=R-(1 if R>=0 else -1)*sq;ds=(1 if R>=0 else -1)*r/sq;return sg,ds,True

def _safe_incident_start(surfs,ax,xv,yv):
 s0=surfs[0];finite_R=[abs(s.R) for s in surfs if math.isfinite(s.R)];ap=min(60.,max(20.,.45*min(finite_R) if finite_R else 20.));z0=s0.z-max(10.,2.5*ap);dz,dx,dy=_dir(ax);dt=(s0.z-z0)/dz;return z0,xv-dt*dx,yv-dt*dy,dz,dx,dy

def _surf_step(z,x,y,dz,dx,dy,s,opl=None,maxit=14):
 tv=(s.z-z)/dz;xv=x+tv*dx;yv=y+tv*dy;r=math.hypot(xv,yv);sg,_,ok=_sag_ds(s.R,r)
 if not ok:return None
 t=max(1e-10,(s.z+sg-z)/dz)
 for _ in range(maxit):
  xx=x+t*dx;yy=y+t*dy;r=math.hypot(xx,yy);sg,ds,ok=_sag_ds(s.R,r)
  if not ok:return None
  radial=(xx*dx+yy*dy)/r if r>1e-14 else 0.;f=z+t*dz-s.z-sg;df=dz-ds*radial
  if abs(df)<1e-14:return None
  step=f/df;t-=step
  if abs(step)<1e-10:break
 if not math.isfinite(t) or t<=0:return None
 zn=z+t*dz;xn=x+t*dx;yn=y+t*dy
 if opl is not None:opl+=s.n1*math.sqrt((zn-z)**2+(xn-x)**2+(yn-y)**2)
 r=math.hypot(xn,yn);_,ds,ok=_sag_ds(s.R,r)
 if not ok:return None
 nz=1.;nx=-ds*xn/r if r>1e-14 else 0.;ny=-ds*yn/r if r>1e-14 else 0.;nn=math.sqrt(nz*nz+nx*nx+ny*ny);nz,nx,ny=nz/nn,nx/nn,ny/nn
 if dz*nz+dx*nx+dy*ny>0:nz,nx,ny=-nz,-nx,-ny
 eta=s.n1/s.n2;cosi=-(dz*nz+dx*nx+dy*ny);kk=1-eta*eta*(1-cosi*cosi)
 if kk<0:return None
 qq=eta*cosi-math.sqrt(kk);oz=eta*dz+qq*nz;ox=eta*dx+qq*nx;oy=eta*dy+qq*ny;norm=math.sqrt(oz*oz+ox*ox+oy*oy);dz2,dx2,dy2=oz/norm,ox/norm,oy/norm;eps=1e-8
 return zn+dz2*eps,xn+dx2*eps,yn+dy2*eps,dz2,dx2,dy2,opl

def _trace_stop(surfs,ax,x0,y0,stop_z=0.):
 z,x,y,dz,dx,dy=_safe_incident_start(surfs,ax,x0,y0)
 for s in [ss for ss in surfs if ss.z<stop_z-1e-10]:
  o=_surf_step(z,x,y,dz,dx,dy,s)
  if o is None:return None
  z,x,y,dz,dx,dy,_=o
 tt=(stop_z-z)/dz
 if tt<=0:return None
 return x+tt*dx,y+tt*dy

def _solve_input(surfs,ax,target,stop_z=0.,maxit=7):
 fm=M.front_matrix(surfs,stop_z)
 if fm is None:return None
 A,B=fm[0,0],fm[0,1]
 if abs(A)<1e-12:return None
 q=np.array([(target[0]-B*ax)/A,target[1]/A],float)
 for _ in range(maxit):
  v=_trace_stop(surfs,ax,*q,stop_z)
  if v is None:return None
  f=np.array(v)-target
  if np.linalg.norm(f)<2e-6:return q
  h=1e-3;vx=_trace_stop(surfs,ax,q[0]+h,q[1],stop_z);vy=_trace_stop(surfs,ax,q[0],q[1]+h,stop_z)
  if vx is None or vy is None:return None
  J=np.column_stack(((np.array(vx)-v)/h,(np.array(vy)-v)/h))
  try:q-=np.linalg.solve(J,f)
  except np.linalg.LinAlgError:return None
 v=_trace_stop(surfs,ax,*q,stop_z);return q if v is not None and np.linalg.norm(np.array(v)-target)<2e-5 else None

def chief_and_image(surfs,ax,film=FILM):
 q=_solve_input(surfs,ax,np.array([0.,0.]))
 if q is None:return None
 z,x,y,dz,dx,dy=_safe_incident_start(surfs,ax,*q)
 for s in surfs:
  o=_surf_step(z,x,y,dz,dx,dy,s)
  if o is None:return None
  z,x,y,dz,dx,dy,_=o
 t=(film-z)/dz;return x+t*dx,y+t*dy

def _eikonal(surfs,ax,uv,ref,film=FILM):
 if uv[0]*uv[0]+uv[1]*uv[1]>1:return math.nan
 q=_solve_input(surfs,ax,np.array(uv)*STOP_R)
 if q is None:return math.nan
 z,x,y,dz,dx,dy=_safe_incident_start(surfs,ax,*q);opl=dx*x+dy*y;crossed=False
 for s in surfs:
  if not crossed and z<0<s.z:
   tt=(0-z)/dz;sx=x+tt*dx;sy=y+tt*dy
   if math.hypot(sx,sy)>STOP_R*(1+1e-8):return math.nan
   crossed=True
  o=_surf_step(z,x,y,dz,dx,dy,s,opl)
  if o is None:return math.nan
  z,x,y,dz,dx,dy,opl=o
 xr,yr=ref;return opl+math.sqrt((film-z)**2+(xr-x)**2+(yr-y)**2)

def direct_otf_batch(surfs,lam,ax,fno,ref,seed,qmc_samples=QMC_SAMPLES):
 m=int(round(math.log2(qmc_samples)));base=qmc.Sobol(2,scramble=True,seed=seed).random_base2(m)*2-1;inside=np.sum(base*base,axis=1)<=1;den=np.count_nonzero(inside);lm=lam*1e-6;ot,os_={},{}
 for f in FREQS:
  nu=lm*abs(fno)*f
  if nu>=1:ot[str(f)]=0j;os_[str(f)]=0j;continue
  d=2*nu;vals=[]
  for shift in (np.array([d/2,0.]),np.array([0.,d/2])):
   num=0j
   for u in base[inside]:
    sp=_eikonal(surfs,ax,u+shift,ref);sm=_eikonal(surfs,ax,u-shift,ref)
    if math.isfinite(sp) and math.isfinite(sm):num+=np.exp(2j*math.pi*(sp-sm)/lm)
   vals.append(num/max(den,1))
  ot[str(f)],os_[str(f)]=vals
 return {'otf_t':ot,'otf_s':os_,'pf':den/max(np.count_nonzero(inside),1)}

def _evaluate_surfaces(design,gaps=None,qmc_samples=QMC_SAMPLES):
 all_mtf=[];pfs={};chief={};efls=[];fnos=[]
 for wi,(lam,ww) in enumerate(zip(WAVES,WWEIGHTS)):
  surfs=M.make_surfaces(design,lam,gaps=gaps) if gaps is not None else M.make_surfaces(design,lam);efl,focus=M.cardinal(surfs);fm=M.front_matrix(surfs,0.)
  if fm is None or not math.isfinite(efl):return {'merit_J':1e9,'config_hash':CONFIG_HASH}
  epd=2*STOP_R/abs(float(fm[0,0]));fno=abs(efl)/epd;efls.append(efl);fnos.append(fno)
  for fi,frac in enumerate(FIELDS):
   ax=math.radians(MAX_FIELD_DEG*frac);ref=chief_and_image(surfs,ax)
   if ref is None:return {'merit_J':1e9,'config_hash':CONFIG_HASH}
   chief[(wi,fi)]=ref;d=direct_otf_batch(surfs,lam,ax,fno,ref,QMC_SEED+wi*100+fi,qmc_samples);pfs[(wi,fi)]=d['pf']
   for f in FREQS:all_mtf.extend([abs(d['otf_t'].get(str(f),0j)),abs(d['otf_s'].get(str(f),0j))])
 vals=np.maximum(np.asarray(all_mtf,float),MTF_EPS);mtf_log_global=-np.mean(np.log(vals));tail_n=max(1,int(math.ceil(MTF_TAIL_FRACTION*len(vals))));mtf_log_tail=-np.mean(np.log(np.sort(vals)[:tail_n]));onaxis=sum(WWEIGHTS[w]*pfs[(w,0)] for w in range(len(WAVES)));ill=[]
 for fi,frac in enumerate(FIELDS):
  rel=sum(WWEIGHTS[w]*(pfs[(w,fi)]/max(pfs[(w,0)],ILLUM_EPS)) for w in range(len(WAVES)))*math.cos(math.radians(MAX_FIELD_DEG*frac))**4;ill.append(rel)
 dist=[]
 for fi,frac in enumerate(FIELDS[1:],1):
  ideal=abs(efls[1])*math.tan(math.radians(MAX_FIELD_DEG*frac));actual=abs(chief[(1,fi)][0]);dist.append(100*(actual-ideal)/max(ideal,1e-9))
 lca=max(abs(chief[(0,-1 if False else len(FIELDS)-1)][0]-chief[(2,len(FIELDS)-1)][0]),0.);efl=float(efls[1]);fno=float(fnos[1]);dist_max=max(abs(x) for x in dist) if dist else 0.;J=W_MTF_LOG_GLOBAL*mtf_log_global+W_MTF_LOG_TAIL*mtf_log_tail+W_ILLUM_LOG*(-_safe_log(min(ill),ILLUM_EPS))+W_ONAXIS_PUPIL_LOG*(-_safe_log(onaxis,ILLUM_EPS))+W_DIST*_pseudo_huber(dist_max/DIST_SCALE_PCT)+W_LCA*_pseudo_huber(lca/LCA_SCALE_MM)+W_EFL*_pseudo_huber((efl-TARGET_EFL)/(TARGET_EFL*EFL_REL_SIGMA))+W_FNO*_pseudo_huber((fno-TARGET_FNO)/(TARGET_FNO*FNO_REL_SIGMA))
 return {'merit_J':float(J),'efl':efl,'fno':fno,'mtf_mean':float(np.mean(vals)),'mtf_geomean_reg':float(math.exp(np.mean(np.log(vals)))),'mtf_p10':float(np.quantile(vals,.1)),'onaxis_pupil_fill':float(onaxis),'min_illum':float(min(ill)),'dist_max':float(dist_max),'lca':float(lca),'config_hash':CONFIG_HASH}

def evaluate_final(design,qmc_samples=QMC_SAMPLES):return _evaluate_surfaces(design,None,qmc_samples)
def evaluate_final_with_gaps(design,gaps,qmc_samples=QMC_SAMPLES):return _evaluate_surfaces(design,tuple(float(x) for x in gaps),qmc_samples)
