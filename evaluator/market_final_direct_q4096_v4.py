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
import math, json, hashlib, importlib.util, time, warnings, os
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
_SPEC=importlib.util.spec_from_file_location('M',str(_CORE_PATH)); M=importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(M)

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
 'mtf_tail_fraction':MTF_TAIL_FRACTION,'focus_penalty':'none; fixed-film diffraction MTF already contains defocus',
 'pupil_fill_normalization':'valid rays / Sobol samples inside unit disk',
 'relative_illumination':'per-wavelength field/on-axis pupil-fill ratio * cos(field)^4',
 'distortion_metric':'max absolute field distortion percent','lca_metric':'max wavelength focus spread mm',
 'market_constraints':{'target_efl_mm':TARGET_EFL,'target_fno':TARGET_FNO},
}
CONFIG_HASH=hashlib.sha256(json.dumps(CONFIG,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def _dir(ax,ay=0.):
 tx,ty=math.tan(ax),math.tan(ay);dz=1/math.sqrt(1+tx*tx+ty*ty);return dz,tx*dz,ty*dz
def _sag_ds(R,r):
 if math.isinf(R):return 0.,0.,True
 q=R*R-r*r
 if q<=0:return 0.,0.,False
 sq=math.sqrt(q);sg=R-(1 if R>=0 else -1)*sq;ds=(1 if R>=0 else -1)*r/sq;return sg,ds,True
def _safe_incident_start(surfs,ax,xv,yv):
 s0=surfs[0];finite_R=[abs(s.R) for s in surfs if math.isfinite(s.R)];ap=min(60.,max(20.,.45*min(finite_R) if finite_R else 20.));z0=s0.z-max(10.,2.5*ap);dz,dx,dy=_dir(ax);dt=(s0.z-z0)/dz;return z0,xv-dt*dx,yv-dt*dy,dz,dx,dy
def _surf_step(z,x,y,dz,dx,dy,s,opl=None,maxit=14):
 tv=(s.z-z)/dz;xv=x+tv*dx;yv=y+tv*dy;r=math.hypot(xv,yv);sg,_,ok=_sag_ds(s.R,r)
 if not ok:return None
 t=max((s.z+sg-z)/dz,1e-10)
 for _ in range(maxit):
  xx=x+t*dx;yy=y+t*dy;r=math.hypot(xx,yy);sg,ds,ok=_sag_ds(s.R,r)
  if not ok:return None
  radial=(xx*dx+yy*dy)/r if r>1e-14 else 0.;f=z+t*dz-s.z-sg;df=dz-ds*radial
  if abs(df)<1e-14:return None
  step=f/df;t-=step
  if abs(step)<1e-10:break
 if t<=0 or not math.isfinite(t):return None
 zn=z+t*dz;xn=x+t*dx;yn=y+t*dy
 if opl is not None:opl+=s.n1*math.sqrt((zn-z)**2+(xn-x)**2+(yn-y)**2)
 r=math.hypot(xn,yn);_,ds,ok=_sag_ds(s.R,r)
 if not ok:return None
 nz=1.;nx=-ds*xn/r if r>1e-14 else 0.;ny=-ds*yn/r if r>1e-14 else 0.;nn=math.sqrt(nz*nz+nx*nx+ny*ny);nz,nx,ny=nz/nn,nx/nn,ny/nn
 if dz*nz+dx*nx+dy*ny>0:nz,nx,ny=-nz,-nx,-ny
 eta=s.n1/s.n2;cosi=-(dz*nz+dx*nx+dy*ny);kk=1-eta*eta*(1-cosi*cosi)
 if kk<0:return None
 qq=eta*cosi-math.sqrt(kk);oz=eta*dz+qq*nz;ox=eta*dx+qq*nx;oy=eta*dy+qq*ny;nn=math.sqrt(oz*oz+ox*ox+oy*oy);dz2,dx2,dy2=oz/nn,ox/nn,oy/nn;eps=1e-8
 return zn+dz2*eps,xn+dx2*eps,yn+dy2*eps,dz2,dx2,dy2,opl

def trace_stop(surfs,ax,x0,y0,stop_z=0.):
 z,x,y,dz,dx,dy=_safe_incident_start(surfs,ax,x0,y0)
 for s in [ss for ss in surfs if ss.z<stop_z-1e-10]:
  q=_surf_step(z,x,y,dz,dx,dy,s)
  if q is None:return None
  z,x,y,dz,dx,dy,_=q
 t=(stop_z-z)/dz
 if t<=0:return None
 return np.array([x+t*dx,y+t*dy])
def solve_input(surfs,ax,target,stop_z=0.,maxit=7):
 fm=M.front_matrix(surfs,stop_z)
 if fm is None:return None
 A,B=fm[0,0],fm[0,1]
 if abs(A)<1e-12:return None
 q=np.array([(target[0]-B*ax)/A,target[1]/A],float)
 for _ in range(maxit):
  v=trace_stop(surfs,ax,q[0],q[1],stop_z)
  if v is None:return None
  f=v-target
  if np.linalg.norm(f)<2e-6:return q
  h=1e-3;vx=trace_stop(surfs,ax,q[0]+h,q[1],stop_z);vy=trace_stop(surfs,ax,q[0],q[1]+h,stop_z)
  if vx is None or vy is None:return None
  J=np.column_stack(((vx-v)/h,(vy-v)/h))
  try:dq=np.linalg.solve(J,f)
  except np.linalg.LinAlgError:return None
  q-=dq
 return q if (trace_stop(surfs,ax,q[0],q[1],stop_z) is not None) else None

def _eikonal(surfs,ax,uv,ref,film=FILM,stop_radius=None):
 sr=STOP_R if stop_radius is None else stop_radius
 if np.dot(uv,uv)>1:return np.nan
 target=np.asarray(uv)*sr;q=solve_input(surfs,ax,target)
 if q is None:return np.nan
 z,x,y,dz,dx,dy=_safe_incident_start(surfs,ax,q[0],q[1]);opl=dx*x+dy*y;crossed=False
 for s in surfs:
  if not crossed and z<0<s.z:
   t=-z/dz;sx=x+t*dx;sy=y+t*dy
   if math.hypot(sx,sy)>sr*(1+1e-8):return np.nan
   crossed=True
  o=_surf_step(z,x,y,dz,dx,dy,s,opl)
  if o is None:return np.nan
  z,x,y,dz,dx,dy,opl=o
 xr,yr=ref;return opl+math.sqrt((film-z)**2+(xr-x)**2+(yr-y)**2)

def direct_otf_batch(surfs,lam,ax,fno,ref,seed,qmc_samples=QMC_SAMPLES):
 m=int(round(math.log2(qmc_samples)));base=qmc.Sobol(2,scramble=True,seed=seed).random_base2(m)*2-1;lm=lam*1e-6;S0=np.array([_eikonal(surfs,ax,u,ref) for u in base]);valid0=np.isfinite(S0);den=np.count_nonzero(valid0);ot={};os={}
 if den==0:return {'otf_t':{},'otf_s':{},'pf':0.}
 for f in FREQS:
  nu=lm*abs(fno)*f;d=2*nu
  if nu>=1:ot[str(f)]=0j;os[str(f)]=0j;continue
  vals=[]
  for sh in (np.array([d/2,0.]),np.array([0.,d/2])):
   Sp=np.array([_eikonal(surfs,ax,u+sh,ref) for u in base]);Sm=np.array([_eikonal(surfs,ax,u-sh,ref) for u in base]);good=np.isfinite(Sp)&np.isfinite(Sm);vals.append(np.exp(2j*np.pi*(Sp[good]-Sm[good])/lm).sum()/den if np.any(good) else 0j)
  ot[str(f)],os[str(f)]=vals
 n_circle=np.count_nonzero(np.sum(base*base,axis=1)<=1);return {'otf_t':ot,'otf_s':os,'pf':den/max(n_circle,1)}

def _chief(surfs,ax):
 q=solve_input(surfs,ax,np.array([0.,0.]));
 if q is None:return None
 z,x,y,dz,dx,dy=_safe_incident_start(surfs,ax,q[0],q[1])
 for s in surfs:
  o=_surf_step(z,x,y,dz,dx,dy,s)
  if o is None:return None
  z,x,y,dz,dx,dy,_=o
 t=(FILM-z)/dz;return np.array([x+t*dx,y+t*dy])
def _pseudo_huber(z,delta):return delta*delta*(math.sqrt(1+(z/delta)**2)-1)

def _evaluate_surfaces(design,gaps,qmc_samples):
 t0=time.time();surf_by_w=[M.make_surfaces(design,lam,gaps=gaps) for lam in WAVES];se=surf_by_w[1];efl,focus=M.cardinal(se);fm=M.front_matrix(se,0.)
 if fm is None or not np.isfinite(efl) or abs(fm[0,0])<1e-12:return {'merit_J':1e9,'config_hash':CONFIG_HASH}
 entrance_radius=STOP_R/abs(fm[0,0]);fno=abs(efl)/(2*entrance_radius);mtfs=[];pfs=[];dist=[];chiefs=[];focuses=[]
 for wi,(lam,ww) in enumerate(zip(WAVES,WWEIGHTS)):
  sw=surf_by_w[wi];_,fw=M.cardinal(sw);focuses.append(fw)
  for fi,frac in enumerate(FIELDS):
   ax=math.radians(MAX_FIELD_DEG*frac);ref=_chief(sw,ax)
   if ref is None:continue
   chiefs.append((wi,fi,ref));r=direct_otf_batch(sw,lam,ax,fno,ref,QMC_SEED+wi*101+fi,qmc_samples);pfs.append((wi,fi,r['pf']))
   for f in FREQS:
    for key in ('otf_t','otf_s'):mtfs.append(max(abs(r[key].get(str(f),0j)),MTF_EPS))
   if frac>0:ideal=abs(efl*math.tan(ax));dist.append(100*abs(abs(ref[0])-ideal)/max(ideal,1e-9))
 if not mtfs:return {'merit_J':1e9,'config_hash':CONFIG_HASH}
 logs=-np.log(np.asarray(mtfs));global_m=float(np.mean(logs));k=max(1,int(math.ceil(MTF_TAIL_FRACTION*len(logs))));tail=float(np.mean(np.sort(logs)[-k:]));on=[pf for wi,fi,pf in pfs if fi==0];full=[pf for wi,fi,pf in pfs if fi==len(FIELDS)-1];onpf=max(float(np.mean(on)) if on else 0.,ILLUM_EPS);fullpf=max(float(np.mean(full)) if full else 0.,ILLUM_EPS);illum=max(fullpf/onpf*math.cos(math.radians(MAX_FIELD_DEG))**4,ILLUM_EPS);dmax=max(dist) if dist else 0.;lca=max(focuses)-min(focuses) if focuses else 0.;zef=(efl-TARGET_EFL)/(EFL_REL_SIGMA*TARGET_EFL);zfn=(fno-TARGET_FNO)/(FNO_REL_SIGMA*TARGET_FNO);J=W_MTF_LOG_GLOBAL*global_m+W_MTF_LOG_TAIL*tail+W_ILLUM_LOG*(-math.log(illum))+W_ONAXIS_PUPIL_LOG*(-math.log(onpf))+W_DIST*_pseudo_huber(dmax/DIST_SCALE_PCT,2.)+W_LCA*_pseudo_huber(lca/LCA_SCALE_MM,2.)+W_EFL*_pseudo_huber(zef,CARDINAL_HUBER_DELTA)+W_FNO*_pseudo_huber(zfn,CARDINAL_HUBER_DELTA)
 return {'merit_J':float(J),'J':float(J),'efl':float(efl),'fno':float(fno),'mtf_mean':float(np.mean(mtfs)),'mtf_geomean_reg':float(math.exp(-global_m)),'mtf_p10':float(np.quantile(mtfs,.1)),'onaxis_pupil_fill':onpf,'min_illum':illum,'dist_max':float(dmax),'lca':float(lca),'config_hash':CONFIG_HASH,'elapsed_sec':time.time()-t0}
def evaluate_final(design,qmc_samples=QMC_SAMPLES):return _evaluate_surfaces(design,None,qmc_samples)
def evaluate_final_with_gaps(design,gaps,qmc_samples=QMC_SAMPLES):return _evaluate_surfaces(design,tuple(float(x) for x in gaps),qmc_samples)
