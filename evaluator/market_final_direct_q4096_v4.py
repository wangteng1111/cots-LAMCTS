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
import math, json, hashlib, importlib.util, time, warnings
from pathlib import Path
import numpy as np
from scipy.stats import qmc
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

_HERE=Path(__file__).resolve()
_CORE_CANDIDATES=[_HERE.parent.parent/'core'/'mandler_benchmark_core.py',Path('/mnt/data/cots_meta/mandler_benchmark_core.py'),Path('/mnt/data/mandler_meta_v3_final.py')]
_CORE_PATH=next((p for p in _CORE_CANDIDATES if p.exists()),None)
if _CORE_PATH is None: raise FileNotFoundError('mandler_benchmark_core.py not found')
_SPEC=importlib.util.spec_from_file_location('M',str(_CORE_PATH)); M=importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(M)

WAVES=(479.99,546.07,643.85); WWEIGHTS=(0.25,0.5,0.25); FIELDS=(0.,.25,.5,.75,1.)
MAX_FIELD_DEG=22.5; FREQS=(10.,20.,40.,60.); FILM=89.37; TARGET_EFL=100.; TARGET_FNO=2.
QMC_SAMPLES=4096; CERT_QMC_SAMPLES=4096; QMC_SEED=20260908; STOP_R=M.REF_STOP_RADIUS; REF_DES=M.REF_DES
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
 'distortion_metric':'max absolute nonzero-field distortion',
 'cardinal_penalty':'pseudo-Huber in relative EFL/F-number error; no hard gate','no_intermediate_scores':True}
CONFIG_HASH=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()

def _dir(ax,ay=0.):
 tx,ty=math.tan(ax),math.tan(ay); dz=1/math.sqrt(1+tx*tx+ty*ty); return dz,tx*dz,ty*dz

def _sag_ds_vec(R,r):
 r=np.asarray(r,float)
 if math.isinf(R): return np.zeros_like(r),np.zeros_like(r),np.ones_like(r,dtype=bool)
 q=R*R-r*r; ok=q>0; sq=np.sqrt(np.maximum(q,1e-30)); return R-np.sign(R)*sq,np.sign(R)*r/sq,ok

def _safe_incident_start(surfs,ax,x_vertex,y_vertex):
 s0=surfs[0]; finite_R=[abs(s.R) for s in surfs if math.isfinite(s.R)]
 aperture_scale=min(60.,max(20.,.45*min(finite_R) if finite_R else 20.)); z0=s0.z-max(10.,2.5*aperture_scale)
 dz0,dx0,dy0=_dir(ax); dt=(s0.z-z0)/dz0; xv=np.asarray(x_vertex,float); yv=np.asarray(y_vertex,float)
 return z0,xv-dt*dx0,yv-dt*dy0,dz0,dx0,dy0

def _surf_step(z,x,y,dz,dx,dy,s,opl=None,maxit=14):
 tv=(s.z-z)/dz; xv=x+tv*dx; yv=y+tv*dy; rv=np.hypot(xv,yv); sg0,_,ok=_sag_ds_vec(s.R,rv)
 t=np.maximum((s.z+sg0-z)/dz,1e-10); valid=ok&np.isfinite(t)
 for _ in range(maxit):
  xx=x+t*dx; yy=y+t*dy; r=np.hypot(xx,yy); sg,ds,ok2=_sag_ds_vec(s.R,r)
  radial=np.divide(xx*dx+yy*dy,r,out=np.zeros_like(r),where=r>1e-14); f=z+t*dz-s.z-sg; df=dz-ds*radial
  good=valid&ok2&(np.abs(df)>1e-14); step=np.zeros_like(t); step[good]=f[good]/df[good]; t[good]-=step[good]; valid&=good
  if not np.any(good) or np.nanmax(np.abs(step[good]))<1e-10: break
 valid&=(t>0)&np.isfinite(t); zn=z+t*dz; xn=x+t*dx; yn=y+t*dy
 if opl is not None: opl=opl+s.n1*np.sqrt((zn-z)**2+(xn-x)**2+(yn-y)**2)
 r=np.hypot(xn,yn); _,ds,okn=_sag_ds_vec(s.R,r); nz=np.ones_like(r); nx=np.zeros_like(r); ny=np.zeros_like(r); rr=r>1e-14
 nx[rr]=-ds[rr]*xn[rr]/r[rr]; ny[rr]=-ds[rr]*yn[rr]/r[rr]; nn=np.sqrt(nz*nz+nx*nx+ny*ny); nz/=nn; nx/=nn; ny/=nn
 dot=dz*nz+dx*nx+dy*ny; flip=dot>0; nz[flip]*=-1; nx[flip]*=-1; ny[flip]*=-1
 eta=s.n1/s.n2; cosi=-(dz*nz+dx*nx+dy*ny); kk=1-eta*eta*(1-cosi*cosi); valid&=okn&(kk>=0)
 root=np.sqrt(np.maximum(kk,0)); qq=eta*cosi-root; oz=eta*dz+qq*nz; ox=eta*dx+qq*nx; oy=eta*dy+qq*ny
 norm=np.sqrt(oz*oz+ox*ox+oy*oy); dz2=oz/norm; dx2=ox/norm; dy2=oy/norm; eps=1e-8
 return zn+dz2*eps,xn+dx2*eps,yn+dy2*eps,dz2,dx2,dy2,valid,opl

def trace_stop_batch(surfs,ax,x0,y0,stop_z=0.):
 n=len(x0); z0,xs,ys,dz0,dx0,dy0=_safe_incident_start(surfs,ax,x0,y0)
 z=np.full(n,z0); x=np.asarray(xs,float); y=np.asarray(ys,float); dz=np.full(n,dz0); dx=np.full(n,dx0); dy=np.full(n,dy0); valid=np.ones(n,bool)
 for s in [ss for ss in surfs if ss.z<stop_z-1e-10]: z,x,y,dz,dx,dy,ok,_=_surf_step(z,x,y,dz,dx,dy,s,None); valid&=ok
 tt=(stop_z-z)/dz; xs=x+tt*dx; ys=y+tt*dy; valid&=np.isfinite(xs)&np.isfinite(ys)&(tt>0); xs[~valid]=np.nan; ys[~valid]=np.nan
 return np.c_[xs,ys],valid

def solve_input_batch(surfs,ax,targets,stop_z=0.,maxit=7):
 fm=M.front_matrix(surfs,stop_z); N=len(targets)
 if fm is None: return np.full_like(targets,np.nan),np.zeros(N,bool),np.full(N,np.inf)
 A=float(fm[0,0]); B=float(fm[0,1])
 if abs(A)<1e-12: return np.full_like(targets,np.nan),np.zeros(N,bool),np.full(N,np.inf)
 q=np.empty((N,2)); q[:,0]=(targets[:,0]-B*ax)/A; q[:,1]=targets[:,1]/A; conv=np.zeros(N,bool)
 for _ in range(maxit):
  v,ok=trace_stop_batch(surfs,ax,q[:,0],q[:,1],stop_z); f=v-targets; err=np.linalg.norm(f,axis=1); conv|=ok&(err<2e-6); active=ok&~conv
  if not np.any(active): break
  h=1e-3; vx,okx=trace_stop_batch(surfs,ax,q[:,0]+h,q[:,1],stop_z); vy,oky=trace_stop_batch(surfs,ax,q[:,0],q[:,1]+h,stop_z)
  J00=(vx[:,0]-v[:,0])/h; J10=(vx[:,1]-v[:,1])/h; J01=(vy[:,0]-v[:,0])/h; J11=(vy[:,1]-v[:,1])/h; det=J00*J11-J01*J10
  good=active&okx&oky&(np.abs(det)>1e-12); dxq=np.zeros(N); dyq=np.zeros(N)
  dxq[good]=(J11[good]*f[good,0]-J01[good]*f[good,1])/det[good]; dyq[good]=(-J10[good]*f[good,0]+J00[good]*f[good,1])/det[good]
  q[good,0]-=dxq[good]; q[good,1]-=dyq[good]; q[~good&active]=np.nan
 q0=np.nan_to_num(q); v,ok=trace_stop_batch(surfs,ax,q0[:,0],q0[:,1],stop_z); err=np.linalg.norm(v-targets,axis=1); ok=ok&(err<2e-5)&np.isfinite(q).all(axis=1); q[~ok]=np.nan
 return q,ok,err

def trace_to_film_batch(surfs,ax,x0,y0,film=FILM,stop_z=0.,stop_radius=STOP_R):
 n=len(np.atleast_1d(x0)); z0,xs,ys,dz0,dx0,dy0=_safe_incident_start(surfs,ax,np.atleast_1d(x0),np.atleast_1d(y0))
 z=np.full(n,z0); x=np.asarray(xs,float); y=np.asarray(ys,float); dz=np.full(n,dz0); dx=np.full(n,dx0); dy=np.full(n,dy0); valid=np.ones(n,bool); crossed=np.zeros(n,bool)
 for s in surfs:
  cross=(~crossed)&(z<stop_z)&(stop_z<s.z)
  if np.any(cross):
   tt=(stop_z-z)/dz; sx=x+tt*dx; sy=y+tt*dy; valid&=(~cross)|(np.hypot(sx,sy)<=stop_radius*(1+1e-8)); crossed|=cross
  z,x,y,dz,dx,dy,ok,_=_surf_step(z,x,y,dz,dx,dy,s,None); valid&=ok
 cross=(~crossed)&(z<stop_z)
 if np.any(cross):
  tt=(stop_z-z)/dz; sx=x+tt*dx; sy=y+tt*dy; valid&=(~cross)|(np.hypot(sx,sy)<=stop_radius*(1+1e-8))
 tt=(film-z)/dz; xf=x+tt*dx; yf=y+tt*dy; valid&=(tt>0)&np.isfinite(xf)&np.isfinite(yf); xf[~valid]=np.nan; yf[~valid]=np.nan
 return np.c_[xf,yf],valid

def chief_batch(surfs,ax,film=FILM,stop_radius=STOP_R):
 q,ok,_=solve_input_batch(surfs,ax,np.array([[0.,0.]]))
 if not ok[0]: return None
 xy,v=trace_to_film_batch(surfs,ax,[q[0,0]],[q[0,1]],film,0.,stop_radius)
 return (float(xy[0,0]),float(xy[0,1])) if v[0] else None

def eikonal_batch(surfs,ax,uv,ref,film=FILM,stop_radius=STOP_R):
 uv=np.asarray(uv,float); inside=np.sum(uv*uv,axis=1)<=1.; targets=uv*stop_radius; q=np.full_like(uv,np.nan); ok=np.zeros(len(uv),bool)
 if np.any(inside): qq,oo,_=solve_input_batch(surfs,ax,targets[inside]); q[inside]=qq; ok[inside]=oo
 n=len(uv); z0,xs,ys,dz0,dx0,dy0=_safe_incident_start(surfs,ax,np.nan_to_num(q[:,0]),np.nan_to_num(q[:,1]))
 z=np.full(n,z0); x=np.asarray(xs,float); y=np.asarray(ys,float); dz=np.full(n,dz0); dx=np.full(n,dx0); dy=np.full(n,dy0); opl=dx*x+dy*y; valid=ok.copy(); crossed=np.zeros(n,bool)
 for s in surfs:
  cross=(~crossed)&(z<0.)&(0.<s.z)
  if np.any(cross):
   tt=(0.-z)/dz; sx=x+tt*dx; sy=y+tt*dy; valid&=(~cross)|(np.hypot(sx,sy)<=stop_radius*(1+1e-8)); crossed|=cross
  z,x,y,dz,dx,dy,oo,opl=_surf_step(z,x,y,dz,dx,dy,s,opl); valid&=oo
 xr,yr=ref; S=opl+np.sqrt((film-z)**2+(xr-x)**2+(yr-y)**2); S[~valid]=np.nan; return S,valid

def direct_otf_batch(surfs,lam,ax,fno,ref,seed,qmc_samples=QMC_SAMPLES):
 m=int(round(math.log2(qmc_samples)))
 if 2**m!=int(qmc_samples): raise ValueError('qmc_samples must be power of 2')
 base=qmc.Sobol(2,scramble=True,seed=seed).random_base2(m)*2-1; lm=lam*1e-6; arrays=[base]; meta=[]
 for f in FREQS:
  nu=lm*abs(fno)*f; d=2*nu
  if nu>=1: meta.append(None); continue
  idx=[]
  for axis in (0,1):
   shift=np.array([d/2,0.]) if axis==0 else np.array([0.,d/2]); ip=len(arrays); arrays.append(base+shift); im=len(arrays); arrays.append(base-shift); idx.append((ip,im))
  meta.append(idx)
 lens=[len(a) for a in arrays]; alluv=np.vstack(arrays); S,V=eikonal_batch(surfs,ax,alluv,ref); starts=np.cumsum([0]+lens)
 ss=[S[starts[i]:starts[i+1]] for i in range(len(arrays))]; vv=[V[starts[i]:starts[i+1]] for i in range(len(arrays))]; den=np.count_nonzero(vv[0]); ot={}; os={}
 if den==0: return {'otf_t':{},'otf_s':{},'pf':0.}
 for f,idx in zip(FREQS,meta):
  if idx is None: ot[str(f)]=0j; os[str(f)]=0j; continue
  vals=[]
  for ip,im in idx:
   good=vv[ip]&vv[im]; num=np.sum(np.exp(2j*np.pi*(ss[ip][good]-ss[im][good])/lm)) if np.any(good) else 0j; vals.append(num/den)
  ot[str(f)],os[str(f)]=vals
 n_circle=int(np.count_nonzero(np.sum(base*base,axis=1)<=1.)); return {'otf_t':ot,'otf_s':os,'pf':den/max(n_circle,1)}

def _nlog01(x,eps=MTF_EPS):
 xx=float(np.clip(x,0.,1.)); return -math.log((xx+eps)/(1.+eps))
def _pseudo_huber(x,delta=1.):
 x=float(x); delta=float(delta); return delta*delta*(math.sqrt(1.+(x/delta)**2)-1.)

def _evaluate_surfaces(design,gaps=None,qmc_samples=QMC_SAMPLES):
 if int(qmc_samples)!=QMC_SAMPLES: raise ValueError(f'v4 authoritative labels require exactly Q{QMC_SAMPLES}')
 mk=lambda lam:M.make_surfaces(design,lam,**({'gaps':gaps} if gaps is not None else {})); se=mk(WAVES[1]); efl,focus=M.cardinal(se); fm=M.front_matrix(se,0.)
 if fm is None or not math.isfinite(efl) or abs(float(fm[0,0]))<1e-12: return {'score':-1e9,'quality_score_100':0.,'merit_J':1e9,'config_hash':CONFIG_HASH,'error':'nonfinite_cardinal'}
 ent=(2*STOP_R)/abs(float(fm[0,0])); fno=abs(efl)/max(ent,1e-12); mtf_values=[]; dists=[]; lcas=[]; fields=[]; pf_by_field=[]
 for fi,ff in enumerate(FIELDS):
  ax=math.radians(MAX_FIELD_DEG)*ff; chiefs=[]; mono=[]
  for wi,lam in enumerate(WAVES):
   sw=mk(lam); c=chief_batch(sw,ax); chiefs.append(c)
   if c is None: mono.append(None); continue
   efll,_=M.cardinal(sw); fml=M.front_matrix(sw,0.)
   if fml is None or abs(float(fml[0,0]))<1e-12: mono.append(None); continue
   fnol=abs(efll)/max((2*STOP_R)/abs(float(fml[0,0])),1e-12); mono.append(direct_otf_batch(sw,lam,ax,fnol,c,QMC_SEED+1009*fi+37*wi,qmc_samples))
  cref=chiefs[1] if chiefs[1] is not None else (TARGET_EFL*math.tan(ax),0.); row={'field_fraction':ff,'mtf':{},'pupil_fill_by_wavelength':[]}
  for f in FREQS:
   tx=0j; sy=0j; ws=0.
   for wi,w in enumerate(WWEIGHTS):
    if mono[wi] is None or chiefs[wi] is None: continue
    ot=mono[wi]['otf_t'].get(str(f),0j); os_=mono[wi]['otf_s'].get(str(f),0j); dx=chiefs[wi][0]-cref[0]; dy=chiefs[wi][1]-cref[1]
    ot*=np.exp(-2j*np.pi*f*dx); os_*=np.exp(-2j*np.pi*f*dy); wt=w*mono[wi]['pf']; tx+=wt*ot; sy+=wt*os_; ws+=wt
   if ws>0: tx/=ws; sy/=ws
   t=float(np.clip(abs(tx),0.,1.)); s=float(np.clip(abs(sy),0.,1.)); row['mtf'][str(f)]={'t':t,'s':s,'mean':.5*(t+s),'floor':min(t,s)}; mtf_values.extend([t,s])
  pfs=[]
  for wi in range(len(WAVES)): pfs.append(float(np.clip(float(mono[wi]['pf']) if mono[wi] is not None else 0.,0.,1.)))
  row['pupil_fill_by_wavelength']=pfs; pf_by_field.append(pfs)
  if ff>0 and chiefs[1] is not None:
   ideal=abs(efl)*math.tan(ax)
   if abs(ideal)>1e-15: dists.append(abs(100*(chiefs[1][0]-ideal)/ideal))
  if all(c is not None for c in chiefs):
   xs=[c[0] for c in chiefs]; ys=[c[1] for c in chiefs]; lcas.append(math.hypot(max(xs)-min(xs),max(ys)-min(ys)))
  fields.append(row)
 vals=np.asarray(mtf_values,float)
 if vals.size:
  vals=np.clip(vals,0.,1.); logloss=np.asarray([_nlog01(x) for x in vals]); c_mtf_global=W_MTF_LOG_GLOBAL*float(logloss.mean()); k=max(1,int(math.ceil(MTF_TAIL_FRACTION*len(logloss)))); tail_loss=float(np.mean(np.partition(logloss,-k)[-k:])); c_mtf_tail=W_MTF_LOG_TAIL*tail_loss
  mtf_geomean=float(math.exp(float(np.mean(np.log(np.maximum(vals,MTF_EPS)))))); mtf_p10=float(np.quantile(vals,.10)); mtf_worst=float(vals.min()); mtf_mean=float(vals.mean())
 else: c_mtf_global=50.; c_mtf_tail=20.; mtf_geomean=mtf_p10=mtf_worst=mtf_mean=0.
 pfa=np.asarray(pf_by_field,float) if pf_by_field else np.zeros((len(FIELDS),len(WAVES))); on_by_wave=pfa[0] if len(pfa) else np.zeros(len(WAVES)); onaxis_pf=float(np.dot(np.asarray(WWEIGHTS),on_by_wave)); illum=[]
 for fi,ff in enumerate(FIELDS):
  rel=[w*float(pfa[fi,wi])/max(float(on_by_wave[wi]),ILLUM_EPS) for wi,w in enumerate(WWEIGHTS)]; illum.append(float(np.clip(sum(rel)*math.cos(math.radians(MAX_FIELD_DEG)*ff)**4,0.,1.)))
 mi=float(min(illum) if illum else 0.); c_onaxis=W_ONAXIS_PUPIL_LOG*_nlog01(onaxis_pf,ILLUM_EPS); c_illum=W_ILLUM_LOG*_nlog01(mi,ILLUM_EPS)
 dist_mean=float(np.mean(dists)) if dists else 100.; dist_max=float(max(dists)) if dists else 100.; ca=float(max(lcas)) if lcas else 1.
 c_dist=W_DIST*_pseudo_huber(dist_max/DIST_SCALE_PCT); c_lca=W_LCA*_pseudo_huber(ca/LCA_SCALE_MM)
 eef=(efl-TARGET_EFL)/(EFL_REL_SIGMA*TARGET_EFL); efn=(fno-TARGET_FNO)/(FNO_REL_SIGMA*TARGET_FNO); c_efl=W_EFL*_pseudo_huber(eef,CARDINAL_HUBER_DELTA); c_fno=W_FNO*_pseudo_huber(efn,CARDINAL_HUBER_DELTA)
 costs={'mtf_log_global':float(c_mtf_global),'mtf_log_lower_tail':float(c_mtf_tail),'onaxis_pupil_fill':float(c_onaxis),'relative_illumination':float(c_illum),'distortion_max':float(c_dist),'lateral_ca':float(c_lca),'efl':float(c_efl),'fno':float(c_fno)}
 J=float(sum(costs.values())); quality=float(100.*math.exp(-J/QUALITY_DISPLAY_SCALE)) if J<700 else 0.
 return {'score':-J,'quality_score_100':quality,'merit_J':J,'costs':costs,'efl':float(efl),'focus':float(focus),'focus_error':float(focus-FILM),'fno':float(fno),'mtf_mean':mtf_mean,'mtf_geomean_reg':mtf_geomean,'mtf_p10':mtf_p10,'mtf_worst':mtf_worst,'onaxis_pupil_fill':onaxis_pf,'relative_illumination':illum,'min_illum':mi,'dist':dist_max,'dist_max':dist_max,'dist_mean':dist_mean,'lca':ca,'fields':fields,'qmc_samples':int(qmc_samples),'config_hash':CONFIG_HASH}

def evaluate_final(design,qmc_samples=QMC_SAMPLES): return _evaluate_surfaces(design,None,qmc_samples)
def evaluate_final_with_gaps(design,gaps,qmc_samples=QMC_SAMPLES): return _evaluate_surfaces(design,tuple(float(x) for x in gaps),qmc_samples)

def self_test():
 a=evaluate_final(REF_DES,QMC_SAMPLES); assert a['config_hash']==CONFIG_HASH; assert abs(a['efl']-100.)<.02; assert 1.95<a['fno']<2.05; assert .90<=a['onaxis_pupil_fill']<=1.000001; assert _nlog01(0.)>_nlog01(.01)>_nlog01(.1)>_nlog01(.5)>_nlog01(1.)-1e-12; return {'reference':a,'config_hash':CONFIG_HASH}
if __name__=='__main__':
 t=time.time(); r=self_test(); r['seconds']=time.time()-t; print(json.dumps(r,indent=2))
