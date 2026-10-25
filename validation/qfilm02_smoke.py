#!/usr/bin/env python3
"""Reproducible Qfilm0.2 numerical verification and server benchmark."""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from evaluator.qfilm02 import Config, FREQS, area_weights, compact_json, evaluate, fft_pupil_metrics, pupil_grid, score_image, _sag_and_grad, VERSION


def assert_near(label, a, b, tol):
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    if not np.isfinite(err) or err>tol:
        raise AssertionError(f'{label}: actual_error={err:.6g} exceeds tolerance {tol}')
    return err


def analytic_tests(device):
    results={}
    n=64;uv,inside=pupil_grid(n,device)
    pupil=inside.reshape(1,n,n)
    zero=torch.zeros((1,n,n),dtype=torch.float64,device=device)
    psf,otf,pitch,strehl=fft_pupil_metrics(zero,pupil,550.,4.,nfft=256)
    nu=.00055*4.*np.asarray(FREQS)
    airy=2/np.pi*(np.arccos(nu)-nu*np.sqrt(1-nu*nu))
    results['airy_mtf_max_abs_error']=assert_near('Airy analytical OTF',otf[0,:,0].real.cpu().numpy(),airy,.015)
    results['unit_strehl_error']=assert_near('unaberrated Strehl',strehl.item(),1.,1e-9)
    results['energy_error']=assert_near('PSF energy',psf.sum().item(),1.,1e-10)
    x=uv[:,0].reshape(1,n,n);y=uv[:,1].reshape(1,n,n)
    # A pupil tilt leaves monochromatic MTF invariant (translation theorem).
    _,tilt_otf,_,_=fft_pupil_metrics(.0004*x,pupil,550.,4.,nfft=256)
    results['tilt_mtf_invariance']=assert_near('pupil tilt MTF',tilt_otf.abs().cpu().numpy(),otf.abs().cpu().numpy(),.015)
    defocus=2*(x*x+y*y)-1
    _,dpos,_,_=fft_pupil_metrics(.0008*defocus,pupil,550.,4.,nfft=256)
    _,dneg,_,_=fft_pupil_metrics(-.0008*defocus,pupil,550.,4.,nfft=256)
    results['defocus_sign_symmetry']=assert_near('±defocus magnitude',dpos.abs().cpu().numpy(),dneg.abs().cpu().numpy(),1e-8)
    # Conic sag sphere exactness, parabola k=-1 exact sag cr²/2.
    rr=torch.tensor([0.,1.,3.,6.],device=device,dtype=torch.float64)
    sag,grad,ok=_sag_and_grad(rr,30.,-1.,())
    results['parabola_sag_error']=assert_near('parabolic sag',sag.cpu().numpy(),rr.cpu().numpy()**2/60.,1e-11)
    results['parabola_slope_error']=assert_near('parabolic derivative',grad.cpu().numpy(),rr.cpu().numpy()/30.,1e-11)
    sag2,grad2,ok2=_sag_and_grad(rr,30.,0.,())
    sphere_expected=30.-np.sqrt(30.**2-rr.cpu().numpy()**2)
    results['spherical_sag_error']=assert_near('spherical sag',sag2.cpu().numpy(),sphere_expected,1e-11)
    results['spherical_slope_error']=assert_near('spherical derivative',grad2.cpu().numpy(),rr.cpu().numpy()/np.sqrt(30.**2-rr.cpu().numpy()**2),1e-11)
    # Circular vs rectangular field area histogram must be normalized and nonnegative.
    aw=area_weights(100,36,24)
    results['area_normalization_error']=assert_near('area weight sum',aw.sum(),1.,1e-12)
    if np.any(aw<0):raise AssertionError('negative area weights')
    # Absolute-MTF score must distinguish diffraction-limited f/2 from f/16.
    q=np.zeros((100,32));q[:,30]=1;q[:,1]=1;q[:,6]=0
    for fno in [2.,16.]:
        nu=np.asarray(FREQS)*.00055*fno
        t=np.clip(nu,0.,1.)
        mtf=np.where(t>=1.,0.,2/np.pi*(np.arccos(t)-t*np.sqrt(np.maximum(0.,1-t*t))))
        q[:,10:20]=mtf;q[:,20:30]=mtf
        results['ideal_fno'+str(int(fno))+'_J']=score_image(q,aw)['J_task']
    if results['ideal_fno2_J'] >= results['ideal_fno16_J']:
        raise AssertionError('diffraction absolute-MTF rank inverted')
    return results


def thin_lens():
    # Spherical biconvex lens in air. Film is fixed near green paraxial back focus.
    return {'surfaces':[
        {'radius':50.,'thickness':5.,'n_after':1.5168,'v_after':64.17,'clear_aperture':28.},
        {'radius':-50.,'thickness':47.55,'n_after':1.,'v_after':0.,'clear_aperture':28.}
    ],'stop_z_mm':-2.,'image_z_mm':52.55,
    'design_spec':{'stop_diameter_mm':8.,'efl_target_mm':50.}}


def physics_tests(device,full):
    conf=Config(fields=100 if full else 5,pupil_n=64 if full else 32,
                fft_n=256 if full else 128,
                max_fields_per_chunk=10 if full else 5,
                wavelengths_nm=(430.,470.,510.,550.,590.,630.,670.) if full else (510.,550.,630.),
                spectral_weights=(.075,.125,.16,.28,.16,.125,.075) if full else (.25,.5,.25))
    s=time.perf_counter()
    with torch.no_grad():r=evaluate(thin_lens(),conf,device=device)
    if device.startswith('cuda'):torch.cuda.synchronize()
    elapsed=time.perf_counter()-s
    q=r['Q_image'];wave=r['Q_wavefront']
    if q.shape!=(conf.fields,32) or wave.shape!=(conf.fields,16):raise AssertionError('wrong output shape')
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(wave)):raise AssertionError('nonfinite optical fields')
    if not np.all((q[:,10:30]>=0)&(q[:,10:30]<=1)):raise AssertionError('bad MTF range')
    if not r['quality']['valid']:raise AssertionError('optical fields unexpectedly invalid')
    if r['certified']:raise AssertionError('must remain provisional before Huygens validation')
    return {'n_fields':conf.fields,'n_waves':len(conf.wavelengths_nm),'n_pupil':conf.pupil_n**2,
            'seconds':elapsed,'fields_per_sec':conf.fields/elapsed,'J_task':r['quality']['J_task'],
            'central_mtf_t':q[0,10:20].tolist(),'edge_mtf_t':q[-1,10:20].tolist(),
            'min_pupil_fill':float(q[:,2].min()),'max_abs_distortion_pct':float(np.abs(q[:,3]).max()),
            'physics_status':r['physics_status']}


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--device',default='cpu');pa.add_argument('--full',action='store_true');pa.add_argument('--out',default=None)
    a=pa.parse_args()
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA device unavailable')
    record={'schema':'Qfilm0.2-verification-v1','version':VERSION,'device':a.device,'torch_version':torch.__version__,
            'gpu_name':torch.cuda.get_device_name(a.device) if a.device.startswith('cuda') else None,
            'certified':False}
    record['analytic']=analytic_tests(a.device)
    record['real_prescription_smoke']=physics_tests(a.device,a.full)
    if a.out:
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()