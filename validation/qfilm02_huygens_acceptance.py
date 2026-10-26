#!/usr/bin/env python3
"""Independent scalar Huygens numerical validation for Qfilm0.2 optical physics.

The analytic converging sphere and synthetic real-lens comparisons are separate:
PASS of the analytic test is NOT certification of the current Qfilm FFT.
"""
from __future__ import annotations
import argparse,json,math,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from evaluator.qfilm02_huygens import HuygensConfig,huygens_psf_exit_plane,mtf_from_psf,real_lens_reference
from evaluator.qfilm02 import Config,evaluate
from validation.qfilm02_smoke import thin_lens
from validation.qfilm02_accelerated import rich_lens

FREQ=(10.,20.,40.,60.)

def ideal_sphere(device):
    n=64;R=5.;L=50.;lambda_nm=550.
    u=(torch.arange(n,device=device,dtype=torch.float64)+.5)*(2./n)-1.
    vv,uu=torch.meshgrid(u,u,indexing='ij')
    mask=(uu*uu+vv*vv<=1.)
    xe=uu*R;ye=vv*R
    op=-torch.sqrt(L*L+xe*xe+ye*ye)
    q=huygens_psf_exit_plane(xe,ye,op,mask,lambda_nm,2./n,L,0.,
          config=HuygensConfig(image_n=128,image_pitch_um=1.,output_block=128))
    measured=mtf_from_psf(q['psf'],q['pixel_pitch_mm'],FREQ)[:,0].cpu().numpy()
    nu=np.asarray(FREQ)*lambda_nm*1e-6*L/(2*R)
    airy=(2/np.pi)*(np.arccos(nu)-nu*np.sqrt(1-nu*nu))
    error=np.max(np.abs(measured-airy))
    out={'mode':'analytic_converging_sphere','n_pupil':n,'n_image':128,'mtf':measured.tolist(),
         'analytic_airy_mtf':airy.tolist(),'max_abs_error':float(error),
         'mapping_ok':q['sufficient_mapping'],'passed':bool(error<.025)}
    print('IDEAL',json.dumps(out),flush=True)
    return out

def sample_real(device,which,field_index):
    p=thin_lens() if which=='thin' else rich_lens()
    field_count=3;lam=550.
    c=Config(fields=field_count,pupil_n=64,fft_n=256,wavelengths_nm=(lam,),spectral_weights=(1.,),
             target_efl_mm=p['design_spec']['efl_target_mm'],max_fields_per_chunk=3,
             frequencies_lpmm=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.))
    t=time.perf_counter()
    a=evaluate(p,c,device=device)
    t_ff=time.perf_counter()-t
    t=time.perf_counter()
    h=real_lens_reference(p,field_index=field_index,field_total=field_count,
                          lam_nm=lam,pupil_n=64,device=device,
                          config=HuygensConfig(image_n=192,image_pitch_um=1.5,output_block=96))
    with torch.no_grad():
        hm=mtf_from_psf(h['psf'],h['pixel_pitch_mm'],c.frequencies_lpmm).cpu().numpy()
    qt=a['Q_image'][field_index,10:20]
    qs=a['Q_image'][field_index,20:30]
    difft=np.abs(hm[:,0]-qt)
    diffs=np.abs(hm[:,1]-qs)
    out={'name':which,'field_index':field_index,'field_h':float(field_index/(field_count-1)),
         'fft_sec':t_ff,'huygens_sec':time.perf_counter()-t,'fft_mtf_t':qt.tolist(),
         'huygens_mtf_t':hm[:,0].tolist(),'max_t_mtf_abs':float(difft.max()),
         'mean_t_mtf_abs':float(difft.mean()),'max_s_mtf_abs':float(diffs.max()),
         'bad_mapping_fraction':h['bad_mapping_fraction'],
         'sufficient_mapping':h['sufficient_mapping'],
         'certified':False}
    print('REAL',json.dumps(out),flush=True)
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--out',default=None)
    args=parser.parse_args();dev=args.device
    r={'schema':'Qfilm0.2-Huygens-reference-v1','device':dev,'physics_certified':False,'analytic':None,'real':[]}
    r['analytic']=ideal_sphere(dev)
    cases=[('thin',0),('thin',2),('complex',0),('complex',2)]
    for which,fi in cases:
        try:r['real'].append(sample_real(dev,which,fi))
        except Exception as e:
            r['real'].append({'name':which,'field_index':fi,'error':repr(e),'traceback':traceback.format_exc()})
            print('ERROR',which,fi,repr(e),flush=True)
    out=Path(args.out or (Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'))/'qfilm02_huygens_reference.json'))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n')
    print('SUMMARY',json.dumps({'ideal_pass':r['analytic']['passed'],'real_cases':len(r['real'])}),flush=True)
    return 0 if r['analytic']['passed'] else 2
if __name__=='__main__':raise SystemExit(main())