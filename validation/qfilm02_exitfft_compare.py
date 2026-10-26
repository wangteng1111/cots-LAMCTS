#!/usr/bin/env python3
"""Physical exit-plane gridded FFT vs Huygens vs old stop-grid FFT.

This is a scientific diagnostic; it deliberately does not call any method
certified when pupil/window convergence has not been demonstrated.
"""
from __future__ import annotations
import argparse,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch,numpy as np
from evaluator.qfilm02_exitfft import real_lens_exit_fft,mtf_exit_fft
from evaluator.qfilm02_huygens import real_lens_reference,mtf_from_psf,HuygensConfig
from evaluator.qfilm02 import Config,evaluate
from validation.qfilm02_smoke import thin_lens
from validation.qfilm02_accelerated import rich_lens
freq=(10.,20.,40.,60.)

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');a=p.parse_args()
 rows=[]
 for label,lens,fi in [('thin',thin_lens(),0),('thin',thin_lens(),2),
                      ('complex',rich_lens(),0),('complex',rich_lens(),2)]:
  r={'name':label,'field_i':fi}
  try:
   start=time.perf_counter()
   exitq=real_lens_exit_fft(lens,fi,3,64,550.,device=a.device)
   E=mtf_exit_fft(exitq,freq).cpu().numpy()
   r['exitfft_seconds']=time.perf_counter()-start
   r['exitfft_mtf_t']=E[:,0].tolist()
   start=time.perf_counter()
   hy=real_lens_reference(lens,fi,64,550.,HuygensConfig(image_n=192,image_pitch_um=1.5,output_block=96),a.device,3)
   H=mtf_from_psf(hy['psf'],hy['pixel_pitch_mm'],freq).cpu().numpy()
   r['huygens_seconds']=time.perf_counter()-start
   r['huygens_mtf_t']=H[:,0].tolist()
   c=Config(fields=3,pupil_n=64,fft_n=256,target_efl_mm=lens['design_spec']['efl_target_mm'],
            wavelengths_nm=(550.,),spectral_weights=(1.,),frequencies_lpmm=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.))
   q=evaluate(lens,c,device=a.device)
   L=q['Q_image'][fi,[11,13,15,17]]
   r['old_mtf_t']=L.tolist()
   r['exitfft_vs_huygens_max']=float(np.max(np.abs(E-H)))
   r['old_vs_huygens_max']=float(np.max(np.abs(L-H[:,0])))
   r['physical_bbox']=exitq['pupil_physical_bbox']
  except Exception as e:r['error']=traceback.format_exc()
  print('RESULT',json.dumps(r),flush=True);rows.append(r)
 out=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'))/'qfilm02_exitfft_compare.json'
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()