#!/usr/bin/env python3
"""Scalar exit-plane Huygens convergence vs pupil density and film sampling window."""
from __future__ import annotations
import argparse,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,torch
from evaluator.qfilm02_huygens import real_lens_reference,HuygensConfig,mtf_from_psf
from validation.qfilm02_smoke import thin_lens
from validation.qfilm02_accelerated import rich_lens
FREQS=(5.,10.,15.,20.,30.,40.,50.,60.,70.,80.)

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--device',default='cuda:0');a=pa.parse_args()
    report={'schema':'Qfilm0.2-Huygens-convergence-v1','cases':[],'certified':False}
    for label,lens,fi in [('thin',thin_lens(),0),('thin',thin_lens(),2),
                          ('complex',rich_lens(),0),('complex',rich_lens(),2)]:
        rows=[]
        # Fixed detector pitch: window difference exposes energy truncation.
        for n_pupil,n_image in [(64,192),(128,192),(128,384)]:
            t=time.perf_counter()
            try:
                h=real_lens_reference(lens,field_index=fi,field_total=3,pupil_n=n_pupil,
                           lam_nm=550.,device=a.device,
                           config=HuygensConfig(image_n=n_image,image_pitch_um=1.5,output_block=64))
                mtf=mtf_from_psf(h['psf'],h['pixel_pitch_mm'],FREQS).cpu().numpy()
                rows.append({'n_pupil':n_pupil,'n_image':n_image,'seconds':time.perf_counter()-t,
                             'mtf':mtf.tolist(),'mapping_ok':h['sufficient_mapping'],
                             'bad_mapping_fraction':h['bad_mapping_fraction']})
                print('CASE',label,fi,n_pupil,n_image,'seconds',rows[-1]['seconds'],flush=True)
            except Exception:
                rows.append({'n_pupil':n_pupil,'n_image':n_image,'error':traceback.format_exc()})
                print('FAIL',label,fi,n_pupil,n_image,flush=True)
        compared=[]
        if all('mtf' in r for r in rows):
            for l,r in [(rows[0],rows[1]),(rows[1],rows[2])]:
                e=np.abs(np.asarray(l['mtf'])-np.asarray(r['mtf']))
                compared.append({'lhs':f"{l['n_pupil']}p_{l['n_image']}i",'rhs':f"{r['n_pupil']}p_{r['n_image']}i",
                                 'mtf_max_abs':float(e.max()),'mtf_mean_abs':float(e.mean()),
                                 'mtf_p95_abs':float(np.quantile(e,.95))})
        datum={'name':label,'field':fi,'comparisons':compared,'rows':rows}
        report['cases'].append(datum)
        out=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'))/'qfilm02_huygens_convergence.json'
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
        print('RESULT',json.dumps({'name':label,'field':fi,'comparisons':compared}),flush=True)
if __name__=='__main__':main()