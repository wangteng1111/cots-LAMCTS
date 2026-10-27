#!/usr/bin/env python3
"""Pinned upstream DeepLens true execution versus Qfilm on an exactly shared optical prescription.
Do not read two unrelated demo lenses and compare their MTF.
"""
from __future__ import annotations
import argparse, json, os, sys, pathlib, time, traceback, math, copy
import numpy as np
import torch
REPO=pathlib.Path(__file__).resolve().parents[1]
DEEP=pathlib.Path('/var/lib/cots-lamcts/deeplens_upstream')
DEPS=pathlib.Path('/var/lib/cots-lamcts/deeplens_python_deps')
sys.path.insert(0,str(REPO));sys.path.insert(0,str(DEPS));sys.path.insert(0,str(DEEP))
from evaluator.qfilm02 import Config,evaluate as qfilm_eval,_surfaces_at
from validation.qfilm02_smoke import thin_lens
from validation.qfilm02_accelerated import rich_lens

SHA='911bb9d334fce7e20a4407c4506ac683b52ff246'
WAVELENGTH_UM=0.55
BK7={'B':[1.03961212,0.231792344,1.01046945],'C':[0.00600069867,0.0200179144,103.560653]}

def conversion(p, sensor_size=12.8, sensor_res=3072, material_mode='inline'):
    """Qfilm vertex coordinate → DeepLens forward sequential JSON (including stop)."""
    source=copy.deepcopy(p)
    src=source['surfaces']
    zs=[];z=0.
    for s in src:zs.append(z);z+=float(s['thickness'])
    stop_z=float(source['stop_z_mm']); film_z=float(source['image_z_mm'])
    offset=max(0.,-min(0.,stop_z))
    items=[];before='air'
    glass_names=[]
    for idx,s in enumerate(src):
        after='air' if float(s['n_after'])<=1.000001 else ('N-BK7' if material_mode=='bk7' and idx==0 else f"{s['n_after']:.6f}/{s['v_after']:.5f}")
        kw={'type':'Aspheric' if abs(float(s.get('conic',0.)))>1e-13 or any(abs(float(x))>0 for x in s.get('asphere',[])) else 'Spheric',
            'r':float(s.get('clear_aperture',2*source['design_spec']['stop_diameter_mm']))/2,
            'c':0. if math.isinf(float(s['radius'])) else 1./float(s['radius']),
            'roc':float(s['radius']),
            'mat1':before,'mat2':after,'d':zs[idx]+offset}
        if kw['type']=='Aspheric':kw.update({'k':float(s.get('conic',0.)),'ai':list(s.get('asphere',[])),'use_ai2':False})
        items.append((zs[idx]+offset,kw));before=after
        glass_names.append(after)
    items.append((stop_z+offset,{'type':'Aperture','r':float(source['design_spec']['stop_diameter_mm'])/2,
                   'c':0.,'mat1': 'air','mat2':'air','d':stop_z+offset}))
    # Proper glass on aperture is given by the immediately preceding refracting interface.
    items.sort(key=lambda x:x[0])
    medium='air'
    for i,(z,s) in enumerate(items):
        if s['type']=='Aperture':
            s['mat1']=medium;s['mat2']=medium
        else:medium=s['mat2']
        next_z=items[i+1][0] if i+1<len(items) else film_z+offset
        s['d_next']=float(next_z-z)
        if s['d_next']<-1e-9:raise ValueError('Invalid surface order')
    d={'info':'Alpha Lense DeepLens vs Qfilm physical prescription equivalence',
        'foclen':source['design_spec']['efl_target_mm'],
        'fnum':float(source['design_spec']['efl_target_mm'])/float(source['design_spec']['stop_diameter_mm']),
        'r_sensor':float(sensor_size)*0.70710678,
        'd_sensor':float(film_z+offset),
        'sensor_size':[float(sensor_size),float(sensor_size)],
        'sensor_res':[int(sensor_res),int(sensor_res)],
        'primary_wvln':WAVELENGTH_UM,
        'surfaces':[v for z,v in items]}
    return d,source

def mtf_from_psf(psf, pitch_mm, frequencies):
    p=np.asarray(psf,dtype=np.float64)
    assert p.ndim==2 and np.isfinite(p).all() and p.sum()>0
    p=p/p.sum()
    otf=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(p)))
    otf=otf/otf[p.shape[0]//2,p.shape[1]//2]
    fy=np.fft.fftshift(np.fft.fftfreq(p.shape[0],pitch_mm));fx=np.fft.fftshift(np.fft.fftfreq(p.shape[1],pitch_mm))
    m=np.abs(otf)
    yi=p.shape[0]//2; xi=p.shape[1]//2
    return [float(np.interp(freq,fx[xi:],m[yi,xi:])) for freq in frequencies],[float(np.interp(freq,fy[yi:],m[yi:,xi])) for freq in frequencies]

def safe_call(fn,**kwargs):
    start=time.perf_counter()
    try:
      with torch.inference_mode():value=fn(**kwargs)
      torch.cuda.synchronize() if torch.cuda.is_available() else None
      return value,{'ok':True,'seconds':time.perf_counter()-start}
    except Exception as e:return None,{'ok':False,'seconds':time.perf_counter()-start,'error':repr(e),'traceback':traceback.format_exc()[-4800:]}

def run_case(name, source, outpath, res=3072, ks=96, spp=4096, do_coherent=False):
    from deeplens import GeoLens
    data,p=conversion(source,sensor_res=res,material_mode='bk7' if name=='thin' else 'inline')
    jsonpath=outpath/f'{name}_deeplens_same_prescription.json';jsonpath.write_text(json.dumps(data,indent=2)+'\n')
    fileinfo={'name':name,'model_source':'identical Qfilm prescription converted to native DeepLens geometry',
              'deepjson':str(jsonpath),'sensor_res':res,'ks':ks,'spp_huygens':spp,
              'wavelength_um':WAVELENGTH_UM,'methods':{},'fields':[]}
    lens,linfo=safe_call(GeoLens,filename=str(jsonpath),device='cuda:0',dtype=torch.float64)
    fileinfo['init']=linfo
    if lens is None:print('CASE_INIT_FAIL',name,linfo['error'],flush=True);return fileinfo
    fileinfo['deep_focal_mm']=float(lens.foclen)
    fileinfo['sensor_pitch_mm']=float(lens.pixel_size)
    fileinfo['exit_pupil']=list(map(float,lens.get_exit_pupil()))
    conf=Config(fields=3,pupil_n=128,fft_n=512,max_fields_per_chunk=3,
            wavelengths_nm=(550.,),spectral_weights=(1.,),frame_width_mm=8.,frame_height_mm=6.,
            target_efl_mm=float(p['design_spec']['efl_target_mm']))
    qr,qinfo=safe_call(qfilm_eval,p=p,config=conf,device='cuda:0')
    fileinfo['qfilm_eval']=qinfo
    if qr is None:return fileinfo
    fileinfo['qfilm_efl_mm']=float(qr['efl_mm'])
    fileinfo['qfilm_Q_image'] = qr['Q_image'].tolist()
    fileinfo['qfilm_quality']=qr['quality']
    fqs=np.asarray(conf.frequencies_lpmm,dtype=float)
    image_radius=(8.**2+6.**2)**.5/2
    for i in range(3):
      h=i/2.
      alpha=math.atan(h*image_radius/float(conf.target_efl_mm))
      # Match physical object angle; DeepLens normalized points use its own focal length.
      normalized_y=math.tan(alpha)*float(lens.foclen)/(float(lens.sensor_size[1])/2)
      point=[0.,float(normalized_y),-1.e6]
      entry={'field_index':i,'h':h,'qfilm_input_angle_rad':alpha,'deeplens_object_point':point,
             'qfilm_mtf_t':qr['Q_image'][i,10:20].tolist(),
             'qfilm_mtf_s':qr['Q_image'][i,20:30].tolist(),
             'qfilm_chief_image_x_mm':float(qr['Q_image'][i,31])}
      psf,stats=safe_call(lens.psf_huygens,points=point,ks=ks,wvln=WAVELENGTH_UM,spp=spp,recenter=True)
      entry['huygens']=stats
      if psf is not None:
       ht,hs=mtf_from_psf(psf.detach().cpu().numpy(),float(lens.pixel_size),fqs)
       entry['huygens'].update({'mtf_t':ht,'mtf_s':hs,'sum':float(psf.sum()),
                      'max_abs_mtf_t_diff_vs_qfilm':float(np.max(np.abs(np.array(ht)-qr['Q_image'][i,10:20]))),
                      'max_abs_mtf_s_diff_vs_qfilm':float(np.max(np.abs(np.array(hs)-qr['Q_image'][i,20:30])))});
       np.savez_compressed(outpath/f'{name}_huygens_field{i}.npz',psf=psf.detach().cpu().numpy(),pitch_mm=float(lens.pixel_size))
      if do_coherent and i==0:
       coh,cs=safe_call(lens.psf_coherent,points=point,ks=ks,wvln=WAVELENGTH_UM,spp=1_000_000,recenter=True)
       entry['coherent']=cs
       if coh is not None:
        ct,css=mtf_from_psf(coh.detach().cpu().numpy(),float(lens.pixel_size),fqs)
        entry['coherent'].update({'mtf_t':ct,'mtf_s':css,'sum':float(coh.sum()),
                                'max_abs_mtf_t_diff_vs_qfilm':float(np.max(np.abs(np.array(ct)-qr['Q_image'][i,10:20]))),
                                'max_abs_mtf_t_diff_vs_huygens':float(np.max(np.abs(np.array(ct)-np.array(entry['huygens']['mtf_t'])))) if psf is not None else None})
        np.savez_compressed(outpath/f'{name}_coherent_field{i}.npz',psf=coh.detach().cpu().numpy(),pitch_mm=float(lens.pixel_size))
      fileinfo['fields'].append(entry)
      print('COMPARE',json.dumps({'name':name,'field':i,'huygens_s':stats.get('seconds'),
          'huygens_delta_mtf':entry.get('huygens',{}).get('max_abs_mtf_t_diff_vs_qfilm'),
          'coherent_s':entry.get('coherent',{}).get('seconds'),
          'coherent_ok':entry.get('coherent',{}).get('ok')}),flush=True)
    return fileinfo

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--coherent',action='store_true');ap.add_argument('--res',type=int,default=3072);ap.add_argument('--ks',type=int,default=96);ap.add_argument('--spp',type=int,default=4096);ap.add_argument('--cases',default='thin');args=ap.parse_args()
    out=pathlib.Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','/tmp/qfilm_deeplens_test'));out.mkdir(parents=True,exist_ok=True)
    result={'schema':'qfilm-deeplens-pinned-comparison-v1','source_sha':SHA,
            'torch_version':torch.__version__,'cases':[],'arguments':vars(args)}
    for name in args.cases.split(','):
      if name not in ('thin','complex'):raise ValueError('unknown case '+name)
      p=thin_lens() if name=='thin' else rich_lens()
      if name=='thin':p['surfaces'][0]['sellmeier']=BK7
      try:r=run_case(name,p,out,args.res,args.ks,args.spp,args.coherent)
      except Exception as e:r={'name':name,'fatal':repr(e),'traceback':traceback.format_exc()[-8000:]}
      result['cases'].append(r)
      (out/'deeplens_qfilm_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps({'cases':[(x['name'],x.get('fatal'),x.get('init',{}).get('error'),len(x.get('fields',[]))) for x in result['cases']]}),flush=True)
    return 0 if all(len(x.get('fields',[]))>0 for x in result['cases']) else 2
if __name__=='__main__':sys.exit(main())