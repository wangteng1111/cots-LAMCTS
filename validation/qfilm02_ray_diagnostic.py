#!/usr/bin/env python3
"""Diagnostic ray-by-ray comparison between legacy PyTorch and fused Triton."""
from __future__ import annotations
import json,os,sys,torch,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluator.qfilm02 import _surfaces_at,trace_pupil,pupil_grid
from evaluator.qfilm02_triton import trace_wavelengths
from validation.qfilm02_smoke import thin_lens
from validation.qfilm02_accelerated import rich_lens
def execute(label,p):
 stop=float(p['stop_z_mm']); film=float(p['image_z_mm']);sd=p['design_spec']['stop_diameter_mm']/2.
 uv,inside=pupil_grid(16,'cuda:0')
 angle=torch.tensor([0.,0.035,0.15],device='cuda:0',dtype=torch.float64)
 waves=[510.,550.,630.]
 z,x,y,opl,fx,fy,valid=trace_wavelengths(p['surfaces'],waves,angle,uv,stop,sd,film)
 out={'name':label}
 for k,lam in enumerate(waves):
  s=_surfaces_at(p['surfaces'],lam)
  a=trace_pupil(s,angle[:,None],uv,stop,sd,film)
  chief=trace_pupil(s,angle[:,None],torch.zeros((1,2),device=angle.device,dtype=torch.float64),stop,sd,film)
  names=['z','x','y','opl','fx','fy']
  old=list(a[:6])
  fused=[z[k,:,:-1],x[k,:,:-1],y[k,:,:-1],opl[k,:,:-1],fx[k,:,:-1],fy[k,:,:-1]]
  compare={}
  for name,a0,b0 in zip(names,old,fused):
   dd=(a0-b0).abs()
   compare[name]={'max':float(dd.max()),'mean':float(dd.mean()),'reference_max':float(a0.abs().max()),
                  'samples':[[float(a0[i,j]),float(b0[i,j])] for i,j in [(0,0),(0,20),(1,50),(2,128)]]}
  compare['valid_mismatch']=int((a[6]!=valid[k,:,:-1]).sum().item())
  compare['chief_fx_max']=float((chief[4].flatten()-fx[k,:,-1]).abs().max())
  compare['chief_fy_max']=float((chief[5].flatten()-fy[k,:,-1]).abs().max())
  out[str(lam)]=compare
 print(json.dumps(out),flush=True)
 return out
def main():
 res=[execute('thin',thin_lens()),execute('four_surface_internal_stop',rich_lens())]
 p=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts'))/'qfilm02_ray_diagnostic.json'
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2)+'\n')
if __name__=='__main__':main()
