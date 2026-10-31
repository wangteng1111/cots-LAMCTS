#!/usr/bin/env python3
"""Read-only A/B/C operator diagnosis on one fixed ray bundle.

A: exact-distance scalar RS1 on source quadrature.
B: local first-order angular kernel on the SAME source quadrature, with the
   same common quadratic phase as the existing QfilmRW reference probe.
C: direct MFT of the reconstructed angular grid; CZT roundoff is intentionally
   excluded (its matrix identity is already tested separately).
This is a sampled field/intensity diagnosis, not a global PSF/OTF bound.
"""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from evaluator.qfilmrw.rays import trace_lens
from evaluator.qfilmrw.fourier import reference_source,reconstruct_pupil,direct_rs1
from evaluator.qfilm04_io import atomic_json

@torch.inference_mode()
def phase_sum(qx,qy,coefficient,xy,block=32768):
    out=torch.zeros(len(xy),dtype=torch.complex128,device=qx.device)
    for start in range(0,len(qx),block):
        sl=slice(start,start+block)
        phase=-2*math.pi*(xy[:,0,None]*qx[None,sl]+xy[:,1,None]*qy[None,sl])
        out+=(torch.exp(1j*phase)*coefficient[None,sl]).sum(-1)
    return out


def relative(a,b):
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-300))

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args()
    torch.set_num_threads(a.threads);cfg=json.loads(a.input.read_text());t0=time.monotonic()
    fixture=json.loads((ROOT/'validation/fixtures/qfilmrw_real11.json').read_text());case=next(c for c in fixture['cases'] if c['name']==cfg['name'])
    print('DECOMPOSE trace',cfg['id'],flush=True)
    state=trace_lens(case['prescription'],cfg['wavelength_nm'],cfg['field_h'],cfg['pupil_n'],'cpu',cfg['center_mm'])
    n=cfg['pixel_n'];pitch=cfg['pitch_um']*.001
    xy=torch.tensor([((ix-(n-1)/2)*pitch,(iy-(n-1)/2)*pitch) for iy,ix in cfg['pixel_yx']],dtype=torch.float64)
    qx,qy,coeff,piston,guard=reference_source(state)
    print('DECOMPOSE exact_RS1',flush=True)
    exact=direct_rs1(state,xy,piston)
    print('DECOMPOSE local_kernel_source_sum',flush=True)
    active=state.active
    direct_local=phase_sum(qx[active],qy[active],coeff[active],xy)
    L=state.film_z_mm-state.exit_z_mm;cx=float(state.x[active].median());cy=float(state.y[active].median())
    vx=state.center_mm[0]-cx;vy=state.center_mm[1]-cy;r=math.sqrt(L*L+vx*vx+vy*vy)
    hxx=1/r-vx*vx/r**3;hyy=1/r-vy*vy/r**3;hxy=-vx*vy/r**3
    quadratic=(hxx*xy[:,0]**2+2*hxy*xy[:,0]*xy[:,1]+hyy*xy[:,1]**2)/2
    restore=torch.exp(2j*math.pi*quadratic/(cfg['wavelength_nm']*1e-6))
    direct_local*=restore
    print('DECOMPOSE reconstruct',flush=True)
    pupil=reconstruct_pupil(state,cfg['angular_grid_n']);grid=pupil['grid'];ng=grid.shape[0]
    gx=pupil['q0'][0]+torch.arange(ng,dtype=torch.float64)*pupil['dq'][0]
    gy=pupil['q0'][1]+torch.arange(ng,dtype=torch.float64)*pupil['dq'][1]
    qgy,qgx=torch.meshgrid(gy,gx,indexing='ij')
    print('DECOMPOSE angular_grid_MFT',flush=True)
    gridded=phase_sum(qgx.flatten(),qgy.flatten(),grid.flatten(),xy)*restore
    # Comparisons use a common reference norm so they obey a triangle bound,
    # but the vector errors can cancel and must not be reported as fractions
    # that sum to 100 percent.
    den=torch.linalg.vector_norm(exact).clamp_min(1e-300)
    pstats={k:v for k,v in pupil.items() if k not in ('grid',)}
    out={'schema':'QfilmRWKernelErrorDecomposition-v1','input':cfg,'device':'cpu','torch_version':str(torch.__version__),
      'source_sha256':case['source_sha256'],'active_nodes':int(active.sum()),'phase_neighbor_cycles':guard,
      'A_vs_B_kernel_field_relative_L2':float(torch.linalg.vector_norm(exact-direct_local)/den),
      'B_vs_C_reconstruction_quadrature_field_relative_L2_on_A_norm':float(torch.linalg.vector_norm(direct_local-gridded)/den),
      'A_vs_C_total_field_relative_L2':float(torch.linalg.vector_norm(exact-gridded)/den),
      'A_vs_B_sampled_intensity_relative_L2':relative(exact.abs().square(),direct_local.abs().square()),
      'A_vs_C_sampled_intensity_relative_L2':relative(gridded.abs().square(),exact.abs().square()),
      'coordinates_offset_mm':xy.numpy(),'A_exact_field':exact.numpy(),'B_local_source_field':direct_local.numpy(),'C_local_grid_field':gridded.numpy(),
      'pupil':pstats,'elapsed_seconds':time.monotonic()-t0,
      'scope':'same scalar boundary state and finite source/grid quadrature; sampled probes only; not global MTF or measured-lens validation',
      'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((ROOT/'evaluator/qfilmrw').glob('*.py'))]}}
    # For intensity B-A use A as denominator, consistently with the C-A metric.
    out['A_vs_B_sampled_intensity_relative_L2']=relative(direct_local.abs().square(),exact.abs().square())
    atomic_json(a.output,out,immutable=True)
    print(json.dumps({k:v for k,v in out.items() if k.endswith('L2') or k.endswith('A_norm') or k=='elapsed_seconds'}),flush=True)
if __name__=='__main__':main()
