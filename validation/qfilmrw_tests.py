#!/usr/bin/env python3
"""QfilmRW numerical identities, analytic optics, failure/label boundaries."""
import sys,unittest,math,tempfile,json,copy
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from evaluator.qfilmrw import Task,Policy,assemble_score
from evaluator.qfilmrw.fourier import czt_axis,fixed_sampling_field,windowed_observables,pupil_autocorrelation
from evaluator.qfilmrw.solver import polychromatic_otf
from validation.qfilmrw_real11 import cases,collect,check_condition
from evaluator.qfilmrw.rays import RayState
from evaluator.qfilmrw.fourier import reconstruct_pupil
from evaluator.qfilmrw.contracts import digest
from evaluator.qfilmrw import solve_condition

torch.set_num_threads(2)

class RWTests(unittest.TestCase):
    def test_inverse_map_preserves_constant_angular_density(self):
        n=64;du=2/n;u=(torch.arange(n,dtype=torch.float64)+.5)*du-1
        v,u=torch.meshgrid(u,u,indexing='ij');qx=10*u+v;qy=2*u+9*v
        lam=.00055;L=10.;r=L/torch.sqrt(1-lam**2*(qx*qx+qy*qy))
        x=lam*qx*r;y=lam*qy*r
        amp=88*du**2*lam*r*r/L;active=u*u+v*v<.8**2
        state=RayState(x,y,-r,torch.ones_like(x),amp,torch.ones_like(x),active,0.,L,550.,0.,(0.,0.),{})
        pupil=reconstruct_pupil(state,96);grid=pupil['grid'];good=grid.abs()>0
        self.assertGreater(int(good.sum()),1000)
        expected=pupil['dq'][0]*pupil['dq'][1]
        self.assertLess(float((grid.real[good]/expected-1).abs().max()),1e-9)

    def test_working_set_limit_precedes_transform(self):
        with patch('evaluator.qfilmrw.solver.trace_lens',return_value=SimpleNamespace(center_mm=(0.,0.))),patch('evaluator.qfilmrw.solver.reconstruct_pupil') as transform:
            result=solve_condition({},550.,0.,policy=Policy(max_live_elements=1024))
        self.assertEqual(result['state'],'RESOURCE_LIMIT');transform.assert_not_called()

    def test_bluestein_matches_explicit_nontrivial_origins(self):
        torch.manual_seed(9);x=torch.randn(3,17,dtype=torch.complex128)
        q0,dq,x0,dx,m=-3.7,.17,-.23,.011,23
        q=q0+torch.arange(17,dtype=torch.float64)*dq;t=x0+torch.arange(m,dtype=torch.float64)*dx
        direct=x@torch.exp(-2j*torch.pi*q[:,None]*t[None,:])
        self.assertLess(float((czt_axis(x,m,q0,dq,x0,dx)-direct).abs().max()),1e-12)

    def test_two_axis_mft_centered_coordinates(self):
        torch.manual_seed(2);g=torch.randn(13,13,dtype=torch.complex128);q0=(-.8,-.6);dq=(.12,.14);m=16;pitch=.25
        x=(torch.arange(m,dtype=torch.float64)-(m-1)/2)*pitch*.001
        qx=q0[0]+torch.arange(13,dtype=torch.float64)*dq[0];qy=q0[1]+torch.arange(13,dtype=torch.float64)*dq[1]
        direct=torch.exp(-2j*torch.pi*x[:,None]*qy)@g@torch.exp(-2j*torch.pi*qx[:,None]*x)
        got=fixed_sampling_field(g,q0,dq,m,pitch)
        self.assertLess(float((got-direct).abs().max()),1e-11)

    def test_airy_fixed_quarter_micron(self):
        n=384;dq=1.1;q0=-(n-1)*dq/2
        q=q0+torch.arange(n,dtype=torch.float64)*dq;yy,xx=torch.meshgrid(q,q,indexing='ij')
        lam=.00055;fno=5.;cutoff=1/(lam*fno)
        pupil=((xx*xx+yy*yy)<(cutoff/2)**2).to(torch.complex128)*dq*dq
        field=fixed_sampling_field(pupil,(q0,q0),(dq,dq),2560,.25)
        freqs=(0.,5.,20.,40.,80.)
        obs=windowed_observables(field,.25,freqs)
        v=np.array(freqs)/cutoff;analytic=2/np.pi*(np.arccos(v)-v*np.sqrt(1-v*v))
        self.assertLess(float(np.max(np.abs(obs['otf'].real-analytic[:,None]))),.003)
        self.assertLess(float(np.max(np.abs(obs['otf'].imag))),1e-10)
        self.assertAlmostEqual(obs['support_mm'],.64)

    def test_autocorrelation_dc_and_outside_cutoff(self):
        g=torch.ones((32,32),dtype=torch.complex128)
        out=pupil_autocorrelation({'grid':g,'dq':(2.,2.)},(0.,16.,80.))
        np.testing.assert_allclose(out,[[1,1],[.75,.75],[0,0]],atol=1e-12)

    def test_autocorrelation_retains_signed_phase(self):
        q=torch.arange(32,dtype=torch.float64)*2.;shift=.003
        g=torch.exp(2j*torch.pi*q[None,:]*shift).expand(32,-1)
        out=pupil_autocorrelation({'grid':g,'dq':(2.,2.)},(16.,))
        self.assertLess(abs(out[0,0]-.75*np.exp(-2j*np.pi*16*shift)),1e-12)

    def test_polychromatic_sensor_centers_are_not_erased(self):
        samples=[{'otf':np.ones((1,2)),'center_film_mm':(0.,0.)},{'otf':np.ones((1,2)),'center_film_mm':(.025,0.)}]
        out=polychromatic_otf(samples,(20.,),(1.,1.),(0.,0.))
        self.assertLess(abs(out[0,0]),1e-12);self.assertAlmostEqual(out[0,1].real,1.)

    def test_unvalidated_output_cannot_make_training_J(self):
        self.assertIsNone(assemble_score([{'numerically_verified':True,'model_scope_validated':False}])['J_task'])

    def test_task_rejects_ambiguous_normalization(self):
        with self.assertRaises(ValueError):Task(otf_domain='implicit').validate()
        with self.assertRaises(ValueError):Task(pitch_um=0).validate()
        with self.assertRaises(ValueError):Policy(pupils=(256,128)).validate()

    def test_same_eleven_source_identities(self):
        self.assertEqual(len(cases()),11)

    def test_missing_workstation_results_never_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            result=collect(Path(directory))
        self.assertEqual(result['condition_count'],231)
        self.assertEqual(result['full_conditions_accepted'],0)
        self.assertEqual(result['arithmetic_pairs_verified'],0)

    def test_verifier_rejects_tampered_projection(self):
        task=Task(support_mm=.004);p={'task':task.__dict__,'policy':Policy().__dict__}
        obs=windowed_observables(torch.ones((16,16),dtype=torch.complex128),.25,task.frequencies)
        obs['axis']='pupil'
        from evaluator.qfilm04_io import json_safe
        record=json_safe({'plan':p,'plan_hash':digest(p),'runs':[obs],'comparisons':[],'training_eligible':False,'J_task':None})
        check_condition(record)
        record['runs'][0]['raw_projection_x'][0]*=2
        with self.assertRaises(ValueError):check_condition(record)

    @unittest.skipUnless(torch.cuda.is_available(),'local CUDA unavailable; requires workstation')
    def test_actual_gpu_czt_parity(self):
        torch.manual_seed(8);g=torch.randn(31,31,dtype=torch.complex128)
        cpu=fixed_sampling_field(g,(-20.,-19.),(.7,.8),48,.25)
        for i in range(torch.cuda.device_count()):
            gpu=fixed_sampling_field(g.to(f'cuda:{i}'),(-20.,-19.),(.7,.8),48,.25).cpu()
            self.assertLess(float((cpu-gpu).abs().max())/float(cpu.abs().max()),1e-11)

if __name__=='__main__':unittest.main(verbosity=2)
