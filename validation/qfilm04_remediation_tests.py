#!/usr/bin/env python3
"""Mathematical controls and failure-path regressions for the new Qfilm path."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataclasses import replace
import json, tempfile, unittest, math
import numpy as np
import torch
from unittest.mock import patch
from evaluator.qfilm04_wavefront import (otf_from_projections,signed_mapping,
    WavefrontBundle,integrate_fixed_film,fixed_exit_plane,prepare_wavefront_bundle)
from evaluator.qfilm04_sampling import (ConvergencePolicy,phase_guard,make_plan,
    solve_condition,combine_spectral_otf,validate_plan,plan_identity)
from evaluator.qfilm04_io import atomic_json
from validation.qfilm02_smoke import thin_lens


class RemediationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_complex_extractor_against_two_dimensional_sum(self):
        rng=np.random.default_rng(87)
        for n in (17,32):
            image=rng.random((n,n))**2;pitch=.003
            f=np.array([0.,5.25,11.,63.1]);x=(np.arange(n)-(n-1)/2)*pitch
            actual=otf_from_projections(image.sum(0),image.sum(1),pitch,f)
            exact=np.array([[np.sum(image*np.exp(-2j*np.pi*v*x)[None,:]),
                             np.sum(image*np.exp(-2j*np.pi*v*x)[:,None])] for v in f])/image.sum()
            np.testing.assert_allclose(actual,exact,rtol=0,atol=1e-14)

    def test_two_color_physical_translation(self):
        n=81;pitch=.001;f=np.array([0.,5.7,20.,70.])
        x=(np.arange(n)-(n-1)/2)*pitch
        # Integer pixel offsets produce exact discrete translations with no cropping.
        p=np.zeros(n);p[38:43]=[1,2,5,3,1]
        a=otf_from_projections(p,p,pitch,f)
        b=otf_from_projections(p,p,pitch,f)
        result=combine_spectral_otf([{"otf":a,"center_film_mm":[0,0]},
            {"otf":b,"center_film_mm":[.007,-.004]}],[550,670],[.3,.7],f)
        exact=otf_from_projections(.3*p+.7*np.roll(p,7),.3*p+.7*np.roll(p,-4),pitch,f)
        np.testing.assert_allclose(result,exact,rtol=0,atol=2e-15)

    def test_signed_mapping_orientation_and_fold(self):
        y,x=torch.meshgrid(torch.linspace(-1,1,33),torch.linspace(-1,1,33),indexing="ij")
        valid=torch.ones_like(x,dtype=torch.bool)
        j,good,fold,_=signed_mapping(-x,y,valid,2/32)
        self.assertFalse(fold);self.assertTrue(bool((j[good]<0).all()))
        _,_,fold,_=signed_mapping(x*x+.01*x,y,valid,2/32)
        self.assertTrue(fold)
        _,_,fold,_=signed_mapping(x*x,y,valid,2/32)
        self.assertTrue(fold)

    def test_block_phase_bound_contains_random_points(self):
        rng=np.random.default_rng(8)
        for _ in range(500):
            a,b=rng.normal(size=(2,2));L=rng.uniform(.1,40);rho=rng.uniform(.001,1)
            c=rng.normal(size=2);opl=rng.normal();lam=.00055
            def val(q):return (opl+np.sqrt(L*L+sum((q-b)**2))-np.sqrt(L*L+sum((q-a)**2)))/lam
            bound=abs(val(c))+min(2,np.linalg.norm(a-b)/L)*rho/lam
            theta=rng.uniform(0,2*np.pi);r=rho*np.sqrt(rng.random())
            self.assertLessEqual(abs(val(c+r*np.array([np.cos(theta),np.sin(theta)]))),bound+1e-10)

    def test_raw_projection_area_and_direct_kernel(self):
        n=32;u=(torch.arange(n,dtype=torch.float64)+.5)*2/n-1
        y,x=torch.meshgrid(u,u,indexing="ij");a=x*x+y*y<1
        X=x*.5;Y=y*.5;L=30.;opl=-torch.sqrt(L*L+X*X+Y*Y)
        uv=torch.stack([x.flatten(),y.flatten()],-1)
        bundle=WavefrontBundle(X,Y,opl,torch.ones_like(x)*(1/n)**2,a,
            torch.ones_like(x),torch.ones_like(x),uv,0.,L,550.,0.,(0.,0.),{})
        r=integrate_fixed_film(bundle,24,1.,[5.,20.,70.],pixel_block=7,retain_intensity=True)
        image=r["raw_intensity"]
        np.testing.assert_allclose(r["raw_projection_x"],image.sum(0),rtol=2e-15)
        np.testing.assert_allclose(r["raw_projection_y"],image.sum(1),rtol=2e-15)
        self.assertAlmostEqual(r["raw_intensity_integral"],float(image.sum())*1e-6,places=12)
        # Independently evaluate the legacy spherical kernel at a single pixel.
        px=py=-11.5e-3;lam=.00055;xx=X[a].numpy();yy=Y[a].numpy();oo=opl[a].numpy()
        distance=np.sqrt(L*L+(px-xx)**2+(py-yy)**2)
        E=np.sum(np.exp(2j*np.pi*np.remainder((oo+distance-oo[0]-L)/lam,1))/(n*n*distance*lam))
        self.assertAlmostEqual(image[0,0]/abs(E)**2,1.,places=7)

    def small(self,budget=1):
        p=thin_lens()
        # Explicit clear apertures are required by the physical surface envelope.
        for s in p["surfaces"]:s["clear_aperture"]=10.
        policy=ConvergencePolicy(pupil_candidates=(16,32),support_candidates_mm=(.016,.032),
            pitch_um=1.,phase_tiles_per_axis=2,midpoint_pairs=8,max_pair_evaluations=budget)
        plan=make_plan(p,550.,0.,[5.,20.],policy)
        return p,policy,plan

    def test_plan_source_and_policy_tamper_rejected(self):
        p,policy,plan=self.small()
        validate_plan(p,plan,policy,(36.,24.))
        changed=json.loads(json.dumps(plan));changed["center_mm"][0]+=.1
        with self.assertRaises(ValueError):validate_plan(p,changed,policy,(36.,24.))
        with self.assertRaises(ValueError):validate_plan(p,plan,replace(policy,pitch_um=.5),(36.,24.))

    def test_resource_exhaustion_never_certifies(self):
        p,policy,plan=self.small()
        # Exercise resource handling independently of whether this low-N toy passes phase.
        for l in plan["levels"]:
            for g in l["phase_guards"]:g["passed"]=True
        plan["plan_hash"]=plan_identity(plan)
        report=solve_condition(p,plan,policy)
        self.assertEqual(report["state"],"RESOURCE_LIMIT")
        self.assertFalse(report["flags"]["label_valid"])
        self.assertFalse(report["training_eligible"])
        self.assertEqual(len(report["runs"]),0)
        replay=solve_condition(p,report["resolved_sampling_plan"],policy)
        self.assertEqual(replay["state"],"RESOURCE_LIMIT")
        self.assertEqual(replay["plan_hash"],report["plan_hash"])
        self.assertTrue(replay["frozen_schedule_replay"])

    def test_immutable_terminal_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"terminal.json"
            atomic_json(p,{"state":"complete","missing":np.nan,"otf":np.array([1+2j])},immutable=True)
            data=json.loads(p.read_text());self.assertIsNone(data["missing"])
            self.assertEqual(data["otf"][0],{"real":1.,"imag":2.})
            with self.assertRaises(ValueError):atomic_json(p,{"state":"running"},immutable=True)

    def test_second_execution_replays_actual_first_device_grids(self):
        p,policy,plan=self.small(budget=100_000_000)
        # Small physical stop gives a phase-resolved low-cost control. Keep the
        # production phase threshold; do not force a failed guard to pass.
        p["design_spec"]["stop_diameter_mm"]=.2
        plan=make_plan(p,550.,0.,[5.,20.],policy)
        first=solve_condition(p,plan,policy)
        self.assertGreater(len(first["runs"]),0)
        frozen=first["resolved_sampling_plan"]
        second=solve_condition(p,frozen,policy)
        self.assertEqual(first["state"],second["state"])
        self.assertEqual(first["plan_hash"],second["plan_hash"])
        self.assertNotEqual(first["execution_id"],second["execution_id"])
        self.assertEqual(len(first["runs"]),len(second["runs"]))
        for a,b in zip(first["runs"],second["runs"]):
            np.testing.assert_array_equal(a["otf"],b["otf"])

    def test_no_gpu_fallback(self):
        p,policy,plan=self.small()
        with patch("torch.cuda.is_available",return_value=False):
            with self.assertRaises(RuntimeError):solve_condition(p,plan,policy,device="cuda:0")

    def test_final_evaluator_never_returns_stale_fft_mtf(self):
        from evaluator.qfilm04 import evaluate,PhysicalHuygensConfig
        from evaluator.qfilm02 import Config
        p=thin_lens()
        for s in p["surfaces"]:s["clear_aperture"]=10.
        cfg=Config(fields=2,pupil_n=16,fft_n=32,wavelengths_nm=(550.,),spectral_weights=(1.,),
                   target_efl_mm=p["design_spec"]["efl_target_mm"])
        phy=PhysicalHuygensConfig(detector_n=2560,detector_pitch_um=.25)
        policy=ConvergencePolicy(pupil_candidates=(256,512),support_candidates_mm=(.64,.96),
                                 midpoint_pairs=8,phase_tiles_per_axis=1,max_pair_evaluations=1)
        result=evaluate(p,cfg,phy,"c",convergence=policy)
        self.assertFalse(result["quality"]["valid"])
        self.assertIsNone(result["quality"]["J_task"])
        self.assertTrue(np.isnan(result["Q_image"][:,10:30]).all())
        self.assertFalse(result["training_eligible"])


if __name__=="__main__":unittest.main(verbosity=2)
