#!/usr/bin/env python3
"""Admission proofs and independent off-axis verification failure injection."""
import copy,json,sys,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from validation import qfilm04_real11_supervised as r
from evaluator.qfilm04_sampling import ConvergencePolicy
from scripts.test_bridge_v06 import BridgeTests
from infra.qfilm_convergence_status import verify_pair

class MatrixTests(unittest.TestCase):
    def test_complete_six_grid_budget_is_required(self):
        policy=ConvergencePolicy();rows=r.budget_candidates(policy)
        affordable=[x for x in rows if x['minimum_pair_evaluations']<=policy.max_pair_evaluations]
        self.assertEqual(len(affordable),2)
        self.assertEqual({x['selected_n'] for x in affordable},{512})
        self.assertEqual(min(x['minimum_pair_evaluations'] for x in rows),132392366899200)
        self.assertGreater(min(x['minimum_pair_evaluations'] for x in rows if x['selected_n']>=1024),policy.max_pair_evaluations)

    def test_failed_required_phase_grid_rejects_before_any_full_integration(self):
        guard={'passed':False,'max_neighbor_block_bound_cycles':1.}
        with patch.object(r,'phase_guard',return_value=guard),patch.object(r,'prepare_wavefront_bundle') as prepare:
            d=r.admit({},550.,0.,ConvergencePolicy(),10.,(0.,0.),object(),lambda d:None)
        self.assertFalse(d['admitted']);prepare.assert_not_called()
        self.assertEqual(len(d['candidate_checks']),2)

    def test_admission_is_possible_with_safe_phase_and_sufficient_budget(self):
        with patch.object(r,'phase_guard',return_value={'passed':True}),patch.object(r,'prepare_wavefront_bundle',return_value=object()):
            d=r.admit({},550.,0.,ConvergencePolicy(),10.,(0.,0.),object(),lambda d:None)
        self.assertTrue(d['admitted'])
        self.assertNotIn('full_condition_verified',d)

    def test_offaxis_raw_data_verified_with_explicit_prescription_identity(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);BridgeTests().fixture(root)
            for backend in ('c','g'):
                path=root/f'ef40-blue-{backend}-terminal.json';data=json.loads(path.read_text())
                plan=data['resolved_sampling_plan'];plan['field_h']=.5
                plan['plan_hash']=hashlib.sha256(json.dumps({k:v for k,v in plan.items() if k!='plan_hash'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
                data['plan_hash']=plan['plan_hash'];path.write_text(json.dumps(data))
            from infra.qfilm_convergence_status import CASES
            expected=CASES['ef40-blue']
            self.assertTrue(verify_pair(root,'ef40-blue',expected=expected,field_h=.5)['verified'])
            with self.assertRaises(AssertionError):verify_pair(root,'ef40-blue',expected=expected,field_h=1.)

if __name__=='__main__':unittest.main()
