#!/usr/bin/env python3
import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from validation.qfilm04_real11_collect import collect,NAMES

class CollectorTests(unittest.TestCase):
    def test_missing_jobs_remain_231_missing_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            d=collect(Path(directory))
        self.assertEqual(d['native_condition_terminals'],0)
        self.assertEqual(d['verified_full_conditions'],0)
        self.assertEqual(sum(len(c['conditions']) for c in d['cases']),231)
        self.assertFalse(d['usable_for_full_matrix_numerical_comparison'])

    def test_summary_flag_cannot_replace_native_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs=Path(directory);name=NAMES[0];ident=name+'-h0-w430'
            art=jobs/f'000-000-013-qfilm06-zz-real11-00-{name}-20261030'/'artifacts';out=art/ident;out.mkdir(parents=True)
            sources=json.loads((ROOT/'validation/fixtures/qfilm04_real11_source_sha256.json').read_text())
            (out/'condition-terminal.json').write_text(json.dumps({'id':ident,'name':name,'field_h':0.,'wavelength_nm':430.,
                'source_sha256':sources['source_sha256'][name],'state':'numerically_converged_cpu_gpu_verified',
                'full_condition_verified':True,'arithmetic_tile':{'passed':True}}))
            d=collect(jobs);row=d['cases'][0]['conditions'][0]
            self.assertFalse(row['full_condition_verified']);self.assertFalse(row['arithmetic_pair_passed'])
            self.assertEqual(d['native_condition_terminals'],1)
            self.assertEqual(d['verified_full_conditions'],0)

    def test_wrong_prescription_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs=Path(directory);name=NAMES[0];ident=name+'-h0-w430'
            out=jobs/f'000-000-013-qfilm06-zz-real11-00-{name}-20261030'/'artifacts'/ident;out.mkdir(parents=True)
            (out/'condition-terminal.json').write_text(json.dumps({'id':ident,'name':name,'field_h':0.,'wavelength_nm':430.,
                'source_sha256':'wrong','state':'numerically_converged_cpu_gpu_verified','full_condition_verified':True}))
            d=collect(jobs)
            self.assertEqual(d['cases'][0]['conditions'][0]['state'],'CONDITION_IDENTITY_MISMATCH')
            self.assertEqual(d['verified_full_conditions'],0)

if __name__=='__main__':unittest.main()
