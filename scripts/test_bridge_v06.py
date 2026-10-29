#!/usr/bin/env python3
"""Failure injection: offline supervision, monotonic outbox, real child exit."""
import base64,importlib,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
TEST_ROOT=tempfile.TemporaryDirectory(prefix='bridge-v06-')
os.environ['COTS_BRIDGE_ROOT']=TEST_ROOT.name
sys.path.insert(0,str(ROOT/'infra'))
import cots_workstation_bridge_v06 as b
import qfilm_convergence_status as q

class BridgeTests(unittest.TestCase):
    def setUp(self):
        for p in b.OUTBOX.glob('*.json'):p.unlink()
        (b.v3.STATE/'publish_watermarks_v06.json').unlink(missing_ok=True)

    def test_latest_observation_coalesces(self):
        p='remote/agent_status.json'
        for stamp in (30,10,20):b.enqueue(p,json.dumps({'timestamp':stamp}).encode(),'heartbeat')
        items=list(b.OUTBOX.glob('*.json'));self.assertEqual(len(items),1)
        self.assertEqual(json.loads(base64.b64decode(json.loads(items[0].read_text())['data_b64']))['timestamp'],30)

    def test_terminal_cannot_regress_to_running(self):
        p='remote/results/test.json'
        b.enqueue(p,b'{"state":"finished","finished_at":30}','final')
        b.enqueue(p,b'{"state":"running","updated_at":40}','old running')
        self.assertEqual(json.loads(list(b.OUTBOX.glob('*.json'))[0].read_text())['freshness'][0],1)

    def test_offline_enqueue_preserves_payload(self):
        with patch.object(b,'ORIGINAL_PUT',side_effect=OSError('offline')),patch.object(b.v3,'gh_content',return_value=None):
            b.enqueue('remote/results/test.json',b'{"updated_at":1}','test')
            with self.assertRaises(OSError):b.publish_one()
            self.assertEqual(len(list(b.OUTBOX.glob('*.json'))),1)

    def test_remote_newer_heartbeat_not_overwritten(self):
        b.enqueue('remote/agent_status.json',b'{"timestamp":1}','test')
        with patch.object(b.v3,'gh_content',return_value={'content':base64.b64encode(b'{"timestamp":2}').decode()}),patch.object(b,'ORIGINAL_PUT') as upload:
            self.assertTrue(b.publish_one());upload.assert_not_called()

    def test_immutable_final_is_not_overwritten(self):
        b.enqueue('remote/finals/test.json',b'{"state":"failed","finished_at":2}','test')
        with patch.object(b.v3,'gh_content',return_value={'content':base64.b64encode(b'{"state":"finished","finished_at":1}').decode()}),patch.object(b,'ORIGINAL_PUT') as upload:
            b.publish_one();upload.assert_not_called()

    def test_receipt_survives_network_failure(self):
        jid='bridge-v06-claimed-test'
        b.atomic(b.v5._receipt_path(jid),{'job_id':jid,'state':'bridge_error'})
        with patch.object(b.v3,'gh_content',side_effect=OSError('offline')):
            self.assertTrue(b.cached_claimed(jid))

    def test_worker_persists_actual_failure(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT.name) as directory:
            job=Path(directory);src=job/'src';(src/'validation').mkdir(parents=True)
            (src/'validation/failure.py').write_text('raise SystemExit(7)\n')
            request=job/'worker-request.json';request.write_text(json.dumps({'job_id':'worker-failure-test','entrypoint':'validation/failure.py','args':[],'timeout_sec':10,'code_commit':'a'*40}))
            r=subprocess.run([sys.executable,str(ROOT/'infra/cots_bridge_worker.py'),str(request),str(src),sys.executable],timeout=15)
            self.assertEqual(r.returncode,0);self.assertEqual(json.loads((job/'worker-exit.json').read_text())['returncode'],7)

    def test_supervisor_keeps_polling_during_network_failure(self):
        # Execute a real child through the actual supervision loop. Every
        # publication is durably queued; no HTTP or git fetch may be called.
        jid='offline-supervision-test';sha='b'*40
        def fake_git(argv,**kwargs):
            if argv[:3]==['git','worktree','add']:
                src=Path(argv[4]);(src/'validation').mkdir(parents=True)
                (src/'validation/success.py').write_text('import time;time.sleep(.2)\n')
            else:raise AssertionError('unexpected network/git operation '+repr(argv))
            return ''
        req={'job_id':jid,'code_commit':sha,'entrypoint':'validation/success.py','args':[],'env':{},'timeout_sec':10,'memory_gb':0}
        with patch.object(b.v3,'run',side_effect=fake_git),patch.object(b,'cached_stop',return_value=False),patch.object(b.v3,'put_file',side_effect=b.enqueue),patch.object(b.v3,'agent_status',return_value={'timestamp':time.time()}),patch.object(b.v3,'gh_content',side_effect=AssertionError('network in supervision loop')):
            status=b.run_job(req)
        self.assertEqual(status['state'],'finished');self.assertEqual(status['returncode'],0)
        self.assertTrue((b.v3.JOBS/jid/'worker-exit.json').exists())

    def fixture(self,root):
        # Synthetic evidence exercises the verifier; it is not optical data.
        import copy,hashlib
        case='ef40-blue';policy={'pitch_um':.25,'phase_target_cycles':.25,'pupil_otf_tolerance':.002,
            'support_otf_tolerance':.002,'integral_relative_tolerance':.005,'pitch_otf_tolerance':.001,'joint_mtf_tolerance':.005}
        guard={'max_neighbor_block_bound_cycles':.1,'max_sampled_half_step_cycles':.1,'midpoint_all_transmitted':True}
        plan={'wavelength_nm':430.,'field_h':0.,'frozen':True,'policy':policy,'model_version':'qfilm04-fixed-plane-scalar-v3',
            'frequencies_lpmm':[5.,10.,15.,20.,30.,40.,50.,60.,70.,80.],
            'levels':[{'pupil_n':n,'phase_guards':[guard]*3} for n in (256,512,1024)]}
        plan['plan_hash']=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        runs=[]
        for n,si,pitch in ((256,0,.25),(512,0,.25),(256,1,.25),(512,1,.25),(512,1,.125),(1024,2,.125)):
            runs.append({'pupil_n':n,'support_index':si,'pitch_um':pitch,'pixel_n':int((.64,.96,1.28)[si]/(pitch*.001)),
                'center_film_mm':[0,0],'exit_z_mm':10.,'otf':[[{'real':.5,'imag':0.} for _ in range(2)] for _ in range(10)],
                'raw_intensity_integral':1.})
        d={'state':'numerically_converged','flags':{k:True for k in ('pupil_converged','support_converged','pitch_converged','joint_converged')},
            'resolved_sampling_plan':plan,'plan_hash':plan['plan_hash'],'runs':runs,'selected':runs[3],
            'comparisons':[{'axis':'joint','passed':True}]}
        for backend,device in (('c','cpu'),('g','cuda:0')):
            data=copy.deepcopy(d);data.update(actual_device=device,execution_id=backend)
            (root/f'{case}-{backend}-terminal.json').write_text(json.dumps(data))
        (root/'summary-terminal.json').write_text(json.dumps({'source_commit':'911bb9d334fce7e20a4407c4506ac683b52ff246',
            'cases':[{'id':case,'source_sha256':q.CASES[case][2]}]}))

    def test_independent_verifier_accepts_consistent_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            self.assertTrue(q.verify_pair(root,'ef40-blue')['verified'])

    def test_verifier_rejects_false_passed_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            p=root/'ef40-blue-c-terminal.json';d=json.loads(p.read_text())
            d['runs'][-1]['otf'][0][0]['real']=.1
            p.write_text(json.dumps(d))
            with self.assertRaises(AssertionError):q.verify_pair(root,'ef40-blue')

    def test_verifier_rejects_loosened_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            p=root/'ef40-blue-c-terminal.json';d=json.loads(p.read_text())
            d['resolved_sampling_plan']['policy']['pitch_otf_tolerance']=.1
            p.write_text(json.dumps(d))
            with self.assertRaises(AssertionError):q.verify_pair(root,'ef40-blue')

    def test_missing_terminal_never_counts_as_pass(self):
        d=q.collect('/nonexistent-bridge-verification-fixture')
        self.assertFalse(d['all_three_verified'])

    def test_prepared_requests_retain_original_validation(self):
        jid=q.NEW_IDS['ef40-blue'];request={'job_id':jid}
        with patch.object(b,'ORIGINAL_VALIDATE',return_value={'checked':True}) as check:
            self.assertTrue(b.validate_request(request,f'remote/prepared/bridge-v06/{jid}.json')['checked'])
            check.assert_called_once_with(request,f'remote/queue/{jid}.json')
        with self.assertRaises(ValueError):b.validate_request({'job_id':'unapproved-id'},'remote/prepared/bridge-v06/unapproved-id.json')

if __name__=='__main__':unittest.main(verbosity=2)
