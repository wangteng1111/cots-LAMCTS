#!/usr/bin/env python3
"""A new attempt ID, with explicit reuse only of fully verified old terminals."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'infra'))
from qfilm_convergence_status import CASES,OLD_IDS,verify_pair

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=list(CASES),required=True);a=ap.parse_args()
    bridge=Path(os.environ.get('COTS_BRIDGE_ROOT','/var/lib/cots-lamcts'))
    source=bridge/'jobs'/OLD_IDS[a.case]/'artifacts'
    try:verified=verify_pair(source,a.case)
    except (OSError,ValueError,KeyError,AssertionError,StopIteration,TypeError,ZeroDivisionError):verified=None
    if verified:
        out=Path(os.environ['COTS_JOB_ARTIFACT_DIR']);out.mkdir(parents=True,exist_ok=True)
        (out/'verified-reuse.json').write_text(json.dumps({'source_job_id':OLD_IDS[a.case],
            'new_computation_performed':False,'verification':verified},indent=2)+'\n')
        return 0
    return subprocess.call([sys.executable,'-u',str(ROOT/'validation/qfilm04_remediation_run.py'),
        '--mode','solve','--case',a.case,'--backends','g','c','--threads','4','--block','32','--save-full-psf'])

if __name__=='__main__':raise SystemExit(main())
