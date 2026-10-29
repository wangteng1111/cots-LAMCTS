#!/usr/bin/env python3
"""Read-only, independently verified usability report for the 11-lens matrix."""
import argparse,collections,csv,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'infra')]
from qfilm_convergence_status import verify_pair
NAMES=('ef100mm_f2.8','ef35mm_f2.0','ef40mm_f2.8','ef50mm_f1.8','ef85mm_f1.8',
       'rf16mm_f2.8','rf24mm_f1.8','rf35mm_f1.8','rf50mm_f1.8','sigma70mm_f2.8','yongnuo_50mm_f1.8')
FIELDS=(0.,.5,1.);WAVES=(430.,470.,510.,550.,590.,630.,670.)
SOURCE='911bb9d334fce7e20a4407c4506ac683b52ff246'

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None

def collect(jobs):
    sources=json.loads((ROOT/'validation/fixtures/qfilm04_real11_source_sha256.json').read_text())
    assert sources['source_commit']==SOURCE
    cases=[];rows=[]
    for i,name in enumerate(NAMES):
        jid=f'000-000-013-qfilm06-zz-real11-{i:02d}-{name}-20261030'
        art=jobs/jid/'artifacts';matrix=read(art/'matrix-terminal.json')
        case={'name':name,'job_id':jid,'matrix_terminal_present':matrix is not None,'conditions':[]}
        for h in FIELDS:
            for wave in WAVES:
                ident=f'{name}-h{h:g}-w{wave:g}';root=art/ident;native=read(root/'condition-terminal.json')
                row={'name':name,'field_h':h,'wavelength_nm':wave,'state':'missing_condition_terminal',
                     'full_condition_verified':False,'arithmetic_pair_passed':False,'J_task':None}
                if native:
                    expected={'id':ident,'name':name,'field_h':h,'wavelength_nm':wave,'source_sha256':sources['source_sha256'][name]}
                    identity_ok=all(native.get(k)==v for k,v in expected.items())
                    row['state']=native.get('state','unknown') if identity_ok else 'CONDITION_IDENTITY_MISMATCH'
                    tile=native.get('arithmetic_tile',{})
                    row['arithmetic_pair_passed']=bool(identity_ok and tile.get('cpu_device')=='cpu' and
                        tile.get('gpu_device','').startswith('cuda:') and tile.get('active_nodes_equal') is True and
                        tile.get('pitch_um')==.25 and all(isinstance(tile.get(k),(int,float)) and
                            0<=tile[k]<=1e-6 for k in ('complex_otf_max_abs','raw_intensity_relative_max_abs','relative_integral_difference')))
                    if row['arithmetic_pair_passed']:row['arithmetic_complex_otf_max_abs']=tile['complex_otf_max_abs']
                    # Recompute from native raw OTF/integral records even if a
                    # mutable summary claims success. Missing records never pass.
                    if identity_ok:
                        try:
                            verified=verify_pair(root,ident,expected=(name,wave,sources['source_sha256'][name]),field_h=h)
                            manifests=0
                            for backend in ('c','g'):
                                terminal=read(root/f'{ident}-{backend}-terminal.json')
                                for run in terminal['runs']:
                                    raw=run['raw_2d_artifact'];path=(root/raw['file']).resolve()
                                    assert root.resolve() in path.parents and path.is_file() and path.stat().st_size==raw['bytes']
                                    digest=hashlib.sha256()
                                    with path.open('rb') as stream:
                                        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
                                    assert digest.hexdigest()==raw['sha256'];manifests+=1
                            row.update(full_condition_verified=True,state='independently_verified',verification=verified,raw_arrays_verified=manifests)
                        except (OSError,ValueError,KeyError,AssertionError,StopIteration,TypeError,ZeroDivisionError) as exc:
                            row['full_verification_reason']=type(exc).__name__
                case['conditions'].append(row);rows.append(row)
        case['native_condition_terminals']=sum(r['state']!='missing_condition_terminal' for r in case['conditions'])
        case['verified_full_conditions']=sum(r['full_condition_verified'] for r in case['conditions'])
        case['arithmetic_pairs_passed']=sum(r['arithmetic_pair_passed'] for r in case['conditions'])
        case['state_counts']=dict(collections.Counter(r['state'] for r in case['conditions']))
        case['all_21_verified']=case['verified_full_conditions']==21
        cases.append(case)
    return {'schema':'Qfilm04Real11IndependentUsability-v1','observed_unix':time.time(),'source_commit':SOURCE,
            'expected_conditions':231,'native_condition_terminals':sum(c['native_condition_terminals'] for c in cases),
            'verified_full_conditions':sum(c['verified_full_conditions'] for c in cases),
            'lenses_all_21_verified':sum(c['all_21_verified'] for c in cases),'cases':cases,
            'training_eligible':False,'physics_certified':False,'empirically_calibrated':False,
            'usable_for_full_matrix_numerical_comparison':all(c['all_21_verified'] for c in cases),
            'interpretation':'Numerical convergence alone does not certify physical transport, measured lens accuracy, or training labels.'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs',type=Path,default=Path(os.environ.get('COTS_BRIDGE_ROOT','/var/lib/cots-lamcts'))/'jobs')
    ap.add_argument('--output-dir',type=Path,default=Path(os.environ.get('COTS_JOB_ARTIFACT_DIR','artifacts/real11_usability')))
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    report=collect(a.jobs)
    (a.output_dir/'usability-terminal.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fields=['name','field_h','wavelength_nm','state','full_condition_verified','arithmetic_pair_passed','J_task']
    with (a.output_dir/'conditions.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for case in report['cases']:writer.writerows(case['conditions'])
    lines=['# Qfilm 0.4: 11-prescription usability report','',
        f"Native condition terminals: {report['native_condition_terminals']}/231; independently verified full conditions: {report['verified_full_conditions']}/231.",'',
        '| Prescription | Native terminals | Tile CPU/CUDA pairs | Full conditions verified |','|---|---:|---:|---:|']
    lines += [f"| {c['name']} | {c['native_condition_terminals']}/21 | {c['arithmetic_pairs_passed']}/21 | {c['verified_full_conditions']}/21 |" for c in report['cases']]
    lines += ['',report['interpretation'],'','Training eligibility, empirical calibration and physical certification remain false.']
    (a.output_dir/'usability.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
