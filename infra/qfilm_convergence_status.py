"""Independent, strict native-terminal verification for the three controls."""
from pathlib import Path
import hashlib,json,math,time

CASES={'ef40-blue':('ef40mm_f2.8',430.,'18d8506e0f3a9c77b3a9aa53161b98930d93464e45a419d9b2091293dfa51f79'),
       'ef40-green':('ef40mm_f2.8',550.,'18d8506e0f3a9c77b3a9aa53161b98930d93464e45a419d9b2091293dfa51f79'),
       'rf50-red':('rf50mm_f1.8',670.,'da4b28edb742b435a3f9fc45186205157653614816f8438f5bf2d074ac716fae')}
OLD_IDS={case:f'000-000-00{i+1}-qfilm04-remediation-{case}-20261029-01' for i,case in enumerate(CASES)}
NEW_IDS={case:f'000-000-01{i+1}-qfilm06-{case}-20261029-01' for i,case in enumerate(CASES)}

def verify_pair(root,case,*,expected=None,field_h=0.):
    # Explicit expected data extends the same raw-data checks to the 11-lens
    # matrix. Existing three-control callers retain their pinned identities.
    name,wave,source_sha=CASES[case] if expected is None else expected
    summary=json.loads((root/'summary-terminal.json').read_text())
    assert summary['source_commit']=='911bb9d334fce7e20a4407c4506ac683b52ff246'
    record=next(v for v in summary['cases'] if v['id']==case)
    assert record['source_sha256']==source_sha
    pair=[]
    for backend in ('c','g'):
        d=json.loads((root/f'{case}-{backend}-terminal.json').read_text())
        assert d['state']=='numerically_converged'
        assert all(d['flags'][k] is True for k in ('pupil_converged','support_converged','pitch_converged','joint_converged'))
        p=d['resolved_sampling_plan']
        assert p['wavelength_nm']==wave and p['field_h']==field_h and p['frozen'] is True
        assert p['policy']['pitch_um']==.25
        assert p['model_version']=='qfilm04-fixed-plane-scalar-v3'
        assert p['frequencies_lpmm']==[5.,10.,15.,20.,30.,40.,50.,60.,70.,80.]
        encoded=json.dumps({k:v for k,v in p.items() if k!='plan_hash'},sort_keys=True,allow_nan=False,separators=(',',':')).encode()
        assert hashlib.sha256(encoded).hexdigest()==d['plan_hash']==p['plan_hash']
        assert d['runs'] and any(v['axis']=='joint' and v['passed'] is True for v in d['comparisons'])
        policy=p['policy']
        limits={'phase_target_cycles':.25,'pupil_otf_tolerance':.002,'support_otf_tolerance':.002,
            'integral_relative_tolerance':.005,'pitch_otf_tolerance':.001,'joint_mtf_tolerance':.005}
        assert all(0<policy[k]<=target for k,target in limits.items())
        runs={(r['pupil_n'],r['support_index'],r['pitch_um']):r for r in d['runs']}
        assert len(runs)==len(d['runs'])
        for key,r in runs.items():
            n,si,pitch=key
            guard=next(g for g in p['levels'] if g['pupil_n']==n)['phase_guards'][si]
            assert guard['max_neighbor_block_bound_cycles']<=policy['phase_target_cycles']
            assert guard['max_sampled_half_step_cycles']<=policy['phase_target_cycles'] and guard['midpoint_all_transmitted']
            assert r['raw_intensity_integral']>0 and math.isfinite(r['raw_intensity_integral'])
        selected=d['selected'];n,si,pitch=selected['pupil_n'],selected['support_index'],selected['pitch_um']
        assert selected==runs[(n,si,pitch)] and pitch==.25 and si>=1
        def difference(x,y,magnitude=False):
            values=[]
            for xx,yy in zip(x['otf'],y['otf']):
                for xc,yc in zip(xx,yy):
                    a=complex(xc['real'],xc['imag']);b=complex(yc['real'],yc['imag'])
                    values.append(abs(abs(a)-abs(b)) if magnitude else abs(a-b))
            assert len(values)==20 and all(math.isfinite(v) for v in values)
            return max(values)
        def flux_difference(x,y):return abs(x['raw_intensity_integral']-y['raw_intensity_integral'])/y['raw_intensity_integral']
        coarse=runs[(n//2,si,pitch)]
        assert difference(coarse,selected)<=policy['pupil_otf_tolerance']
        prior=max((r for k,r in runs.items() if k[1]==si-1 and k[2]==pitch),key=lambda r:r['pupil_n'])
        assert difference(prior,selected)<=policy['support_otf_tolerance']
        fine=runs[(n,si,pitch/2)];joint=runs[(n*2,si+1,pitch/2)]
        assert difference(selected,fine)<=policy['pitch_otf_tolerance']
        assert difference(selected,joint)<=policy['joint_mtf_tolerance']
        assert difference(selected,joint,True)<=policy['joint_mtf_tolerance']
        assert all(flux_difference(x,y)<=policy['integral_relative_tolerance'] for x,y in (
            (coarse,selected),(prior,selected),(selected,fine),(selected,joint)))
        pair.append(d)
    a,b=pair
    assert a['actual_device']=='cpu' and b['actual_device'].startswith('cuda:')
    assert a['execution_id']!=b['execution_id'] and a['plan_hash']==b['plan_hash']
    assert len(a['runs'])==len(b['runs'])
    maximum=0.;flux_maximum=0.
    for x,y in zip(a['runs'],b['runs']):
        assert all(x[k]==y[k] for k in ('pupil_n','support_index','pitch_um','pixel_n','center_film_mm','exit_z_mm'))
        assert len(x['otf'])==len(y['otf'])==10
        for xx,yy in zip(x['otf'],y['otf']):
            assert len(xx)==len(yy)==2
            for xc,yc in zip(xx,yy):
                delta=abs(complex(xc['real'],xc['imag'])-complex(yc['real'],yc['imag']))
                assert math.isfinite(delta);maximum=max(maximum,delta)
        flux=abs(x['raw_intensity_integral']-y['raw_intensity_integral'])/y['raw_intensity_integral']
        assert math.isfinite(flux);flux_maximum=max(flux_maximum,flux)
    assert maximum<=1e-6 and flux_maximum<=1e-6
    return {'verified':True,'plan_hash':a['plan_hash'],'executions':[d['execution_id'] for d in pair],
        'max_complex_otf_cpu_gpu_difference':maximum,'max_relative_integral_difference':flux_maximum,
        'joint_converged':True,'source_sha256':source_sha,'training_eligible':False}

def collect(jobs):
    rows=[]
    for case in CASES:
        row={'case':case,'verified':False,'attempts':[]}
        for jid in (NEW_IDS[case],OLD_IDS[case]):
            root=Path(jobs)/jid/'artifacts';terminal=root/'summary-terminal.json'
            if not terminal.exists():
                row['attempts'].append({'job_id':jid,'state':'no_native_terminal','directory_exists':root.exists()});continue
            try:
                verified=verify_pair(root,case);row.update(verified,verified_job_id=jid)
                row['attempts'].append({'job_id':jid,'state':'verified'});break
            except (OSError,ValueError,KeyError,AssertionError,StopIteration,TypeError,ZeroDivisionError) as exc:
                row['attempts'].append({'job_id':jid,'state':'terminal_not_verified','reason':type(exc).__name__})
        rows.append(row)
    return {'schema':'QfilmThreeIndependentTerminalVerification-v1','observed_unix':time.time(),
        'cases':rows,'all_three_verified':all(r['verified'] for r in rows),'training_eligible':False}
