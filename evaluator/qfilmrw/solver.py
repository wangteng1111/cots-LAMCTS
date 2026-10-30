"""Observable-driven QfilmRW solve and evidence ledger; no legacy-score fallback."""
from dataclasses import asdict,replace
import math,time,hashlib
from pathlib import Path
import numpy as np
import torch
from .contracts import Task,Policy,VERSION,digest
from .rays import trace_lens,RWError
from .fourier import reconstruct_pupil,fixed_sampling_field,windowed_observables,pupil_autocorrelation,reference_probe

_PACKAGE=Path(__file__).parent
IMPLEMENTATION_SIGNATURE=hashlib.sha256(b''.join(file.read_bytes() for file in
    [*(_PACKAGE/name for name in ('contracts.py','rays.py','fourier.py','solver.py')),
     *(_PACKAGE.parent/name for name in ('qfilm02.py','qfilm04_wavefront.py','optical_primitives.py'))])).hexdigest()


def discrepancy(a,b):
    return {'complex_otf_max_abs':float(np.max(np.abs(a['otf']-b['otf']))),
            'integral_relative':abs(a['raw_integral']-b['raw_integral'])/max(abs(b['raw_integral']),1e-300),
            'kind':'convergence_estimate'}

@torch.inference_mode()
def solve_condition(p,wavelength_nm,field_h,task=Task(),policy=Policy(),device='cpu',progress=None,save_raw=None,mode='verify'):
    task.validate();policy.validate()
    if mode not in ('survey','verify'):raise ValueError('invalid solve mode')
    start=time.monotonic();center=None;last=None;state=None;pupil=None;field=None
    plan={'version':VERSION,'task':asdict(task),'policy':asdict(policy),'wavelength_nm':wavelength_nm,
          'field_h':field_h,'mode':mode,'prescription_hash':digest(p),'implementation_signature':IMPLEMENTATION_SIGNATURE}
    report={'schema':'QfilmRWCondition-v1','version':VERSION,'plan':plan,'plan_hash':digest(plan),
        'actual_device':str(torch.device(device)),'state':'running','runs':[],'comparisons':[],
        'J_task':None,'numerically_verified':False,'model_scope_validated':False,'empirically_calibrated':False,
        'training_eligible':False,'psf_sampling':'point samples; no detector pixel-area convolution',
        'flags':{k:False for k in ('pupil','angular_grid','support','pitch','joint','reference_spotcheck')}}
    def emit(stage):
        if progress:progress({'stage':stage,'elapsed_seconds':time.monotonic()-start,'device':str(device),'field_h':field_h,'wavelength_nm':wavelength_nm})
    def run(st,grid_n,t,axis):
        nonlocal pupil,field
        if time.monotonic()-start>policy.max_seconds:raise RWError('RESOURCE_LIMIT','declared condition deadline exceeded')
        m=t.pixel_n;length=1<<(grid_n+m-2).bit_length()
        # Bound the largest FFT batch plus output, not source*detector pairs.
        live=max(grid_n*length,m*length)+m*m+10*grid_n*grid_n
        if live>policy.max_live_elements:raise RWError('RESOURCE_LIMIT','declared transform working-set element cap exceeded')
        emit(axis+'_reconstruct');t0=time.monotonic();pupil=reconstruct_pupil(st,grid_n);t1=time.monotonic()
        if t.support_mm/2>=pupil['max_unaliased_half_window_mm']:raise RWError('ANGULAR_GRID_ALIASING','requested window exceeds Fourier-grid period')
        emit(axis+'_czt');field=fixed_sampling_field(pupil['grid'],pupil['q0'],pupil['dq'],m,t.pitch_um)
        obs=windowed_observables(field,t.pitch_um,t.frequencies,retain=save_raw is not None)
        obs.update(pupil_n=st.x.shape[0],angular_grid_n=grid_n,axis=axis,center_film_mm=st.center_mm,
            exit_z_mm=st.exit_z_mm,q0=pupil['q0'],dq=pupil['dq'],
            residual_phase_neighbor_cycles=pupil['residual_phase_neighbor_cycles'],
            active_angular_cells=pupil['active_angular_cells'],inverse_map_failed_cells=pupil['inverse_map_failed_cells'],
            excluded_boundary_angular_area=pupil['excluded_boundary_angular_area'],
            inverse_map_failed_interior_cells=pupil['inverse_map_failed_interior_cells'],
            unsupported_source_patch_queries=pupil['unsupported_source_patch_queries'],
            trace_metadata=st.metadata,profile={'reconstruction_seconds':t1-t0,'transform_extract_seconds':time.monotonic()-t1})
        if 'intensity' in obs:
            image=obs.pop('intensity');obs['raw_artifact']=save_raw(image,{'axis':axis,'pupil_n':st.x.shape[0],'angular_grid_n':grid_n,'pitch_um':t.pitch_um,'support_mm':t.support_mm})
        report['runs'].append(obs);report['selected']=obs
        emit(axis+'_completed');return obs
    def compare(a,b,axis,otf_tol,check_integral=True):
        d=discrepancy(a,b);d.update(axis=axis,otf_tolerance=otf_tol,integral_tolerance=policy.integral_relative_tolerance,
            passed=d['complex_otf_max_abs']<=otf_tol and (not check_integral or d['integral_relative']<=policy.integral_relative_tolerance))
        report['comparisons'].append(d);return d['passed']
    try:
        for n in policy.pupils:
            emit('trace_'+str(n));t0=time.monotonic();state=trace_lens(p,wavelength_nm,field_h,n,device,center,task.frame_mm)
            center=state.center_mm;row=run(state,policy.angular_grid_n,task,'pupil')
            row['profile']['trace_seconds']=time.monotonic()-t0-sum(row['profile'].values())
            if last is not None:
                ok=compare(last,row,'pupil',policy.pupil_otf_tolerance)
                report['flags']['pupil']=ok and row['residual_phase_neighbor_cycles']<=policy.residual_phase_target_cycles
            last=row
            # Sparse independent reference is tested at the second resolution
            # before spending more work on this unvalidated local model.
            if len(report['runs'])>=2:
                probe=reference_probe(state,pupil,field,task.pitch_um,row['peak_yx'])
                report['reference_probe']=probe
                report['flags']['reference_spotcheck']=probe['relative_complex_field_l2']<=policy.reference_relative_l2_tolerance
                if report['flags']['pupil']:break
        if report['state']=='REFERENCE_REQUIRED':pass
        elif mode=='survey':report['state']='SURVEY_COMPLETE'
        elif not report['flags']['reference_spotcheck']:report['state']='REFERENCE_REQUIRED'
        elif not report['flags']['pupil']:report['state']='SOURCE_SAMPLING_UNCONVERGED'
        else:
            base=last
            refined=run(state,policy.angular_grid_n*2,task,'angular_grid')
            report['flags']['angular_grid']=compare(base,refined,'angular_grid',policy.grid_otf_tolerance)
            expanded=replace(task,support_mm=.96 if task.support_mm==.64 else task.support_mm*1.5)
            support=run(state,policy.angular_grid_n*2,expanded,'support')
            report['flags']['support']=compare(refined,support,'support',policy.support_otf_tolerance)
            pitch_task=replace(task,pitch_um=task.pitch_um/2)
            pitch=run(state,policy.angular_grid_n*2,pitch_task,'pitch')
            report['flags']['pitch']=compare(refined,pitch,'pitch',policy.pitch_otf_tolerance)
            next_n=state.x.shape[0]*2
            emit('joint_trace');joint_state=trace_lens(p,wavelength_nm,field_h,next_n,device,center,task.frame_mm)
            joint_task=replace(expanded,pitch_um=task.pitch_um/2)
            joint=run(joint_state,policy.angular_grid_n*2,joint_task,'joint')
            report['flags']['joint']=compare(refined,joint,'joint',policy.joint_otf_tolerance) and joint['residual_phase_neighbor_cycles']<=policy.residual_phase_target_cycles
            # Verify the direct-reference discrepancy on the final refinement,
            # not just the initial pupil. These remain sparse checks.
            probe=reference_probe(joint_state,pupil,field,joint_task.pitch_um,joint['peak_yx'])
            report['joint_reference_probe']=probe
            report['flags']['reference_spotcheck'] &= probe['relative_complex_field_l2']<=policy.reference_relative_l2_tolerance
            report['selected']=refined
            report['numerically_verified']=all(report['flags'].values())
            report['state']='NUMERICALLY_CONVERGED_LOCAL_MODEL' if report['numerically_verified'] else 'NUMERICAL_OR_MODEL_GATE_NOT_PASSED'
        # Full-plane OTF is explicitly a different observable, never substituted.
        if pupil is not None:
            report['full_plane_local_model_otf']=pupil_autocorrelation(pupil,task.frequencies)
            report['full_plane_otf_objective']='QfilmRW-full-plane-local-Fourier-v1'
            if task.otf_domain=='full_plane':report['objective_otf']=report['full_plane_local_model_otf']
            elif 'selected' in report:report['objective_otf']=report['selected']['otf']
            report['full_plane_sampling']={'angular_grid_n':pupil['grid'].shape[0],'q0':pupil['q0'],'dq':pupil['dq']}
            if task.otf_domain=='full_plane':
                report['numerically_verified']=False
                report['state']='FULL_PLANE_CONVERGENCE_NOT_IMPLEMENTED'
    except RWError as exc:report.update(state=exc.code,error=str(exc))
    except torch.cuda.OutOfMemoryError:report.update(state='RESOURCE_LIMIT',error='CUDA allocation failed')
    except Exception as exc:
        import traceback
        report.update(state='NUMERICAL_FAILURE',error=repr(exc),traceback=traceback.format_exc())
    report['elapsed_seconds']=time.monotonic()-start
    report['error_ledger']=[
        {'source':'ray/OPD and field reconstruction','kind':'convergence_estimate','verified':report['flags']['pupil'] and report['flags']['angular_grid']},
        {'source':'local first-order Fourier model vs exact-distance RS1','kind':'reference_discrepancy','global_bound':False,'spotcheck_passed':report['flags']['reference_spotcheck']},
        {'source':'window, pitch, joint','kind':'convergence_estimate','verified':all(report['flags'][k] for k in ('support','pitch','joint'))},
        {'source':'full-plane/window equivalence','kind':'deterministic_bound','value':None,'outside_energy_fraction_bound':None,'border_band_is_tail_bound':False},
        {'source':'scalar z-flux transport, polarization/coatings, empirical calibration','validated':False}]
    return report


def assemble_score(conditions,geometric_q=None,area_weights=None):
    """No J unless the entire selected simulation model has been validated.

    Independent model validation can be added later without changing the J
    formula. Empirical calibration is a separate requirement for measured-lens
    prediction, not a prerequisite for a validated simulator-surrogate label.
    """
    if not conditions or not all(c.get('numerically_verified') and c.get('model_scope_validated') for c in conditions):
        return {'valid':False,'J_task':None,'reason':'numerical and model-scope validation required','training_eligible':False}
    if geometric_q is None or area_weights is None:return {'valid':False,'J_task':None,'reason':'complete geometry/radiometry task observables required','training_eligible':False}
    from evaluator.qfilm02 import score_image
    # This low-level entry expects a fully assembled, validated Q matrix.
    return score_image(geometric_q,area_weights)


def polychromatic_otf(samples,frequencies,weights,reference_center_mm):
    """Incoherent spectral sum in a common physical sensor coordinate system."""
    weights=np.asarray(weights,dtype=float)
    if len(samples)!=len(weights) or np.any(weights<=0) or not np.isfinite(weights).all():raise ValueError('invalid spectral weights')
    out=np.zeros((len(frequencies),2),dtype=complex)
    for sample,weight in zip(samples,weights/weights.sum()):
        shift=np.asarray(sample['center_film_mm'])-np.asarray(reference_center_mm)
        out+=weight*np.asarray(sample['otf'])*np.exp(-2j*np.pi*np.asarray(frequencies)[:,None]*shift[None,:])
    return out
