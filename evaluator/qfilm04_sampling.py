"""Versioned sampling plans, explicit resource limits and numerical acceptance.

Phase bounds concern sampled source pairs only; actual complex-OTF refinement
is additionally required. Engineering tolerances are not rigorous error bounds.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import numpy as np
import torch
from evaluator.qfilm04_wavefront import (MODEL_VERSION, EXTRACTOR_VERSION,
    WavefrontError, fixed_exit_plane, prepare_wavefront_bundle,
    integrate_fixed_film, _trace)


def digest(value):
    def canonical(v):
        if isinstance(v,(float,np.floating)) and not math.isfinite(v):
            if math.isnan(v):raise ValueError("NaN cannot identify an optical model")
            return {"__qfilm_float__":"+Infinity" if v>0 else "-Infinity"}
        if isinstance(v,dict):return {k:canonical(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [canonical(x) for x in v]
        return v
    return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ConvergencePolicy:
    pupil_candidates: tuple = (256,512,1024,2048,4096)
    support_candidates_mm: tuple = (.64,.96,1.28)
    pitch_um: float = .25
    phase_target_cycles: float = .25
    phase_alarm_cycles: float = .5
    phase_tiles_per_axis: int = 4
    midpoint_pairs: int = 128
    pupil_otf_tolerance: float = .002
    support_otf_tolerance: float = .002
    integral_relative_tolerance: float = .005
    pitch_otf_tolerance: float = .001
    joint_mtf_tolerance: float = .005
    max_pair_evaluations: int = 150_000_000_000_000
    transport_model: str = "uniform_stop_flux"
    kernel: str = "legacy_deep_ray"
    max_bad_jacobian_fraction: float = .02
    def validate(self):
        if (len(self.pupil_candidates)<2 or tuple(sorted(set(self.pupil_candidates)))!=tuple(self.pupil_candidates)
            or any(not isinstance(n,int) or n<16 for n in self.pupil_candidates)
            or any(b!=2*a for a,b in zip(self.pupil_candidates,self.pupil_candidates[1:]))):
            raise ValueError("at least two doubling pupil grids required")
        if (len(self.support_candidates_mm)<2 or
            tuple(sorted(set(self.support_candidates_mm)))!=tuple(self.support_candidates_mm)
            or any(not math.isfinite(w) or w<=0 for w in self.support_candidates_mm)):
            raise ValueError("increasing positive physical supports required")
        for k in ("pitch_um","phase_target_cycles","phase_alarm_cycles","pupil_otf_tolerance",
                  "support_otf_tolerance","integral_relative_tolerance","pitch_otf_tolerance","joint_mtf_tolerance"):
            if not math.isfinite(getattr(self,k)) or getattr(self,k)<=0:
                raise ValueError("positive finite convergence thresholds required")
        if self.phase_target_cycles>self.phase_alarm_cycles or self.phase_tiles_per_axis<1 or self.midpoint_pairs<1 or self.max_pair_evaluations<1:
            raise ValueError("invalid phase checks or resource budget")
        if self.transport_model not in ("uniform_stop_flux","input_ray_tube_lossless") or self.kernel not in ("legacy_deep_ray","rs1"):
            raise ValueError("unsupported transport or propagation convention")
        if not math.isfinite(self.max_bad_jacobian_fraction) or not 0<=self.max_bad_jacobian_fraction<=.1:
            raise ValueError("invalid mapping singularity threshold")
        for w in self.support_candidates_mm:
            count=w/(self.pitch_um*.001)
            if abs(count-round(count))>1e-6 or round(count)<16:
                raise ValueError("supports must contain an integer number of physical pixels")


def _pairs(bundle):
    a=bundle.active
    all_pairs=[]
    for dim in (0,1):
        low=[slice(None),slice(None)];high=low.copy()
        low[dim]=slice(None,-1);high[dim]=slice(1,None)
        low,high=tuple(low),tuple(high)
        mask=a[low]&a[high]
        indices=torch.arange(a.numel(),device=a.device).reshape_as(a)
        all_pairs.append((indices[low][mask],indices[high][mask]))
    return torch.cat([s[0] for s in all_pairs]),torch.cat([s[1] for s in all_pairs])


@torch.inference_mode()
def phase_guard(p,bundle,support_mm,policy,frame_mm=(36.,24.)):
    """Bound every sampled adjacent pair on a tiling of the full detector.

    |dc(P)| <= |dc(P0)| + min(2, |Qb-Qa|/L)*rho/lambda.
    Midpoint ray checks sample continuous-pupil behavior, not a proof of it.
    """
    ia,ib=_pairs(bundle)
    if not len(ia): raise WavefrontError("PUPIL_UNCONVERGED","no active neighbor pairs")
    X,Y,O=[v.flatten() for v in (bundle.x,bundle.y,bundle.opl)]
    lam=bundle.wavelength_nm*1e-6;L=bundle.film_z_mm-bundle.exit_z_mm
    m=policy.phase_tiles_per_axis; width=support_mm/m
    tile=(np.arange(m)+.5)*width-support_mm/2
    radius=width/math.sqrt(2); bounds=[]
    # Chunk source pairs to bound peak memory independently of pupil size.
    worst_bound=-1.;worst_pair=0
    for first in range(0,len(ia),262144):
        aa,bb=ia[first:first+262144],ib[first:first+262144]
        xa,ya,oa=X[aa],Y[aa],O[aa];xb,yb,ob=X[bb],Y[bb],O[bb]
        slope=torch.minimum(torch.full_like(xa,2.),torch.hypot(xb-xa,yb-ya)/L)
        upper=torch.zeros_like(xa)
        for ox in tile:
            for oy in tile:
                xx=bundle.center_mm[0]+float(ox);yy=bundle.center_mm[1]+float(oy)
                ra=torch.sqrt(L*L+(xx-xa)**2+(yy-ya)**2)
                rb=torch.sqrt(L*L+(xx-xb)**2+(yy-yb)**2)
                upper=torch.maximum(upper,(ob-oa+rb-ra).abs()/lam+slope*radius/lam)
        v,idx=upper.max(0)
        if float(v)>worst_bound:worst_bound=float(v);worst_pair=first+int(idx)
        bounds.append(float(v))
    ids=torch.linspace(0,len(ia)-1,min(policy.midpoint_pairs,len(ia)),device=X.device).long()
    ids=torch.unique(torch.cat([ids,torch.tensor([worst_pair],device=X.device)]))
    aa,bb=ia[ids],ib[ids]
    mid=(bundle.uv[aa]+bundle.uv[bb])*.5
    _,xm,ym,om,valid,_,_=_trace(p,bundle.wavelength_nm,bundle.field_h,mid,bundle.exit_z_mm,frame_mm)
    max_half=0.
    for ox,oy in ((0.,0.),(-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5)):
        xx=bundle.center_mm[0]+ox*support_mm;yy=bundle.center_mm[1]+oy*support_mm
        rm=torch.sqrt(L*L+(xx-xm)**2+(yy-ym)**2)
        for ii in (aa,bb):
            re=torch.sqrt(L*L+(xx-X[ii])**2+(yy-Y[ii])**2)
            if bool(valid.any()): max_half=max(max_half,float(((om-O[ii]+rm-re)/lam).abs()[valid].max()))
    midpoint_ok=bool(valid.all()) and max_half<=policy.phase_target_cycles
    return {"max_neighbor_block_bound_cycles":worst_bound,
        "sampled_pair_count":len(ia),"detector_tiles_per_axis":m,
        "midpoint_pairs_checked":len(ids),"midpoint_all_transmitted":bool(valid.all()),
        "max_sampled_half_step_cycles":max_half,"midpoint_check_passed":midpoint_ok,
        "passed":worst_bound<=policy.phase_target_cycles and midpoint_ok,
        "alarm":worst_bound>policy.phase_alarm_cycles,
        "continuous_pupil_certified":False,"is_mtf_error_bound":False}


def plan_identity(plan):
    return digest({k:v for k,v in plan.items() if k!="plan_hash"})


@torch.inference_mode()
def build_sampling_plan(p,wavelength_nm,field_h,policy=ConvergencePolicy(),
                        frame_mm=(36.,24.),progress_callback=None):
    """Resolve once on CPU and replay the exact coordinates/levels on both devices."""
    policy.validate(); exit_z=fixed_exit_plane(p); center=None; levels=[];pilot_geometry=None
    for n in policy.pupil_candidates:
        bundle=prepare_wavefront_bundle(p,wavelength_nm,field_h,n,exit_z,center,
            "cpu",frame_mm,policy.transport_model,policy.max_bad_jacobian_fraction)
        if center is None:
            center=bundle.center_mm;pilot_geometry=dict(bundle.metadata)
        guards=[]
        for support in policy.support_candidates_mm:
            guards.append(phase_guard(p,bundle,support,policy,frame_mm))
        levels.append({"pupil_n":n,"phase_guards":guards})
        if progress_callback:progress_callback({"stage":"phase_plan","pupil_n":n,"guards":guards})
        del bundle
    plan={"schema":"Qfilm04ResolvedSamplingPlan-v1","model_version":MODEL_VERSION,
        "plan_stage":"phase_screened_candidates",
        "extractor_version":EXTRACTOR_VERSION,"source_hash":digest(p),"policy":asdict(policy),
        "frame_mm":list(frame_mm),"field_h":float(field_h),"wavelength_nm":float(wavelength_nm),
        "exit_z_mm":exit_z,"center_mm":list(center),"levels":levels,
        "pilot_geometry":pilot_geometry,
        "planning_device":"cpu","frozen":True}
    plan["plan_hash"]=plan_identity(plan)
    return plan


def validate_plan(p,plan,policy,frame_mm):
    if (plan.get("plan_hash")!=plan_identity(plan) or plan.get("source_hash")!=digest(p)
        or digest(plan.get("policy"))!=digest(asdict(policy)) or plan.get("model_version")!=MODEL_VERSION
        or plan.get("extractor_version")!=EXTRACTOR_VERSION or plan.get("frame_mm")!=list(frame_mm)
        or not plan.get("frozen") or [s["pupil_n"] for s in plan.get("levels",[])]!=list(policy.pupil_candidates)):
        raise ValueError("sampling plan identity/source/model/policy mismatch")
    if not all(math.isfinite(float(v)) for v in (*plan["center_mm"],plan["exit_z_mm"],plan["wavelength_nm"],plan["field_h"])):
        raise ValueError("nonfinite frozen coordinates")
    f=np.asarray(plan.get("frequencies_lpmm",[]),dtype=float)
    if len(f)==0 or not np.isfinite(f).all() or np.any(f<0) or np.any(f>=500/policy.pitch_um):
        raise ValueError("invalid frozen physical frequencies")
    if len(plan["center_mm"])!=2 or not 0<=plan["field_h"]<=1 or plan["wavelength_nm"]<=0:
        raise ValueError("invalid frozen optical condition")
    if not np.isclose(plan["exit_z_mm"],fixed_exit_plane(p),atol=1e-12,rtol=0):
        raise ValueError("frozen plane does not match the physical surface envelope")
    if "run_schedule" in plan:
        seen=set()
        for grid in plan["run_schedule"]:
            if (len(grid)!=3 or grid[0] not in policy.pupil_candidates or
                not isinstance(grid[1],int) or not 0<=grid[1]<len(policy.support_candidates_mm)
                or grid[2] not in (policy.pitch_um,policy.pitch_um/2) or tuple(grid) in seen):
                raise ValueError("invalid or duplicate frozen execution grid")
            seen.add(tuple(grid))


def otf_change(a,b):return float(np.max(np.abs(a["otf"]-b["otf"])))
def integral_change(a,b):
    return abs(a["raw_intensity_integral"]-b["raw_intensity_integral"])/max(abs(b["raw_intensity_integral"]),1e-300)


@torch.inference_mode()
def solve_condition(p,plan,policy=ConvergencePolicy(),device="cpu",pixel_block=64,
                    checkpoint_callback=None,progress_callback=None,raw_artifact_callback=None):
    """Execute all convergence axes; never promote a budget-limited result."""
    frame=tuple(plan["frame_mm"]);validate_plan(p,plan,policy,frame)
    device=torch.device(device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    report={"state":"running","plan_hash":plan["plan_hash"],"resolved_sampling_plan":plan,
        "actual_device":str(device),"torch_version":str(torch.__version__),
        "execution_id":digest({"plan":plan["plan_hash"],"device":str(device),
            "torch":str(torch.__version__),"nonce":__import__("uuid").uuid4().hex}),
        "flags":{"geometry_valid":True,"wavefront_supported":True,"pupil_converged":False,
            "support_converged":False,"pitch_converged":False,"joint_converged":False,
            "reference_validated":False,"label_valid":False,"empirically_calibrated":False},
        "training_eligible":False,"physics_certified":False,"runs":[],"comparisons":[],
        "pair_evaluations":0,"scalar_flux_calibrated":False}
    cache={};budget=0;selected=None
    frozen_schedule=plan.get("run_schedule")
    allowed=None if frozen_schedule is None else {tuple(g) for g in frozen_schedule}
    report["frozen_schedule_replay"]=frozen_schedule is not None
    def emit():
        if checkpoint_callback:checkpoint_callback(report)
    def run(n,si,pitch):
        nonlocal budget
        key=(n,si,pitch)
        if key in cache:return cache[key]
        support=policy.support_candidates_mm[si];pixels=int(round(support/(pitch*.001)))
        level=next(v for v in plan["levels"] if v["pupil_n"]==n)
        if not level["phase_guards"][si]["passed"]:
            raise WavefrontError("PUPIL_UNCONVERGED","frozen phase guard failed")
        bound=n*n*pixels*pixels
        if budget+bound>policy.max_pair_evaluations:
            raise WavefrontError("RESOURCE_LIMIT","declared pair-evaluation budget exhausted")
        if allowed is not None and key not in allowed:
            raise WavefrontError("FROZEN_PLAN_DIVERGENCE","adaptive decision requested a grid outside the frozen executed schedule")
        bundle=prepare_wavefront_bundle(p,plan["wavelength_nm"],plan["field_h"],n,
            plan["exit_z_mm"],plan["center_mm"],str(device),frame,policy.transport_model,
            policy.max_bad_jacobian_fraction)
        # Frequencies are in the immutable plan, so a replay cannot change extraction.
        result=integrate_fixed_film(bundle,pixels,pitch,plan["frequencies_lpmm"],
            pixel_block,policy.kernel,progress_callback,retain_intensity=raw_artifact_callback is not None)
        if raw_artifact_callback is not None:
            image=result.pop("raw_intensity")
            result["raw_2d_artifact"]=raw_artifact_callback(image,{"pupil_n":n,"support_index":si,"pitch_um":pitch})
        budget+=bound;report["pair_evaluations"]=budget
        result.update(pupil_n=n,support_index=si)
        report["runs"].append(result);cache[key]=result
        emit();return result
    try:
        # Replays execute EVERY first-device grid, in the SAME order, before
        # checking decisions. A tolerance-borderline change cannot silently
        # substitute a different actual mesh on the second device.
        if frozen_schedule is not None:
            for n,si,pitch in frozen_schedule:run(n,si,pitch)
        previous_support=None
        for si,_ in enumerate(policy.support_candidates_mm):
            passing=[v["pupil_n"] for v in plan["levels"] if v["phase_guards"][si]["passed"]]
            last=None;current=None
            for n in passing:
                current=run(n,si,policy.pitch_um)
                if last is not None:
                    diff=otf_change(last,current);flux=integral_change(last,current)
                    ok=diff<=policy.pupil_otf_tolerance and flux<=policy.integral_relative_tolerance
                    report["comparisons"].append({"axis":"pupil","support_index":si,
                        "n_coarse":last["pupil_n"],"n_fine":n,"complex_otf_change":diff,
                        "integral_relative_change":flux,"passed":ok})
                    if ok:break
                last=current;current=None
            if current is None:
                raise WavefrontError("PUPIL_UNCONVERGED","no converged phase-safe pupil pair")
            report["flags"]["pupil_converged"]=True;selected=current;report["selected"]=selected
            if previous_support is not None:
                diff=otf_change(previous_support,current);flux=integral_change(previous_support,current)
                ok=diff<=policy.support_otf_tolerance and flux<=policy.integral_relative_tolerance
                report["comparisons"].append({"axis":"support","coarse_mm":previous_support["support_mm"],
                    "fine_mm":current["support_mm"],"complex_otf_change":diff,
                    "integral_relative_change":flux,"passed":ok})
                if ok:report["flags"]["support_converged"]=True;break
            previous_support=current
        if not report["flags"]["support_converged"]:
            raise WavefrontError("SUPPORT_UNCONVERGED","support sequence exhausted")
        fine=run(selected["pupil_n"],selected["support_index"],policy.pitch_um/2)
        diff=otf_change(selected,fine);flux=integral_change(selected,fine)
        ok=diff<=policy.pitch_otf_tolerance and flux<=policy.integral_relative_tolerance
        report["comparisons"].append({"axis":"pitch","complex_otf_change":diff,
            "integral_relative_change":flux,"passed":ok})
        if not ok:raise WavefrontError("PITCH_UNCONVERGED","detector pitch cross-check failed")
        report["flags"]["pitch_converged"]=True
        next_n=selected["pupil_n"]*2;next_si=selected["support_index"]+1
        if next_n not in policy.pupil_candidates or next_si>=len(policy.support_candidates_mm):
            raise WavefrontError("JOINT_UNCONVERGED","no joint refinement level in declared plan")
        joint=run(next_n,next_si,policy.pitch_um/2)
        diff=float(np.max(np.abs(np.abs(selected["otf"])-np.abs(joint["otf"]))))
        complex_diff=otf_change(selected,joint);flux=integral_change(selected,joint)
        # Complex convergence also matters for subsequent multi-color accumulation.
        ok=(diff<=policy.joint_mtf_tolerance and complex_diff<=policy.joint_mtf_tolerance
            and flux<=policy.integral_relative_tolerance)
        report["comparisons"].append({"axis":"joint","mtf_change":diff,"complex_otf_change":complex_diff,
            "integral_relative_change":flux,"passed":ok})
        if not ok:raise WavefrontError("JOINT_UNCONVERGED","joint pupil/window/pitch refinement failed")
        report["flags"]["joint_converged"]=True;report["state"]="numerically_converged"
    except WavefrontError as exc:
        report.update(state=exc.code,error=str(exc))
        if exc.code in ("NO_TRANSMITTED_WAVEFRONT",):report["flags"]["geometry_valid"]=False
        if exc.code.endswith("MAPPING_UNSUPPORTED"):report["flags"]["wavefront_supported"]=False
    if report["runs"] and "selected" not in report:
        report["latest_provisional"]=report["runs"][-1]
    final_plan=json.loads(json.dumps(plan))
    schedule=[[r["pupil_n"],r["support_index"],r["pitch_um"]] for r in report["runs"]]
    selected_grid=None if selected is None else [selected["pupil_n"],selected["support_index"],selected["pitch_um"]]
    if frozen_schedule is not None:
        if schedule!=frozen_schedule or selected_grid!=plan.get("selected_grid"):
            report.update(state="FROZEN_PLAN_DIVERGENCE",error="executed schedule or selected mesh differs from the first execution")
            report["flags"]["joint_converged"]=False
    else:
        final_plan.update(plan_stage="execution_schedule_frozen",run_schedule=schedule,selected_grid=selected_grid,
            planning_phase_plan_hash=plan["plan_hash"])
        final_plan["plan_hash"]=plan_identity(final_plan)
    report["planning_phase_plan_hash"]=plan.get("planning_phase_plan_hash",plan["plan_hash"])
    report["plan_hash"]=final_plan["plan_hash"]
    report["resolved_sampling_plan"]=final_plan
    emit();return report


def make_plan(p,wavelength_nm,field_h,frequencies,policy=ConvergencePolicy(),**kwargs):
    plan=build_sampling_plan(p,wavelength_nm,field_h,policy,**kwargs)
    plan["frequencies_lpmm"]=list(frequencies)
    plan["plan_hash"]=plan_identity(plan)
    return plan


def combine_spectral_otf(samples,wavelengths,weights,frequencies,model="normalized_fixed_weights"):
    """Incoherent colors; preserve translation until after complex summation.

    Physical spectral flux requires independently validated transport. Never
    estimate spectral throughput from captured-window integrals.
    """
    w=np.asarray(weights,dtype=float)
    if len(samples)!=len(w) or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError("invalid spectral samples or weights")
    if model!="normalized_fixed_weights":
        raise WavefrontError("SPECTRAL_TRANSPORT_UNVALIDATED","physical spectral mode requires validated energy transport")
    w=w/w.sum();f=np.asarray(frequencies);green=int(np.argmin(abs(np.asarray(wavelengths)-550)))
    ref=np.asarray(samples[green]["center_film_mm"])
    total=np.zeros((len(f),2),dtype=np.complex128)
    for sample,weight in zip(samples,w):
        shift=np.asarray(sample["center_film_mm"])-ref
        total+=weight*sample["otf"]*np.exp(-2j*np.pi*f[:,None]*shift[None,:])
    return total
