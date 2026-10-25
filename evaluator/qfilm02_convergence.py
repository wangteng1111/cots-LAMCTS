"""Conservative numeric convergence gate for provisional Qfilm0.2 focal-plane labels.

It certifies *numerical resolution only*, not FFT exit-pupil physics validity.
Never sets physics-certified or training-eligible True.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np

def compare_qfilm(a,b,mtf_max_tol=.01,mtf_p95_tol=.005,j_tol=.002):
    qa=np.asarray(a["Q_image"],dtype=float)
    qb=np.asarray(b["Q_image"],dtype=float)
    if qa.shape!=qb.shape or qa.shape[1]!=32:
        raise ValueError("incomparable field grid")
    err=np.abs(qa[:,10:30]-qb[:,10:30])
    finite=bool(np.all(np.isfinite(qa)) and np.all(np.isfinite(qb)))
    ja=a["quality"].get("J_task")
    jb=b["quality"].get("J_task")
    jd=abs(float(ja)-float(jb)) if ja is not None and jb is not None else np.inf
    metrics={"mtf_max_abs":float(np.max(err)),
        "mtf_p95_abs":float(np.quantile(err,.95)),
        "mtf_mean_abs":float(np.mean(err)),
        "mtf_per_field_max_abs":np.max(err,axis=1).tolist(),
        "J_abs":float(jd),
        "finite":finite,
        "both_valid":bool(a["quality"].get("valid") and b["quality"].get("valid"))}
    metrics["numerically_converged"]=bool(finite and metrics["both_valid"]
         and metrics["mtf_max_abs"]<=mtf_max_tol
         and metrics["mtf_p95_abs"]<=mtf_p95_tol
         and jd<=j_tol)
    return metrics

def evaluate_multigrid(p,config,device="cuda:0",levels=(64,128,256,512,1024),
                       evaluate_fn=None,mtf_max_tol=.01,mtf_p95_tol=.005,j_tol=.002):
    if evaluate_fn is None:
        from evaluator.qfilm02 import evaluate as evaluate_fn
    if any(x<16 or x & (x-1) for x in levels):
        raise ValueError("pupil levels must be powers of two >= 16")
    if not levels or tuple(sorted(set(levels)))!=tuple(levels):
        raise ValueError("pupil levels must be ascending and nonduplicated")
    results=[];checks=[]
    for n in levels:
        # Bound per-field FFT memory: nfft=4*n, 7 colors and saved wave arrays.
        # Keep chunks small at n=512/1024 without silently reducing physics fidelity.
        max_safe_fields=max(1,int((128*128*100)/(n*n)))
        cfg=replace(config,pupil_n=n,fft_n=max(n*4,config.fft_n),
                    max_fields_per_chunk=min(config.max_fields_per_chunk,max_safe_fields))
        r=evaluate_fn(p,cfg,device=device)
        results.append(r)
        if len(results)>1:
            check=compare_qfilm(results[-2],results[-1],mtf_max_tol,mtf_p95_tol,j_tol)
            check["from_n"]=levels[len(results)-2];check["to_n"]=n
            checks.append(check)
            if check["numerically_converged"]:
                break
    out=results[-1]
    out["numerical_certification"]={
        "pupil_n":levels[len(results)-1],
        "checked_levels":list(levels[:len(results)]),
        "convergence_checks":checks,
        "numerically_converged":bool(checks and checks[-1]["numerically_converged"]),
        "physics_certified":False,
        "training_eligible":False,
        "reason":"exit_pupil_huygens_spectral_and_rigorous_psf_validation_required"}
    out["certified"]=False
    return out
