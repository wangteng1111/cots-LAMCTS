"""Optional Qfilm0.3 physically propagated, fixed-film MTF scoring backend.

Runs the complete candidate through GPU ray tracing and direct Huygens for EVERY
configured field/wavelength, with stop grid and exit Jacobian amplitude.
No final score substitution with a surrogate or intermediate optical merit.
The non-MTF fields still originate from provisional Qfilm02; MUST retain
training_eligible=False and physics_certified=False.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np,torch,time,math
from evaluator.qfilm02 import Config,evaluate as qfilm_baseline,score_image
from evaluator.qfilm03_physical_huygens import Propagation,eval_fixed_film,coherent_otf_axis

@dataclass(frozen=True)
class PhysicalHuygensConfig:
    pupil_n:int=128
    detector_n:int=128
    detector_pitch_um:float=5.
    detector_block:int=384
    amplitude_mode:str="sqrt_exit_jac"  # physical ray-tube energy transport
    record_all_mtf:bool=False
    def validate(self,base):
        Propagation(pupil_n=self.pupil_n,pixel_n=self.detector_n,
                    pixel_pitch_um=self.detector_pitch_um,
                    pixel_block=self.detector_block,weight_mode=self.amplitude_mode,
                    frame_width_mm=base.frame_width_mm,frame_height_mm=base.frame_height_mm).validate()
        if max(base.frequencies_lpmm)>=500./self.detector_pitch_um:
            raise ValueError("physical detector frequency exceeds Nyquist")
        if self.detector_n*self.detector_pitch_um*1e-3<.25:
            raise ValueError("PSF detector physical support is insufficient")

def evaluate(source,config:Config=Config(),device="cuda:0",
             physical:PhysicalHuygensConfig=PhysicalHuygensConfig()):
    config.validate()
    physical.validate(config)
    t0=time.perf_counter()
    with torch.inference_mode():baseline=qfilm_baseline(source,config,device=device)
    q=np.asarray(baseline["Q_image"],dtype=float).copy()
    wavelengths=np.asarray(config.wavelengths_nm,dtype=float)
    spectral_weights=np.asarray(config.spectral_weights,dtype=float)
    spectral_weights/=spectral_weights.sum()
    freq=np.asarray(config.frequencies_lpmm,dtype=float)
    setup=Propagation(pupil_n=physical.pupil_n,pixel_n=physical.detector_n,
       pixel_pitch_um=physical.detector_pitch_um,
       pixel_block=physical.detector_block,weight_mode=physical.amplitude_mode,
       frame_width_mm=config.frame_width_mm,frame_height_mm=config.frame_height_mm)
    raw=[]
    for fi in range(config.fields):
        h=float(fi)/(config.fields-1)
        samples=[]
        for wave_nm in wavelengths:
            with torch.inference_mode():res=eval_fixed_film(source,float(wave_nm),h,setup,device=device)
            mtf_complex=coherent_otf_axis(res["psf"],physical.detector_pitch_um*1e-3,freq)
            if not np.all(np.isfinite(mtf_complex)):
                raise ValueError(f"invalid physical PSF at field={fi},lambda={wave_nm}")
            samples.append({"lambda_nm":float(wave_nm),"chief_xy_mm":res["center_film_mm"],
                "otf":mtf_complex,"valid_rays":res["valid_rays"],
                "bad_jacobian_fraction":res["bad_jacobian_fraction"]})
        # Keep all colors in the SAME physically fixed film coordinates;
        # relative chief displacement is applied as coherent OTF phase shift,
        # but intensities (not fields) are summed across wavelengths.
        green_idx=int(np.argmin(np.abs(wavelengths-550.)))
        reference=samples[green_idx]["chief_xy_mm"]
        accumulated=np.zeros((len(freq),2),dtype=np.complex128)
        for wi,sample in enumerate(samples):
            shift=np.asarray(sample["chief_xy_mm"])-np.asarray(reference)
            phase=np.exp(-2j*np.pi*freq[:,None]*shift[None,:])
            accumulated+=spectral_weights[wi]*sample["otf"]*phase
        vals=np.clip(np.abs(accumulated),0.,1.)
        if not np.all(np.isfinite(vals)):raise ValueError("non-finite image-plane physical MTF")
        q[fi,10:20]=vals[:,0];q[fi,20:30]=vals[:,1]
        if physical.record_all_mtf:
            raw.append({"field_h":h,"n_valid_color_rays":[s["valid_rays"] for s in samples],
              "bad_jacobian_fraction":[s["bad_jacobian_fraction"] for s in samples],
              "mono_chief_xy_mm":[s["chief_xy_mm"] for s in samples]})
    out=dict(baseline)
    out["Q_image"]=q
    out["quality"]=score_image(q,baseline["area_weights"])
    out.update(schema="QfilmResult-0.3",version="Qfilm0.3-physical-direct-huygens-v1",
       physics_status="provisional_fixed_film_exit_huygens_jacobian",
       training_eligible=False,physics_certified=False,certified=False,
       physical_metadata={"pupil_n":physical.pupil_n,"detector_n":physical.detector_n,
                          "detector_pitch_um":physical.detector_pitch_um,
                          "weight_mode":physical.amplitude_mode,
                          "field_wavelength_evaluations":config.fields*len(wavelengths),
                          "remaining_provisional_metrics":["relative_illumination","chromatic_EE","coatings","tolerances"]},
       timings={"total_seconds":time.perf_counter()-t0})
    if physical.record_all_mtf:out["physical_diagnostics"]=raw
    out["assumptions"]=[x for x in baseline.get("assumptions",[]) if x!="scalar_fft_no_huygens_certification"]+[
       "finite_window_psf_normalization","direct_scalar_huygens_not_certified","green_only_encircled_energy"]
    return out
