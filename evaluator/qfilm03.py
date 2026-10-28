"""Qfilm0.3: fast Qfilm physics plus pinned upstream DeepLens Huygens verification.

The fast evaluator always remains the score authority; reference MTF never silently
replaces a ground-truth J.  Verification requires identical prescription, stop,
film plane, material dispersion, angular field, wavelength, and physical MTF units.
Until full 100-field/7-wavelength propagation and photometric validation:
    physics_certified=False, training_eligible=False.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import json, math, pathlib, tempfile, time, traceback
import numpy as np
import torch
from evaluator.qfilm02 import Config, evaluate as torch_qfilm
from evaluator.qfilm02_fast import evaluate as fused_qfilm

VERSION="Qfilm0.3-raywave-deeplens-reference-v1"
DEEP_SHA="911bb9d334fce7e20a4407c4506ac683b52ff246"

@dataclass(frozen=True)
class ValidationConfig:
    mode: str = "fast"   # fast | verify; "verify" checks a sparse optical subset
    wavelength_nm: float = 550.0
    fields: tuple[float,...] = (0.0,0.5,1.0)
    huygens_spp: int = 1000000
    sensor_resolution: int = 3072
    psf_kernel: int = 96
    physical_sensor_width_mm: float = 12.8
    max_mtf_disagreement: float = 0.04
    base_backend: str = "torch"  # torch | hybrid; torch does not need Triton JIT
    reference_centering: str = "deterministic_physical_chief"  # alternative: native_stochastic
    source_dir: str = "/var/lib/cots-lamcts/deeplens_upstream"

    def validate(self,config):
        if self.mode not in ("fast","verify"): raise ValueError("mode must be fast or verify")
        if self.base_backend not in ("torch","hybrid"): raise ValueError("unknown Qfilm backend")
        if self.reference_centering not in ("deterministic_physical_chief","native_stochastic"):
            raise ValueError("unsupported Huygens reference PSF centering")
        if not (0<self.wavelength_nm and math.isfinite(self.wavelength_nm)): raise ValueError("bad wavelength")
        if self.huygens_spp<1024 or self.sensor_resolution<128 or self.psf_kernel<16:raise ValueError("under-resolved reference")
        if self.psf_kernel>self.sensor_resolution:raise ValueError("invalid PSF crop")
        if any(not 0<=f<=1 for f in self.fields):raise ValueError("normalized fields must be within 0..1")
        if self.mode=="verify" and min(abs(x-self.wavelength_nm) for x in config.wavelengths_nm)>1e-6:
            raise ValueError("verification wavelength must belong to Qfilm spectral list")

def _clock(fn):
    t=time.perf_counter()
    out=fn()
    if torch.cuda.is_available():torch.cuda.synchronize()
    return out,float(time.perf_counter()-t)

def _is_schott_bk7(source):
    """Fail closed: only certified matching dispersion shared by these prototypes."""
    from validation.qfilm_deeplens_same_prescription import BK7
    glasses=[s for s in source["surfaces"] if float(s["n_after"])>1.000001]
    if not glasses:return False
    for s in glasses:
        if abs(float(s["n_after"])-1.5168)>0.0001 or "sellmeier" not in s:return False
        sell=s["sellmeier"]
        for key in ("B","C"):
            if len(sell[key])!=3:return False
            if not np.allclose(sell[key],BK7[key],rtol=1e-8,atol=1e-12):return False
    return True

def _material_parity(source, deep, cfg):
    """Use supplier glass names only when raytracers agree at EVERY evaluation wavelength.

    DeepLens named glasses can be optical-catalog COTS glasses beyond N-BK7.
    The conversion produced initial surrogate material names; replace them and
    reconstruct each interface's input medium after stop insertion/sorting.
    """
    from deeplens.material import Material
    from evaluator.qfilm02 import _surfaces_at
    src=source["surfaces"]
    refractors=[d for d in deep["surfaces"] if d["type"] not in ("Aperture","Stop")]
    if len(refractors)!=len(src):
        return False,[],"refracting surfaces count mismatch"
    names=[];medium="air";report=[]
    for i,ss in enumerate(src):
        if float(ss["n_after"])<=1.000001:
            mat_name="air"
        else:
            mat_name=ss.get("glass_name") or ss.get("material_name")
            if mat_name is None and _is_schott_bk7(source):
                mat_name="N-BK7"
            if mat_name is None:
                return False,report,"glass_name required for non-BK7 catalog glass verification"
            try:mat=Material(mat_name)
            except Exception as e:return False,report,"DeepLens material unavailable: "+repr(e)
            deviations=[]
            for w in cfg.wavelengths_nm:
                qn=_surfaces_at(src,w)[i].n2
                dn=float(mat.ior(float(w)*1e-3))
                deviations.append(abs(qn-dn))
            if max(deviations)>2e-6:
                return False,report,f"glass {mat_name} dispersion mismatch: max n delta {max(deviations):.4g}"
            report.append({"index":i,"material":mat_name,"max_index_error":max(deviations)})
        names.append(mat_name)
    it=iter(names)
    for d in deep["surfaces"]:
        if d["type"] in ("Aperture","Stop"):
            d["mat1"]=medium;d["mat2"]=medium
        else:
            d["mat1"]=medium;d["mat2"]=next(it);medium=d["mat2"]
    return True,report,None

def _reference(source,cfg,rcfg,qfilm,device):
    # Upstream optical core is unmodified; only optional nonoptical imports shimmed.
    from validation.qfilm_deeplens_same_prescription import conversion,mtf_from_psf
    from deeplens import GeoLens
    if not isinstance(source,dict):
        return {"status":"unsupported_prescription","reason":"dict prescription required"}
    deep,src=conversion(source,sensor_size=rcfg.physical_sensor_width_mm,
                        sensor_res=rcfg.sensor_resolution,material_mode="bk7" if _is_schott_bk7(source) else "inline")
    matched,materials,reason=_material_parity(source,deep,cfg)
    if not matched:
        return {"status":"material_mismatch","reason":reason,"material_parity":materials}
    with tempfile.TemporaryDirectory(prefix="qfilm03_deep_") as td:
        file=pathlib.Path(td)/"lens.json"
        file.write_text(json.dumps(deep))
        lens,init_sec=_clock(lambda:GeoLens(filename=str(file),device=device,dtype=torch.float64))
        # Qfilm and DeepLens paraxial EFL must agree: detect material or geometry mismatch
        depefl=float(lens.foclen)
        efl_delta=abs(depefl-float(qfilm["efl_mm"]))
        result={"status":"running","source_sha":DEEP_SHA,"material_parity":materials,"device":str(device),
                "init_seconds":init_sec,"deeplens_efl_mm":depefl,
                "qfilm_efl_mm":float(qfilm["efl_mm"]),"efl_abs_difference_mm":efl_delta,
                "huygens_spp":rcfg.huygens_spp,"sensor_resolution":rcfg.sensor_resolution,
                "sensor_pixel_mm":float(lens.pixel_size),"psf_kernel":rcfg.psf_kernel,
                "psf_window_mm":float(lens.pixel_size)*rcfg.psf_kernel,
                "wavelength_nm":rcfg.wavelength_nm,"reference_centering":rcfg.reference_centering,"fields":[]}
        if not math.isfinite(depefl) or depefl<=0 or efl_delta>0.005:
            result["status"]="geometry_mismatch_or_non_imaging"
            return result
        freqs=np.asarray(cfg.frequencies_lpmm,dtype=float)
        image_radius=math.hypot(cfg.frame_width_mm,cfg.frame_height_mm)/2
        for h in rcfg.fields:
            fi=int(round(float(h)*(cfg.fields-1)))
            h_actual=fi/(cfg.fields-1)
            alpha=math.atan(h_actual*image_radius/cfg.target_efl_mm)
            normalized_y=math.tan(alpha)*depefl/(float(lens.sensor_size[1])/2)
            point=[0.0,float(normalized_y),-1.0e6]
            # Native upstream Huygens samples only SPP_PSF=8192 additional
            # rays to guess the chief-ray-centered PSF window, independent of
            # its million-ray coherent integration. That stochastic center
            # changes red off-axis MTF by as much as 0.256 in matched tests.
            # Anchor the detector patch to the exact Qfilm physical-stop
            # chief intercept; ray positions/OPL themselves have matched the
            # independent DeepLens trace to ~1e-10 micrometer.
            old_center=lens.psf_center
            exact_chief_x=None
            if rcfg.reference_centering=="deterministic_physical_chief":
                import types
                from evaluator.qfilm02 import _surfaces_at,trace_pupil
                a=torch.tensor([[alpha]],dtype=torch.float64,device=device)
                zero=torch.zeros((1,2),dtype=torch.float64,device=device)
                ss=_surfaces_at(src["surfaces"],rcfg.wavelength_nm)
                stop_r=float(src["design_spec"]["stop_diameter_mm"])/2.
                chief=trace_pupil(ss,a,zero,float(src["stop_z_mm"]),stop_r,float(src["image_z_mm"]))
                if not bool(chief[6].all()):
                    raise ValueError("cannot center reference: invalid actual chief ray")
                exact_chief_x=float(chief[4].item())
                def deterministic_center(self,points_obj,method="chief_ray",ray=None):
                    # DeepLens input field along +Y => output image along -Y.
                    # PSF routine applies pointc = -psf_center(...).
                    return torch.tensor([[0.,exact_chief_x]],dtype=torch.float64,device=device)
                lens.psf_center=types.MethodType(deterministic_center,lens)
            try:
                # DeepLens uses Y field; Qfilm uses X. Align T/S with the radial field.
                psf,elapsed=_clock(lambda:lens.psf_huygens(
                    points=point,ks=rcfg.psf_kernel,
                    wvln=rcfg.wavelength_nm*1e-3,spp=rcfg.huygens_spp,recenter=True))
            finally:
                lens.psf_center=old_center
            px,py=mtf_from_psf(psf.detach().cpu().numpy(),float(lens.pixel_size),freqs)
            if fi==0: tang,sag=px,py
            else: tang,sag=py,px
            q=qfilm["Q_image"][fi]
            dt=np.abs(np.asarray(tang)-np.asarray(q[10:20]))
            ds=np.abs(np.asarray(sag)-np.asarray(q[20:30]))
            entry={"field_index":fi,"field_h":h_actual,"angle_rad":alpha,
                   "reference_chief_x_mm":exact_chief_x,
                   "huygens_seconds":elapsed,"psf_sum":float(psf.sum().item()),
                   "mtf_t":tang,"mtf_s":sag,"qfilm_t":q[10:20].tolist(),
                   "qfilm_s":q[20:30].tolist(),
                   "mtf_max_abs":float(max(dt.max(),ds.max())),
                   "mtf_mean_abs":float(np.r_[dt,ds].mean()),
                   "mtf_p95_abs":float(np.quantile(np.r_[dt,ds],.95)),
                   "local_agreement":bool(max(dt.max(),ds.max())<=rcfg.max_mtf_disagreement)}
            result["fields"].append(entry)
        result["status"]="checked_subset"
        result["all_sampled_fields_below_tolerance"]=all(x["local_agreement"] for x in result["fields"])
        result["mtf_max_abs"]=max(x["mtf_max_abs"] for x in result["fields"])
        result["reference_seconds"]=sum(x["huygens_seconds"] for x in result["fields"])+init_sec
        return result

def evaluate(source,config:Config=Config(),device="cuda:0",validation:ValidationConfig=ValidationConfig()):
    """Return a Qfilm0.3 record, including reference checks where requested.

    IMPORTANT: verified subset is advisory; no intermediate / surrogate merit is
    returned in place of real Qfilm. Never claim training eligibility yet.
    """
    config.validate();validation.validate(config)
    fn=torch_qfilm if validation.base_backend=="torch" else fused_qfilm
    baseline,fast_s=_clock(lambda:fn(source,config,device=device))
    out=dict(baseline)
    out.update(schema="QfilmResult-0.3",version=VERSION,
               physics_status="provisional_crosscheck_v1",
               certified=False,training_eligible=False,
               timings={"qfilm_seconds":fast_s},
               reference={"status":"not_requested","source_sha":DEEP_SHA})
    if validation.mode=="verify":
        try:
            reference,rs=_clock(lambda:_reference(source,config,validation,baseline,device))
            out["reference"]=reference
            out["timings"]["reference_seconds"]=rs
        except Exception as e:
            out["reference"]={"status":"reference_error","error":repr(e),
                              "traceback":traceback.format_exc()[-4500:]}
    out["timings"]["total_seconds"]=fast_s+out["timings"].get("reference_seconds",0.0)
    return out


def evaluate_physical_huygens(source,config:Config=Config(),device="cuda:0",physical=None):
    """Explicit high-fidelity experiment: re-evaluate every final PSF and J.

    Unlike `evaluate(..., ValidationConfig(mode='verify'))`, this mode
    replaces the 10-frequency image-plane MTF with deterministic, physical
    fixed-film direct Huygens integral for ALL fields/wavelengths, then
    recomputes the single final J. This is NOT certified production truth.
    No surrogate scores or intermediate optics can count as final scores.
    """
    from evaluator.qfilm03_physical_mode import (
        evaluate as physical_eval, PhysicalHuygensConfig)
    if physical is None:physical=PhysicalHuygensConfig()
    return physical_eval(source,config,device=device,physical=physical)
