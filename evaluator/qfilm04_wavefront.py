"""Shared float64 ray bundles and streamed fixed-film scalar integration.

The legacy Qfilm0.3 integrator is intentionally left unchanged.  These fields
have a declared scalar convention, not an empirical radiometric calibration.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
from evaluator.qfilm02 import (_surfaces_at, _start_for_stop, pupil_grid,
                              trace_pupil)
from evaluator.qfilm02_huygens import exit_plane_from_rays

MODEL_VERSION = "qfilm04-fixed-plane-scalar-v3"
EXTRACTOR_VERSION = "physical-centers-axis-dtft-v1"


class WavefrontError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def fixed_exit_plane(p, wavelength_nm=550., margin_mm=.1):
    """Conservative surface-envelope bound, independent of traced ray grids.

    Conic sag is monotone in r on its real branch. Positive asphere terms
    bound the positive excursion; negative terms cannot enlarge this bound.
    A missing clear aperture is unsupported rather than guessed from rays.
    """
    envelopes = []
    for s in _surfaces_at(p["surfaces"], wavelength_nm):
        aperture = getattr(s, "clear_aperture", None)
        if aperture is None or not math.isfinite(aperture) or aperture <= 0:
            raise WavefrontError("EXIT_PLANE_UNSUPPORTED", "finite clear apertures required")
        a = aperture / 2
        sag = 0.
        if not math.isinf(s.R):
            c = 1 / s.R
            q = 1 - (1 + s.conic) * c*c*a*a
            if q < 0:
                raise WavefrontError("EXIT_PLANE_UNSUPPORTED", "aperture exceeds real conic domain")
            sag = max(0., c*a*a/(1 + math.sqrt(q)))
        sag += sum(max(0., float(v)) * a**(4+2*i) for i, v in enumerate(s.asphere))
        envelopes.append(s.z + sag)
    z = max(envelopes) + margin_mm
    if not math.isfinite(z) or z >= float(p["image_z_mm"]):
        raise WavefrontError("EXIT_PLANE_UNSUPPORTED", "no bounded physical air interval before film")
    return z


def signed_mapping(x, y, valid, spacing):
    """Use valid derivative stencils; detect local folds without rejecting orientation."""
    dx0, dx1 = torch.gradient(x, spacing=(spacing, spacing))
    dy0, dy1 = torch.gradient(y, spacing=(spacing, spacing))
    jac = dx1*dy0 - dx0*dy1  # axes are (v,u), physical determinant is d(x,y)/d(u,v)
    stencil = torch.zeros_like(valid)
    stencil[1:-1, 1:-1] = (valid[1:-1, 1:-1] & valid[:-2, 1:-1] &
        valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:])
    good = stencil & torch.isfinite(jac) & (jac.abs() > 1e-10)
    fold = False
    for dim in (0, 1):
        a = [slice(None), slice(None)]; b = a.copy()
        a[dim] = slice(1, None); b[dim] = slice(None, -1)
        a, b = tuple(a), tuple(b)
        fold |= bool((good[a] & good[b] & (jac[a]*jac[b] < 0)).any())
    if not fold and bool((jac[good]>0).any()) and bool((jac[good]<0).any()):
        # A line of exactly singular nodes can separate both orientations;
        # identify connectivity before removing singular nodes from the map.
        from scipy import ndimage
        labels,count=ndimage.label(stencil.cpu().numpy())
        values=torch.where(good,jac,0.).cpu().numpy()
        lo=ndimage.minimum(values,labels,index=np.arange(1,count+1))
        hi=ndimage.maximum(values,labels,index=np.arange(1,count+1))
        fold=bool(np.any((lo<0)&(hi>0)))
    return jac, good, fold, stencil & ~good


@dataclass
class WavefrontBundle:
    x: torch.Tensor
    y: torch.Tensor
    opl: torch.Tensor
    amplitude_area: torch.Tensor
    active: torch.Tensor
    dirz: torch.Tensor
    power: torch.Tensor
    uv: torch.Tensor
    exit_z_mm: float
    film_z_mm: float
    wavelength_nm: float
    field_h: float
    center_mm: tuple
    metadata: dict


def _trace(p, wavelength_nm, field_h, uv, exit_z, frame_mm):
    alpha = math.atan(field_h*math.hypot(*frame_mm)/2/p["design_spec"]["efl_target_mm"])
    angle = torch.tensor([[alpha]], dtype=torch.float64, device=uv.device)
    surfs = _surfaces_at(p["surfaces"], wavelength_nm)
    tr = trace_pupil(surfs, angle, uv, float(p["stop_z_mm"]),
        p["design_spec"]["stop_diameter_mm"]/2, float(p["image_z_mm"]))
    xe, ye, oe, ok = exit_plane_from_rays(*tr, float(p["image_z_mm"]), exit_z)
    return tr, xe.flatten(), ye.flatten(), oe.flatten(), ok.flatten(), angle, surfs


@torch.inference_mode()
def prepare_wavefront_bundle(p, wavelength_nm, field_h, pupil_n, exit_z_mm,
                             center_mm=None, device="cpu", frame_mm=(36., 24.),
                             transport_model="uniform_stop_flux",max_bad_jacobian_fraction=.02):
    if pupil_n < 16 or transport_model not in ("uniform_stop_flux", "input_ray_tube_lossless"):
        raise ValueError("invalid pupil or transport model")
    uv, inside = pupil_grid(pupil_n, device)
    tr, xe, ye, oe, ok, angle, surfs = _trace(p, wavelength_nm, field_h, uv, exit_z_mm, frame_mm)
    n = pupil_n; rs = p["design_spec"]["stop_diameter_mm"]/2; du = 2*rs/n
    mask = (ok & inside).reshape(n, n)
    if int(mask.sum()) < 32:
        raise WavefrontError("NO_TRANSMITTED_WAVEFRONT", "fewer than 32 transmitted rays")
    x, y, opl = (v.reshape(n, n) for v in (xe, ye, oe))
    jexit, good, folded, bad = signed_mapping(x, y, mask, du)
    if folded:
        raise WavefrontError("EXIT_MAPPING_UNSUPPORTED", "signed exit mapping changes orientation locally")
    if float(bad.sum().double()/mask.sum()) > max_bad_jacobian_fraction:
        raise WavefrontError("EXIT_MAPPING_UNSUPPORTED", "singular exit mapping exceeds 2 percent")
    dzf = float(p["image_z_mm"]) - tr[0].reshape(n, n)
    sx = (tr[4]-tr[1]).reshape(n, n)/dzf
    sy = (tr[5]-tr[2]).reshape(n, n)/dzf
    dirz = (1 + sx*sx + sy*sy).rsqrt()
    zstart = min(float(p["stop_z_mm"]), surfs[0].z)-20.
    ix, iy, iv = _start_for_stop(surfs, angle, uv, float(p["stop_z_mm"]), rs, zstart)
    jin, ingood, infold, _ = signed_mapping(ix.reshape(n,n), iy.reshape(n,n),
                                           mask & iv.reshape(n,n), du)
    if transport_model == "input_ray_tube_lossless" and infold:
        raise WavefrontError("INPUT_MAPPING_UNSUPPORTED", "input mapping folds")
    if transport_model == "input_ray_tube_lossless":
        good &= ingood
        power = jin.abs()*du**2*float(torch.cos(angle).item())
        # |U|^2 is irradiance normal to the ray; z-plane flux is |U|^2 dir_z.
        amp = torch.sqrt(power*jexit.abs()*du**2/dirz.clamp_min(1e-12))
    else:
        power = torch.ones_like(x)*du**2
        amp = torch.sqrt(jexit.abs())*du**2
    active = good & torch.isfinite(opl) & torch.isfinite(amp) & (dirz > 0)
    if int(active.sum()) < 32:
        raise WavefrontError("EXIT_MAPPING_UNSUPPORTED", "insufficient valid derivative stencils")
    amp = torch.where(active, amp, 0.)
    chief = trace_pupil(surfs, angle, torch.zeros((1,2), device=device, dtype=torch.float64),
        float(p["stop_z_mm"]), rs, float(p["image_z_mm"]))
    chief_valid = bool(chief[6].all())
    if center_mm is None:
        if chief_valid:
            center_mm = (float(chief[4].item()), float(chief[5].item()))
        else:
            w = torch.where(active, power, 0.); den = w.sum()
            center_mm = tuple(float((tr[i].reshape(n,n)[active]*w[active]).sum()/den) for i in (4,5))
    meta = {"valid_rays": int(mask.sum()), "active_nodes": int(active.sum()),
        "geometric_pupil_fill": float(mask.sum().double()/inside.sum()),
        "bad_jacobian_fraction": float(bad.sum().double()/mask.sum()),
        "derivative_stencil_excluded_fraction": float((mask & ~good).sum().double()/mask.sum()),
        "reference_ray_blocked": not chief_valid, "transport_model": transport_model,
        "transmitted_ray_tube_power": float(power[active].sum()),
        "power_units": ("unit incident irradiance times mm^2; active quadrature only" if
            transport_model=="input_ray_tube_lossless" else "unit stop-plane flux density times mm^2; active quadrature only"),
        "energy_transport_validated": False, "signed_exit_mapping_fold": False,
        "exit_mapping_orientation": int(torch.sign(jexit[active].median()).item()),
        "input_mapping_fold": infold, "input_plane_z_mm": zstart,
        "source_plane_type": "physical", "pupil_n": n}
    return WavefrontBundle(x, y, opl, amp, active, dirz, power, uv,
        exit_z_mm, float(p["image_z_mm"]), wavelength_nm, field_h, tuple(center_mm), meta)


def otf_from_projections(projection_x, projection_y, pitch_mm, frequencies):
    px, py = np.asarray(projection_x, dtype=float), np.asarray(projection_y, dtype=float)
    f = np.asarray(frequencies, dtype=float)
    if (px.ndim != 1 or py.shape != px.shape or not math.isfinite(pitch_mm) or pitch_mm <= 0
        or not np.isfinite(f).all() or np.any(f < 0) or np.any(f >= .5/pitch_mm)
        or not np.isfinite(px).all() or not np.isfinite(py).all()
        or np.any(px < 0) or np.any(py < 0) or px.sum() <= 0
        or not np.isclose(px.sum(), py.sum(), rtol=1e-10, atol=0)):
        raise ValueError("invalid raw intensity projections or physical frequencies")
    x = (np.arange(len(px))-(len(px)-1)/2)*pitch_mm
    phase = np.exp(-2j*np.pi*f[:,None]*x[None,:])
    return np.stack((phase@px, phase@py), axis=-1)/px.sum()


@torch.inference_mode()
def integrate_fixed_film(bundle, pixel_n, pitch_um, frequencies, pixel_block=64,
                         kernel="legacy_deep_ray", progress_callback=None,
                         retain_intensity=False):
    """Stream raw projections without retaining a full detector tensor.

    Kernels contain 1/lambda. Absolute integrals are scalar field integrals,
    not calibrated watts. The exact RS1 derivative kernel is an opt-in check.
    """
    if pixel_n < 16 or pitch_um <= 0 or not math.isfinite(pitch_um) or pixel_block < 1:
        raise ValueError("invalid integration grid")
    if kernel not in ("legacy_deep_ray", "rs1"):
        raise ValueError("unknown boundary operator")
    a = bundle.active; X, Y, O, A, dz = [v[a] for v in (
        bundle.x,bundle.y,bundle.opl,bundle.amplitude_area,bundle.dirz)]
    n = pixel_n; pitch = pitch_um*1e-3; device = X.device
    coords = (torch.arange(n, device=device, dtype=torch.float64)-(n-1)/2)*pitch
    px = torch.zeros(n, device=device, dtype=torch.float64); py = torch.zeros_like(px)
    image = torch.empty((n,n), device=device, dtype=torch.float64) if retain_intensity else None
    L = bundle.film_z_mm-bundle.exit_z_mm; lam = bundle.wavelength_nm*1e-6
    ref = O[0]+L; total = n*n; border = torch.zeros((), device=device, dtype=torch.float64)
    peak = (-1., 0, 0)
    if progress_callback: progress_callback(0,total)
    for first in range(0,total,pixel_block):
        ids = torch.arange(first, min(first+pixel_block,total), device=device)
        iy, ix = ids//n, ids%n
        E = torch.zeros(len(ids),device=device,dtype=torch.complex128)
        # Bound temporary matrix memory even for multi-million-node pupils.
        # Sum complex fields across source chunks, never intensities.
        source_block = 65536
        for source_first in range(0,len(X),source_block):
            sl=slice(source_first,source_first+source_block)
            dx = (coords[ix]+bundle.center_mm[0])[:,None]-X[None,sl]
            dy = (coords[iy]+bundle.center_mm[1])[:,None]-Y[None,sl]
            r = torch.sqrt(L*L+dx*dx+dy*dy)
            phase = torch.remainder((O[None,sl]+r-ref)/lam,1.)*(2*math.pi)
            if kernel == "legacy_deep_ray":
                weight = (A[None,sl]*(1+dz[None,sl])*.5/r/lam).to(torch.complex128)
            else:
                weight = A[None,sl]*(L/r)*(1+1j*lam/(2*math.pi*r))/r/lam
            E += (torch.polar(torch.ones_like(phase),phase)*weight).sum(-1)
        intensity = E.abs().square()
        if not bool(torch.isfinite(intensity).all()):
            raise WavefrontError("NONFINITE_INTENSITY", "nonfinite raw scalar intensity")
        px.index_add_(0,ix,intensity); py.index_add_(0,iy,intensity)
        edge = ((coords[ix].abs() >= n*pitch/2-.020) |
                (coords[iy].abs() >= n*pitch/2-.020))
        border += intensity[edge].sum()
        value, index = intensity.max(0)
        if float(value) > peak[0]: peak = (float(value),int(iy[index]),int(ix[index]))
        if image is not None: image.flatten()[ids] = intensity
        if progress_callback: progress_callback(min(first+pixel_block,total),total)
    sum_i = float(px.sum())
    if not math.isfinite(sum_i) or sum_i <= 0:
        raise WavefrontError("ZERO_INTENSITY", "nonpositive raw scalar integral")
    rx, ry = px.cpu().numpy(), py.cpu().numpy()
    out = {"otf": otf_from_projections(rx,ry,pitch,frequencies),
        "raw_projection_x": rx, "raw_projection_y": ry,
        "raw_intensity_integral": sum_i*pitch*pitch, "pixel_area_mm2": pitch*pitch,
        "integral_convention": "sum |scalar U|^2 dx dy; includes 1/lambda kernel; not calibrated watts",
        "border_20um_flux_fraction": float(border)/sum_i,
        "peak_pixel_yx": peak[1:], "pixel_n": n, "pitch_um": pitch_um,
        "support_mm": n*pitch, "center_film_mm": bundle.center_mm,
        "exit_z_mm": bundle.exit_z_mm, "kernel": kernel, **bundle.metadata}
    out["source_chunk_nodes"]=65536
    if image is not None: out["raw_intensity"] = image.cpu().numpy()
    return out
