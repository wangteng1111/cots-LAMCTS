# Qfilm0.3 red-band wavefront and propagation repair — 2026-10-28

**Status:** reproducible **partial physical-optics correction**, not a production-certified complete photographic evaluator. Original fast Qfilm0.3 and Qfilm0.2 remain unmodified in default operation. New physically propagated, final-scored path is explicitly opt-in.

## Executive result
1. **Identical incident-ray experiment** over 10 cases (N-BK7 biconvex and internal-stop aspheric optical systems) at 550/630/670nm, center and off-axis: Qfilm/DeepLens ray trace, integrated OPL, physical exit-plane OPD and Jacobian are numerically the same under matching input rays. Typical near-axis 670nm max OPD difference ~2e-11 to 1e-10 micrometer, exit-plane coordinate differences ~1e-13–1e-12mm, Jacobian median ratio almost exactly 1. The earlier large MTF discrepancy does NOT arise from a Snell/sag/OPL calculation mismatch.
2. **Main identified error source:** old Qfilm0.2 stop-pupil *paraxial FFT* does not fully represent physical field propagation and coordinate/amplitude mapping. Holding Qfilm ray tracing and OPL fixed, deterministic **physical exit-plane direct Huygens spherical propagation** reduces several red-field disagreements from ~0.18–0.33 to ~0.02–0.04 MTF.
3. **Not every red error is eliminated:** biconvex λ670nm h0.1 remains ~0.126 discrepancy after n384; and center-field seven-color Huygens discrepancy in the internal-stop lens can *worsen at one frequency* even while mean errors improve. Do not assert full correction or scientific physical certification.
4. **Opt-in final score** implemented: `evaluator/qfilm03.evaluate_physical_huygens` calls `evaluator/qfilm03_physical_mode.evaluate`, recomputes actual fixed-film monochromatic PSFs with explicit ray-tube exit Jacobian amplitude, forms frequency-complex OTFs with lateral color shifts, recomputes *final* J over all fields/wavelengths. Not an intermediate/surrogate J. Remains `training_eligible=False`.
5. **GPU performance:** direct deterministic Huygens in representative 128² pupil examples ~0.1s monochromatic PSF for front stop and ~0.5s for internal stop on single RTX 5090. These are different quadrature sample counts and not directly equivalent to DeepLens's 1M-ray reference. Complete 3-fields × 7-colors lens score, including baseline non-MTF metrics: 2.76s front stop and 15.27s internal stop.

## Primary mathematical difference

Old Qfilm0.2 uses a plane-wave Fourier transform of a stop-plane grid:
```
P(u,v) = valid(u,v) * exp[i 2π/λ * (OPL_to_film_chief - piston)]
PSF = |FFT(P)|²
pitch = wavelength * paraxial_fnumber * pupil_n / fft_n
```
While useful and numerically converged, it discards actual exit-plane coordinates, map/amplitude transport, obliquity, and spherical distance to each fixed-film pixel.

New Qfilm Huygens computes real coordinates at a plane **after the last refracting surface** and before film, using the same ray-traced OPL:
```
E(X,Y) = sum_{valid u,v} A(u,v) * [0.5(1+cos(theta)) / r(X,Y,u,v)]
           * exp{i 2π/λ [OPL_exit(u,v) + r(X,Y,u,v)]}
PSF = |E|² / sum_window |E|²
A = sqrt(|d(x_exit,y_exit)/d(u_stop,v_stop)|) * dA_stop
```
Physical Jacobian amplitude is optional vs uniform-stop ablation. The measured **difference between the two amplitude choices is small** for selected test prescriptions (<~0.001 MTF for most), compared with the very large FFT-vs-Huygens difference. Do not conclude the amplitude model has been independently certified for arbitrary extreme pupils.

## Monochromatic red and green MTF: original Qfilm FFT vs native DeepLens and corrected direct Huygens

These are **maximum over both T/S and 5–80 lp/mm** of absolute MTF differences, NOT real lens measurement error. Qfilm original and corrected Huygens use same actual physical lens. Corrected Huygens n=128, sensor pixel 5µm, PSF 128² (0.64mm). DeepLens native 1M rays, identical film, same 0.64mm window.

| Lens | λ nm | field h | old Qfilm FFT ΔMTF | fixed-film Huygens + Jacobian ΔMTF |
|---|---:|---:|---:|---:|
| Singlet | 550 | 0.0 | 0.03079 | **0.01172** |
| Singlet | 630 | 0.0 | 0.10984 | **0.01262** |
| Singlet | 670 | 0.0 | 0.30782 | **0.02856** |
| Singlet | 670 | 0.1 | 0.23351 | **0.12364** — unresolved |
| Internal stop asphere | 550 | 0.0 | 0.03168 | **0.00833** |
| Internal stop asphere | 630 | 0.0 | 0.33271 | **0.02423** |
| Internal stop asphere | 670 | 0.0 | 0.18162 | **0.03851** |
| Internal stop asphere | 670 | 0.1 | 0.27293 | **0.02042** |

**Pupil convergence:** deterministic n64 Huygens is unreliable, with discrepancies up to ~0.75. At n128→192→256, direct Huygens for singlet λ670 h0 stays ~0.0286→0.0283→0.0282 and aspheric λ630 h0 ~0.0242→0.0349→0.0361. The unresolved singlet λ670 h0.1 tends to ~0.126 rather than zero even at n384. Thus that residual is NOT pupil grid undersampling.

Reference DeepLens 1M-ray Huygens itself has finite ray count/window variability, sometimes >0.02–0.08 MTF. Therefore the residual is not necessarily an absolute physical error of Qfilm.

## Integrated full-spectrum fixed-film J: DeepLens PSF-based regression

Uses seven wavelengths [430,470,510,550,590,630,670]nm with weights [0.075,0.125,0.16,0.28,0.16,0.125,0.075], 36×24mm film, h=[0,0.5,1], same actual film plane, physical 5µm detector crop. Baseline Qfilm pupil=512, FFT=2048; new direct Huygens n128; compare against previously independently computed same-prescription native DeepLens color-PSF fixture.

| Lens, field h | Qfilm FFT max Δpoly-MTF | new physical max Δpoly-MTF | FFT mean abs | new physical mean abs |
|---|---:|---:|---:|---:|
| Singlet, 0 | 0.04868 | **0.02159** | 0.02880 | **0.00382** |
| Singlet, 0.5 | 0.02051 | **0.01717** | 0.00374 | 0.00403 |
| Singlet, 1 | 0.01705 | 0.01731 | 0.00327 | 0.00462 |
| Internal-stop, 0 | 0.06723 | **0.11421** (worse max) | 0.05061 | **0.01415** |
| Internal-stop, 0.5 | 0.01881 | 0.03024 (worse) | 0.00358 | 0.00537 |
| Internal-stop, 1 | 0.01258 | 0.01331 | 0.00266 | 0.00342 |

Final J is **actually recomputed** from updated full-field PSFs:
- Singlet: Qfilm512 J = 2.8264120441; physical Huygens J = 2.8271948397.
- Internal stop: Qfilm512 J = 0.9563732554; physical Huygens J = 0.9637061247.

**Important:** existing relative illumination and film EE terms still inherit their earlier provisional baseline definitions. Do not describe the composite final J as fully photometrically certified.

## Code and reproduction

- [Exact same-ray OPL/OPD/Jacobian GPU audit](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/qfilm03_matched_ray_wavefront_audit.py)
- [New physical exit-plane wavefront/direct Huygens GPU kernel](https://github.com/wangteng1111/cots-LAMCTS/blob/main/evaluator/qfilm03_physical_huygens.py)
- [Opt-in complete lens physical Final-J](https://github.com/wangteng1111/cots-LAMCTS/blob/main/evaluator/qfilm03_physical_mode.py) and [Qfilm0.3 public entrypoint](https://github.com/wangteng1111/cots-LAMCTS/blob/main/evaluator/qfilm03.py)
- [Mono 8-case DeepLens kernel ablation](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/qfilm03_compare_propagation_kernels.py)
- [Pupil n=128–384 red convergence](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/qfilm03_huygens_resolution_convergence.py)
- [Seven-wave Final-J regression](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/qfilm03_physical_finalJ_validation.py)
- [Fixed DeepLens full-frame reference fixture](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/fixtures/qfilm03_fullframe_deeplens_poly.json)

Workstation raw records (branch `workstation-results`):
- [Matched OPL/OPD rays](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-identical-rays-opl-jacobian-20261028-01/qfilm03_common_ray/matched_incident_rays_opl_opd_exit_jacobian.json)
- [Propagation vs FFT ablations](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-propagation-rootcause-20261028-01/qfilm03_propagation_root_cause/matched_physical_huygens_vs_fft.json)
- [Red pupil convergence](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-huygens-red-resolution-20261028-01/qfilm03_huygens_resolution/red_resolution_convergence.json)
- [Complete physical Final-J evaluation](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-physical-finalJ-20261028-02/qfilm03_physical_mode_validation/fullframe_physical_finalJ_regression.json)

Usage:
```python
from evaluator.qfilm02 import Config
from evaluator.qfilm03 import evaluate_physical_huygens
from evaluator.qfilm03_physical_mode import PhysicalHuygensConfig

result = evaluate_physical_huygens(
    prescription,
    Config(fields=100, pupil_n=512, fft_n=2048),
    device="cuda:0",
    physical=PhysicalHuygensConfig(
        pupil_n=256, detector_n=192,
        detector_pitch_um=5.0, detector_block=256,
        amplitude_mode="sqrt_exit_jac",
    ),
)
assert result["training_eligible"] is False
print(result["quality"]["J_task"])
```

## Remaining blockers and strict next steps

1. **Explain singlet 670nm h0.1 residual ≈0.126 and 7-wave aspheric center max 0.114.** Compare actual DeepLens Huygens secondary source plane, objective infinity condition, source grid sampling, normalized PSF support, and physical energy weighting. Do not fit a correction based on one sample.
2. Independently validate convergence of **both** Qfilm and DeepLens direct Huygens as detector window approaches larger physical support; λ670 source phase plus reference kernel can remain sensitive.
3. Validate vector high-NA effects, coatings/polarization, ray clipping, photon flux and physical Jacobian for real COTS elements and complex vignetted structures.
4. Add adaptive high-fidelity physics gate and Top-K candidate selection rank-stability tests before data labeling.
5. Keep previous cheap FFT for fast screening but no claim it is authoritative high-fidelity physics in all red cases; **never silently replace final score by surrogate**.
