# Qfilm0.3 — Image-plane precision audit (2026-10-28)

**Status:** partial validated benchmark and method disagreement quantified; **NOT** a physics certification, **NOT** an absolute error bound against actual lenses. The full-frame spectral and independent Huygens convergence jobs are tracked below.

## Scope and physical interpretation

Target: photographic image-plane performance: monochromatic and polychromatic film MTF at fixed mechanical film, spot/PSF, chromatic response, chief-ray image heights/distortion, and numerical convergence. Two different domains tested:

- **Small field** 8×6mm and normalized 11 radial positions (maximum image height 5mm): DeepLens 1-million-ray Huygens, Qfilm monochromatic n=256/512 over seven wavelengths; 2 positive-EFL demonstrator designs + changed entrance stop.
- **Full frame** 36×24mm (half diagonal 21.633mm): Qfilm 256/512/1024 full-seven-wavelength high-resolution test (center, mid, edge). Additional 100 radial field and independent full-frame DeepLens Huygens tests scheduled.

**Critical:** The 8×6mm cross-validation cannot be extrapolated to 36×24mm full-frame edge performance without the separately labeled test.

Native DeepLens pinned upstream `vccimaging/DeepLens@911bb9d334fce7e20a4407c4506ac683b52ff246` (official Huygens equations; field orientations corrected). TorchOptics pinned `MatthewFilipovich/torchoptics@34fe9c40f874e53db225e703c5580befe8c523b5`. GPU: RTX 5090 double precision. Film plane identical in both optical prescriptions; 550nm paraxial focal lengths match to ~5e-7 mm.

## 1. Spectral film MTF, 11 radial field positions, 7 wavelengths, 1M ray Huygens

Absolute `max |MTF_Qfilm512 - MTF_DeepLensHuygens1M|` across each wavelength's 11 field positions, two orientations and 10 lp/mm frequencies.

| Wavelength nm | N-BK7 singlet | Two-singlet, internal stop, asphere |
|---:|---:|---:|
| 430 | 0.03212 | 0.05068 |
| 470 | 0.02662 | 0.03388 |
| 510 | 0.02269 | 0.03381 |
| 550 | **0.01927** | **0.02786** |
| 590 | 0.01756 | 0.05779 |
| 630 | 0.02983 | 0.18708 |
| 670 | **0.19502** | **0.23373** |

- All fields/wavelengths 77 PSFs each lens; 169 PSFs including 15 on the larger-stop singlet.
- Qfilm n256→512 max in *these* subsets: singlet 0.01728, internal stop 0.02167, larger stop 0.03286.
- At 670nm singlet on-axis: Qfilm512→Qfilm256 MTF max difference only 9.34e-5, while independent Huygens disagreement 0.1950: not explainable by Qfilm grid n alone.
- Huygens 1M ray and individually normalized 96x96 cropped PSF are **not yet proven converged** in the problematic red/strong aberration cases. Do not mislabel disagreement as Qfilm's absolute measured error.
- Magnification and scalar phase/propagation may become valid/invalid in different physical regions. Each case must qualify reference ray and window convergence before certification.

[Raw spectral reference data](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-huygens-11fields-7waves-20261028-01/qfilm03_huygens/huygens_spectral_fields.json)

## 2. Film-plane chief-ray location and chromatic geometry

DeepLens native `calc_chief_ray` selects the sampled ray closest to actual physical stop center. Comparing absolute film chief-ray height (Qfilm X vs DeepLens Y, consistent axis mapping), 11 radial fields×7 wavelengths:

| Case | Maximum absolute chief image height disagreement | 95th percentile | Maximum shift of DeepLens chief due to 262K→1M samples |
|---|---:|---:|---:|
| N-BK7 singlet | 0.130 micrometer | 0.087 micrometer | 0.436 micrometer |
| Internal-stop/asphere | 0.150 micrometer | 0.110 micrometer | 0.370 micrometer |

These data strongly support geometric ray solver agreement but do **not** establish an absolute 0.15µm localization accuracy because the independent DeepLens chief selection has a larger stochastic centering uncertainty. Geometric agreement must not be confused with PSF or throughput validity.

[Chief ray results](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-chief-geometry-20261028-02/qfilm03_geometry/chief_ray_distortion_chromatic.json)

## 3. True 36×24mm film, n=256/512/1024, 3 fields, all 7 wavelengths

| Positive lens | Max ΔMTF n256→512 | Max ΔMTF n512→1024 | |ΔJ| n512→1024 |
|---|---:|---:|---:|
| Singlet | 0.003470 | **0.000917** | 0.000115 |
| Internal stop, conic/asphere | 0.009379 | **0.003364** | 0.000425 |

A numerical discretization floor of approximately 1e-3 (singlet) or 3e-3 (internal-stop example) is observed in this **three-field/seven-wavelength** subset. It is NOT an all-fields bound or physical absolute MTF error. High-res n1024 timings for 3 fields×7 waves: singlet 0.72s, internal stop ~9.95s.

[Raw full-frame 1024 convergence](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-fullframe-1024-20261028-01/qfilm03_highres/fullframe_1024_convergence.json)

## 4. Fixed-film focus sensitivity

At 550nm, 3 fields, ±10µm and ±50µm mechanical film displacement compared to zero offset. Max MTF changes across T/S and frequencies:

| Shift | Qfilm | Huygens |
|---|---:|---:|
| -10µm | 0.01399 | 0.01445 |
| +10µm | 0.01392 | 0.01451 |
| -50µm | 0.07033 | 0.07095 |
| +50µm | 0.06865 | 0.07219 |

They respond consistently to actual focal-plane shifts; lens assembly and film flatness tolerance matter at the same order as requested MTF accuracy.

[Raw focus data, including initial OOM reference-window test](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-huygens-reference-convergence-20261028-01/qfilm03_huygens_quality/reference_sampling_and_focus.json)

## 5. Independent propagation algorithm precision

Earlier complete matched complex pupil field verification:
- DeepLens ASM vs upstream TorchOptics ASM on **exact same complex wavefield**, image plane and grid: max PSF absolute discrepancy 2.84e-14; max MTF discrepancy 8.65e-14.
- TorchOptics ASM vs RS-DIM (same sampled converging scalar input wavefront): max MTF discrepancy 2.40e-5 for that test.

This validates numerical wave propagation *on matched grid inputs* but **not the geometric-to-complex-pupil field reconstruction**, and cannot resolve the discrepancies of 670nm by itself.

[ASM/DIM three-way raw reference](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-3way-gpu-20261027-04/qfilm03_benchmark/three_way.json)

## 6. Reference Huygens convergence and hardware constraints

- Official DeepLens implementation's `psf_huygens` processes 10,000 ray contributions per block. At PSF crop 192x192 and 1M total rays it exceeds 32GB GPU VRAM; original run OOM while trying to allocate 5.49 GiB with ~2.53 GiB available.
- A validation-only runtime variant changes **only the contribution batch size** (10,000→1,024) and requires same-seed 96x96 PSF parity with the pinned upstream before widened-window results are accepted.
- Huygens limited 96px physical crop at sensor pitch 4.167µm = 0.400mm; doubling to192px =0.800mm tests PSF energy tail truncation. Each PSF is separately normalized by DeepLens, so even a converged MTF is **not an absolute photometric irradiance verification**.

[Memory-safe reference test source](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/qfilm03_huygens_safechunk.py)

## 7. Blocks before authoritative Alpha Lense Action→ΔQ pretraining

The following remain **uncertified**: 100 fields×7 wavelengths across varied real COTS catalog families, cross-wave absolute PSF centroid and color MTF at physically shared coordinates, lens photometry (vignetting/irradiance/throughput), coatings and glass batch, pupil amplitude/Jacobian weighting, severe aberration and caustics, mechanical shutter/Copal obstruction and decenter/tilt, fullframe edge valid-ray coverage, and converged external Huygens independent of window and number of rays.

Until physical and reference convergence are demonstrably satisfied, keep `physics_certified=False`, `training_eligible=False`. The observed **inter-solver difference is not a guaranteed bound on true physical error**.

## Tests and status registry

- [11-fields/7-waves Huygens](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/qfilm03_huygens_spectral_fields.py) — completed.
- [7-wave chief geometry](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/qfilm03_chief_geometry.py) — completed.
- [1024 pupil fullframe](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/qfilm03_highres_fullframe.py) — completed.
- [Full 100-fields/7-waves n64–512](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/qfilm03_imageplane_numerical.py) — queued at report creation.
- [True 36×24mm DeepLens Huygens independent](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/qfilm03_true_fullframe_huygens.py) — queued at report creation.
- [Reference Huygens ray/window convergence](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/qfilm03_huygens_safechunk.py) — memory-safe rerun queued at report creation.
