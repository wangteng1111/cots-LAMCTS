# Alpha Lense Qfilm0.3 — DeepLens Integration and TorchOptics GPU Benchmark

**Date:** 2026-10-27  
**Status:** operational prototype; sparse optical Huygens verification passed; **NOT** physics-certified as production COTS pretraining ground truth.

## Code and pinned scientific dependencies

- Qfilm0.3 interface: `evaluator/qfilm03.py`
- Same-physical-prescription DeepLens adapter: `validation/qfilm_deeplens_same_prescription.py`
- Integrated optical + propagation 3-way benchmark: `validation/qfilm03_three_way_benchmark.py`
- Full 100-field x 7-wavelength GPU throughput benchmark: `validation/qfilm03_full_gpu_benchmark.py`
- Matched catalog glass sanity test: `validation/qfilm03_material_parity.py`
- Bootstrap pinned TorchOptics: `scripts/qfilm03_torchoptics_bootstrap.py`
- DeepLens upstream git: `vccimaging/DeepLens@911bb9d334fce7e20a4407c4506ac683b52ff246`
- TorchOptics upstream git: `MatthewFilipovich/torchoptics@34fe9c40f874e53db225e703c5580befe8c523b5`
- Server: `aipss-Rack-Server`, single NVIDIA RTX 5090 for benchmarks; shared venv Torch `2.14.0+cu130`; upstream optical source was not patched.
- Reproduction from workstation: `scripts/cuda_venv_exec.py validation/qfilm03_three_way_benchmark.py`; for Triton run `scripts/cuda_venv_exec.py scripts/qfilm02_triton_bootstrap.py qfilm03_full_gpu_benchmark.py`.
- PyPI blocked by workstation network. Optional plotting/CV/LPIPS imports are shimmed in validation adapter only; upstream optics and TorchOptics propagation are not modified. Their dependency bundles are maintained outside shared Torch env.

## Architecture

1. `Qfilm03.evaluate(..., validation=ValidationConfig(mode="fast"))`: exact same Q_image, Q_wavefront and J_task as underlying evaluated Qfilm0.2 fast engine; no DeepLens overhead.
2. `ValidationConfig(mode="verify")`: additionally run upstream DeepLens `GeoLens.psf_huygens()` for selected physical wavelength(s) and field positions; store per-field T/S MTF discrepancy, solver timings, geometry parity and glass parity. Neither Huygens value nor a surrogate replaces final Qfilm J in this early version.
3. Fast optical physics may use the verified Triton CUDA backend for front-stop systems; for internal-stop systems it falls back to the PyTorch reference backend. This choice is intentionally conservative.
4. All Qfilm0.3 records preserve `physics_certified=False`, `training_eligible=False` until real catalog, broad field/spectral and flux-physical certification is complete.
5. For strict verification, supplier glass names are mapped into DeepLens only when `n(lambda)` is compatible with Qfilm at **every** requested wavelength (tolerance 2e-6). A test `N-BK7` glass matches with max |Δn| = 1.30e-7; malicious/wrong `N-SF6` substitution is correctly rejected. Unknown glass without physical material parity returns `material_mismatch`, never a false "verified".

## Matched photographic optics: Qfilm0.3 vs upstream DeepLens Huygens

Conditions: `Config(fields=3,pupil_n=128,fft_n=512,wavelengths_nm=(550,),frame_width_mm=8,frame_height_mm=6)`, pupil + film physically aligned; 3 fields (center, middle, edge), 1 million Huygens rays per field, PSF 96x96, 4.1667um pixel pitch, 10 frequency probes 5–80 lp/mm, valid T/S orientation mapping.

| Metric | Biconvex singlet (N-BK7) | Two-singlet, internal stop and conic/asphere |
|---|---:|---:|
| DeepLens EFL vs Qfilm EFL error (mm) | 5.46e-7 | 4.62e-7 |
| Center max |ΔMTF| | 0.00712 | 0.02801 |
| Mid-field max |ΔMTF| | 0.01366 | 0.02068 |
| Edge max |ΔMTF| | 0.01973 | 0.03177 |
| Entire 3-field max |ΔMTF| | **0.01973** | **0.03177** |
| Final J change caused by integrating DeepLens | 0 | 0 |
| End-to-end verify time (s) | 7.04 | 7.35 |
| DeepLens reference overhead alone (s) | 7.01 | 6.93 |

The user-defined error threshold 0.04 is only a provisional acceptance guard for these limited demonstration lenses; it does not certify commercial-lens production MTF or broad topology safety.

## Full workload GPU performance

100 radial fields x 7 wavelengths x 64² pupil samples, warm-up then repeated, **one RTX 5090**. A full production physics gate would require higher pupil resolution for difficult fields.

| Metric | Front-stop biconvex | Positive EFL, internal stop |
|---|---:|---:|
| Qfilm0.2 reference Torch median (seconds) | 0.16889 | 30.60672 |
| Qfilm0.3 hybrid fast median (seconds) | **0.05583** | **30.53191** |
| Speedup | 3.03x | 1.002x |
| Peak Torch allocated GPU memory, Qfilm0.2 (GiB) | 1.092 | 0.116 |
| Peak Torch allocated GPU memory, Qfilm0.3 (GiB) | 1.202 | 0.116 |
| Q_image max abs error | 1.24e-11 | 0 |
| J_task abs error | 1.13e-13 | 0 |

**Bottleneck:** internal-stop backward pupil inverse-ray shooting has not been accelerated by the fused code. It remains ~30.5 seconds per 100x7 state. Do not quote the front-stop 0.056 sec for general COTS optical topologies.
Peak values are PyTorch `torch.cuda.max_memory_allocated`, not whole-device memory including CUDA driver caching and other processes.

## Independently evaluated wave propagation: DeepLens vs TorchOptics

TorchOptics is a PyTorch wave-propagation library, **NOT** an end-to-end sequential multi-refractive-surface lens tracer. Full Qfilm times vs TorchOptics propagation kernels are not physically comparable. For a fair direct test, call upstream DeepLens `GeoLens.pupil_field` once to reconstruct a real N-BK7 singlet exit-pupil **complex wavefield**, then propagate exactly the same complex tensor in the same physical plane with DeepLens ASM and TorchOptics ASM.

Conditions: 550 nm, propagation distance 53.2218245 mm, pupil-field grid 1024², propagation grid 2048², pitch 0.0125 mm, PSF crop 96×96. Only spatial frequencies `[5,10,20,30]` lp/mm were included, **strictly below Nyquist 40 lp/mm**; numerical double precision; one RTX 5090.

| Metric | Result |
|---|---:|
| DeepLens ASM propagation median | 2.361 ms |
| TorchOptics ASM propagation median | 4.438 ms |
| DeepLens ASM advantage (TorchOptics time / DeepLens time) | 1.88x |
| Maximum normalized 96² PSF pixel absolute discrepancy | **2.84e-14** |
| Maximum MTF absolute discrepancy | **8.65e-14** |

This establishes the **propagation kernel implementations are consistent given the exact same complex input wavefield**. It does not independently verify the pupil-field reconstruction.

TorchOptics independent tiny converging spherical-field sample test: both `TorchOptics.ASM` and `TorchOptics.DIM` (Rayleigh–Sommerfeld) run from exactly the same 96² field (input pitch 5um, pupil radius 0.12mm, propagation z=6mm, λ=550nm), 64² common normalized PSF crop.

| Metric | TorchOptics ASM | TorchOptics DIM |
|---|---:|---:|
| Median kernel time | 2.739 ms | 2.140 ms |

ASM/DIM max normalized pixel PSF disagreement = **1.34e-6**, MTF absolute disagreement = **2.40e-5**. The DIM may outperform ASM for this **small** grid; this is not a claim for full high-resolution Qfilm.

## Critical interpretation

- Older apparent DeepLens coherent-vs-Qfilm disagreements (>0.3 MTF) were shown sensitive to exit pupil/sensor discretization; once **input** wavefield is identical, upstream DeepLens ASM and TorchOptics ASM are numerically equivalent to double precision.
- Huygens vs fast Qfilm differences still exist (~0.02–0.03 on tested lenses). That is real cross-method uncertainty; it includes convergence and finite window normalization, not an independently measured camera-lens production error.
- At present Qfilm0.3 is **functioning engineering cross-check software**, not a certified evaluator of arbitrary real lens purchases or ready-to-use pretraining-label generator.

## Blocking work

1. Add broad real supplier COTS prescriptions, aspheres, achromats, replace/insert/delete and mechanical placement constraints to physical comparison suite.
2. Verify 100 fields and ≥7 wavelengths at adaptive pupil resolution, not just limited reference samples.
3. Refine true exit-pupil illumination, intensity/flux normalization, coatings, vignetting, color and photographic response. Current cos⁴ term remains a proxy.
4. Improve internal stop backwards-ray solver; work done to date offers no performance gain for this important topology.
5. Expand converged Huygens reference vs TorchOptics RS-DIM/ASM comparisons to consistent physical window, transmitted flux and nonparaxial incident fields.
6. Revisit reference MTF acceptance threshold based on actual COTS action **ranking stability** and task tolerances, not the demonstration 0.04 guard.
7. Keep `training_eligible=False` until these are met; provisional training workflow experiments are permitted with explicit provenance.

## Raw bridge results

- https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-3way-gpu-20261027-04/qfilm03_benchmark/three_way.json
- https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-full-speed-20261027-01/qfilm03_full_speed.json
- https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-material-parity-20261027-01/qfilm03_materials.json

## Usage

```python
from evaluator.qfilm02 import Config
from evaluator.qfilm03 import ValidationConfig, evaluate

# Large-scale exploration, no DeepLens overhead:
fast = evaluate(prescription, Config(), "cuda:0", ValidationConfig(mode="fast", base_backend="hybrid"))

# Explicit sparse high-accuracy reference comparison:
cfg = Config(fields=100, pupil_n=128, fft_n=512)
crosschecked = evaluate(
    prescription,
    cfg,
    "cuda:0",
    ValidationConfig(
        mode="verify",
        fields=(0.0, 0.5, 1.0),
        wavelength_nm=550.0,
        huygens_spp=1_000_000,
        base_backend="torch",
    ),
)
assert crosschecked["training_eligible"] is False
```

Qfilm0.3 `verify` requires a DeepLens-compatible supplier glass catalog name and physically matched dispersion or the accepted N-BK7 Sellmeier test glass. Reference errors and unsupported optics are preserved as records; never silently accepted.
