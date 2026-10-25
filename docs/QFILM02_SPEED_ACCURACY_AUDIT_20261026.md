# Alpha Lense Qfilm0.2 — GPU Speed and Numerical Accuracy Audit
Date: 2026-10-26
Status: **NUMERICALLY TESTED / PHYSICS PROVISIONAL / NOT PRETRAIN-CERTIFIED**

## Workstation
`aipss-Rack-Server`, 2 × RTX 5090, Torch 2.14.0+cu130, Triton 3.8.0.
Bridge deployment via `wangteng1111/cots-LAMCTS`; no old evaluator overwritten.

## Implementation
- `evaluator/qfilm02.py`: reference PyTorch CUDA evaluator; pupil FFT, spectral MTF and optics observations.
- `evaluator/qfilm02_triton.py`: opt-in fused ray-marching CUDA kernel; FP64, conic/asphere ray intersection.
- `evaluator/qfilm02_fast.py`: **safe hybrid**. Front-stop geometry uses fused CUDA; internal-stop geometry and unsupported high-order aspheres fall back to reference PyTorch. A Broyden stop-inversion experiment exists behind opt-in `QFILM_TEST_FUSED_INTERNAL=1` but is NOT a performance win.
- `evaluator/qfilm02_convergence.py`: numerical convergence gate; 64/128/256/512/1024 pupil grids, memory-bounded field chunks, per-frequency/per-field error diagnostics, and unconditional training_eligible=False until rigorous physics certification.
- `scripts/qfilm02_triton_bootstrap.py`: non-root Python-header bootstrap with persistent cache (the workstation lacks system Python.h).

## Speed: Fair steady-state paired benchmark
Test: same spherical biconvex prescription, **100 radial field points × 7 wavelengths** on **one RTX 5090**, both implementations warmed with repeated interleaved trials, tensor results matched.

| Pupil grid | PyTorch CUDA median | Triton fused median | Speedup | Q_image max abs difference |
|:--|--:|--:|--:|--:|
| 64×64 | 0.78270 s | 0.05578 s | **14.03×** | 1.07e-11 |
| 128×128 | 0.84800 s | 0.17772 s | **4.77×** | 1.07e-11 |

See `workstation-results/remote/artifacts/qfilm02-steady-benchmark-20261026-01/qfilm02_steady_benchmark.json`.
Older 35× figure from a first-run comparison was inflated by unequal cold/warm overhead; use steady-state figures above.
No general speedup has been demonstrated yet for internal-stop objectives; safe fallback is ~1×.

## Physics consistency tests
- Ray-by-ray `opl`, fixed-film image hits, valid masks match original Torch to about 1e-14 mm on the tested thin and four-surface synthetic lenses.
- Full `Q_image` from fused/ref is matched to ~1e-11 for the tested configs.
- Mathematical tests: clear circular pupil Airy MTF max abs error 0.000398, diffraction cutoff zero handling, sphere/parabola sag/derivative and PSF energy normalization.
- Field validity now also requires chief ray validity for every wavelength, not only a pupil-fill threshold.

## Pupil-grid convergence
Synthetic reference lens, 5 fields × 3 wavelengths (NOT full 100×7), **MTF max absolute differences**:

| Prescription | 256²→512² | 512²→1024² |
|:--|--:|--:|
| Simple biconvex | 0.00706 | 0.00314 |
| Internal-stop 4-surface example | 0.01138 | 0.00754 |

See `workstation-results/remote/artifacts/qfilm02-megagrid-convergence-20261026-01/qfilm02_megagrid_convergence.json`.
64², 128² and even 256² were demonstrated unreliable for some off-axis/high-aberration fields. Do not use global J convergence as a proxy for local MTF convergence. Grid density must be selected by per-field/frequency convergence.

## Next blocking issues before official labels
1. **Off-axis optical propagation**: current FFT uses an approximate exit-pupil/scalar model. Validate versus a rigorous Huygens or trusted reference implementation; include pupil mapping, apodization, obliquity and off-axis distortion.
2. **Full-frame numerics**: test **100 field points × 7 wavelengths**, not only 5 fields × 3 wavelengths; convergence gate must accept every field/frequency.
3. **Illumination**: current `cos^4` term is explicitly a proxy, not a verified irradiance calculation.
4. **Spectral physics**: current n/V to Cauchy dispersion and fixed spectral weights do not represent verified supplier glass, coatings, or photographic response.
5. **General COTS topology**: test many actual prescription families, strong apertures, freeform rejection, glass changes, vignetting/TIR, wavefront fitting and possible caustic behavior.
6. **Internal-stop throughput**: Triton front shooting with Newton and Broyden passes equivalence but is slower than reference. Keep reference fallback until true speedup is demonstrated.
7. **Robust numerical error**: when an action induces phase undersampling or physically invalid fields, reject the candidate's label; never silently substitute proxy value.

## Training gate
At present `certified=False` and `training_eligible=False` are intentional for all Qfilm0.2 outputs. The convergence module can establish numerical stability but does not confer optical-model validity.
Do not launch authoritative Alpha Lense `Action→ΔQfilm` pretraining until physical certification passes.

## Operational notes
`scripts/cuda_venv_exec.py scripts/qfilm02_triton_bootstrap.py qfilm02_accelerated.py` runs optical equivalence checks on the workstation.
`scripts/cuda_venv_exec.py scripts/qfilm02_triton_bootstrap.py qfilm02_steady_benchmark.py` runs performance benchmarks.
The bootstrap installs dev headers in user-owned `/var/lib/cots-lamcts/qfilm02_build_deps`, not system Python. In user code, choose `Config(max_fields_per_chunk=100)` only for moderate pupil grids (e.g. 64 or 128); for 512 or 1024 use `evaluate_multigrid` to bound FFT memory.
