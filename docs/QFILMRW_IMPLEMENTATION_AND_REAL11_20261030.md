# QfilmRW 0.1 — implementation and 11-prescription test contract

QfilmRW is an executable, separately named ray-wave evaluator. This first release implements the **local Fourier branch** of the Qfilm 0.5 proposal. General compensated ASM, beamlet/multi-branch and vector propagation are not implemented. It is not a replacement validated for all thick photographic lenses.

## Code and use

- `evaluator/qfilmrw/contracts.py`: versioned task, normalization/window, numerical/resource policy.
- `rays.py`: true sequential refracting geometry, OPD, physical exit plane, amplitude, area and ray power stored separately. Reuses the existing float64 ray primitives and exit-envelope bound, not an old PSF solver.
- `fourier.py`: reference-sphere phase, angular-coordinate field reconstruction, float64/complex128 Bluestein CZT, requested-frequency full-plane pupil autocorrelation, windowed complex OTF, exact-distance sparse RS1 reference.
- `solver.py`: pupil/angular-grid/window/pitch/joint gates, resource admission, explicit failure states and error ledger.
- `api.py`, `__main__.py`: `evaluate`, CPU `-c`, CUDA `-g --gpu-index N`. There is no implicit CPU fallback.
- `validation/qfilmrw_real11.py`: three execution lanes (CPU, CUDA 0, CUDA 1), immutable per-condition records, raw PSF checksums and independent collector.
- `validation/qfilmrw_tests.py`: numerical identities, analytic Airy, angular density/Jacobian, spectral centers, failure/label boundaries, actual CUDA tests when available.

```python
from evaluator.qfilmrw import evaluate, Task, Policy
result = evaluate(prescription, fields=(0., .5, 1.), backend='g', gpu_index=0)
```

```bash
python -m evaluator.qfilmrw -c --prescription lens.json --output rw.json
python validation/qfilmrw_real11.py --orchestrate
```

## Physical and numerical boundary

The source is an explicitly declared uniform-stop scalar z-flux model, without coating/Fresnel transmission or polarization. Field amplitude and area are separate; exit-area and angular-map Jacobians each enter their respective coordinate conversion once. Zero transmitted samples are reported as `RAY_AIMING_OR_OCCLUSION_UNRESOLVED`, not a proven blocked or defective lens. Sparse rays and folded exit/angular mappings have separate states.

For source point P and fixed film reference C, set R0=|C-P|. The factored phase is OPL+R0, and angular coordinates are q=(Pxy-Cxy)/(lambda R0). The local model integrates the reconstructed angular density against exp(-2 pi i q·delta). A declared common detector quadratic phase is restored for sparse complex-field reference comparisons. This is a Taylor model of the exact spherical propagation kernel: source-dependent curvature and higher terms remain approximate. It is **not** arbitrary dechirping followed by unchanged ASM.

The boundary operator is scalar RS1 with its exact reference-point prefactor. `direct_rs1` uses the same source state, transport convention and exact source-to-target distances. It is an internal boundary/operator comparison, not independent validation of the lens ray tracer, actual energy transport, commercial Huygens implementation, or real manufactured lenses. Passed sparse probes do not bound unsampled pixels.

The angular field uses inverse-map bilinear interpolation. One-sided boundary derivatives avoid the old repeated erosion of the pupil. Where Newton fails, neighboring-source simplices are used; triangles bridging clipped gaps are excluded. Excluded angular area and unresolved query counts remain in the records. Boundary quadrature is tested by pupil refinement, not claimed to have a proven global energy bound.

PSF output retains **0.25 um point pitch and a 0.64 mm square window (2560² points)**. Pixel-area integration is a distinct future detector operation. Verification uses 0.125 um, a 0.96 mm window and joint refinement. Border-band flux is not an omitted-tail bound.

Default task OTF is the normalized finite-window complex OTF. `pupil_autocorrelation` exposes a separate full-plane local-Fourier diagnostic; it never silently replaces the finite-window objective. Full-plane convergence acceptance is not implemented in 0.1. No validated energy bound establishes equivalence of these two observables.

Existing engineering optical tolerances are retained: pupil/window/angular-grid complex OTF 0.002, pitch 0.001, joint 0.005, relative raw integral 0.005, residual source-phase diagnostic 0.25 cycles. Numerical gates remain estimates, not strict continuous-domain bounds. The sparse same-boundary RS1 complex-field discrepancy gate is 0.005. A failed numerical or model gate produces no final J.

`numerically_verified`, `model_scope_validated`, and `empirically_calibrated` are separate axes. All 0.1 model-scope/empirical flags initially remain false; diagnostic MTFs cannot become training truth. A future validated simulated J may train a surrogate without empirical calibration, but predicting measured-lens performance needs separate empirical evidence. The existing score formula is retained behind validation; missing geometry/radiometry is not synthesized or copied from an uncertified old FFT.

## Same 11 prescriptions

`validation/fixtures/qfilmrw_real11.json` freezes the exact same imported geometry and 7-wavelength material-index tables. Original source commit: `911bb9d334fce7e20a4407c4506ac683b52ff246`; all 11 original byte SHA256 values are checked against the preceding Qfilm04 fixture. Fixture SHA256: `09f589d4859070a036de73fca4a7c4287edf393abcd484d009e6534f81ac7954`. `qfilmrw_export_real11.py` reproduces the conversion from that source.

EF100/2.8, EF35/2, EF40/2.8, EF50/1.8, EF85/1.8, RF16/2.8, RF24/1.8, RF35/1.8, RF50/1.8, Sigma70/2.8, Yongnuo50/1.8.

These are product-named public optical/patent files, not certified factory prescriptions or physical measurements.

Full workstation matrix: **11 × 3 fields (0, .5, 1) × 7 wavelengths (430–670 nm) = 231 conditions**, each with CPU and one actual GPU execution (462 device-condition executions). Two GPU lanes partition the lenses; the CPU lane runs independently. Grid schedules, source identities and code commits are checked before arithmetic parity. Both actual GPUs run the unit parity preflight. Final collector recomputes OTF/integrals/refinement decisions from projections, and verifies saved raw arrays/checksums. CPU/GPU arithmetic parity alone is not optical accuracy.

Survey mode uses pupils 64/128, full 0.25 um / 0.64 mm PSFs, and sparse references; it intentionally omits full verification and never labels a condition numerically accepted. Verify mode uses 128/256/512/1024, angular refinement and the further gates. Time/memory limits are explicit and produce `RESOURCE_LIMIT`; no budget-driven tolerance change is allowed.

The workstation defaults to a 300 million transform-working-set element admission cap. This is a conservative implementation resource policy, not a measured byte-accurate peak memory bound. CUDA allocation failure remains a resource terminal. A low-memory local run can explicitly use a lower cap without being represented as the workstation full verification policy.

## Workstation handover

Bridge 0.6.0 did not act on stop controls for adopted legacy workers and let those workers block new requests. Commit `7b8766c9d6e4da5ddebbe46397d5c792b8508bf9` adds Bridge 0.6.1 identity-checked SIGTERM / delayed SIGKILL, pre-claim cancellation filtering, adopted-process health, and oldest-pending fairness within publication priority. It preserves native checkpoints and never signals the inherited systemd process group. 125 old Qfilm queue/prepared requests received explicit stop/cancel controls.

The checksum-pinned activation script `scripts/bootstrap_bridge_qfilmrw_host.sh` installs that runtime, retains KillMode=process and restricts claims to `^000-000-020-qfilmrw-`. The old deployed runtime cannot remotely execute this upgrade while its adopted-worker queue gate is closed; host activation is required. Until a 0.6.1 heartbeat and disappearance of old job identities are observed, old tasks must not be reported as stopped and new tests must not be reported as running. The QfilmRW driver additionally refuses to run if old job processes remain.

A submitted queue request is not evidence of a claimed, running or accepted computation. Local CPU survey, workstation CPU/GPU execution, numerical convergence, model validity and empirical performance must be reported separately.
