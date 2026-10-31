# QfilmRW 0.2: GPU-only execution and audited field contracts

CPU optical solving and CPU full-matrix reference execution are retired. The public evaluator defaults to GPU, rejects backend c and CPU devices, the CLI no longer accepts -c, and the active 11-case driver has CUDA:0 and CUDA:1 lanes only. Historical CPU results and the legacy artifact reader remain for provenance; they are not scheduled by the active pipeline. CPU involvement in host scheduling, metadata/hash/file checks, and the existing geometry preprocessing is not represented as a separate optical solver or CPU reference lane. This does not claim that every preprocessing kernel is a CUDA implementation.

Numerical/source/reference gates remain. CPU/GPU parity is no longer an acceptance prerequisite; GPU-only result collection checks actual device identity, fixed source and implementation hashes, raw PSF checksums/projections and the declared numerical policy. Dropping the CPU lane does not set model_scope_validated=true, create a final J or confer optical correctness.

Off-axis verification now requires an explicit audited object-space angle, fixed across all wavelengths. Public evaluate accepts field_angles_rad; the CLI uses --field-angles-deg. The matrix driver requires --field-contract pointing to a completed GPU field audit or its exported compact contract. Unresolved endpoints are retained as FIELD_CALIBRATION_UNRESOLVED; they are not replaced with a smaller field.

The field audit explicitly distinguishes:

1. Original test: h * half_diagonal(36x24) converted through the imported paraxial EFL. This is a nominal paraxial image height, not necessarily the actual image height in a distorted or non-paraxial system.
2. Source metadata: declared sensor/image-circle radius and focal metadata from the same frozen original files. Source sensor_size and r_sensor may encode a square or rounded nominal geometry; product names do not establish manufacturing specifications.
3. Calibrated native test: target image heights 0, .5*r_sensor, r_sensor are inverted to object-space angles using the aperture-free geometric chief at 550 nm. Actual physical ray transmission is checked independently at all seven wavelengths.
4. Full-frame challenge: target physical half-diagonal 21.63330765 mm is retained as a separate request, even when it lies outside the source's declared image circle. Rounded 21.6 mm sources are labelled as a rounding discrepancy; RF16's 18.2 mm domain is explicitly smaller. A virtual chief ray calibrates coordinates only and does not prove illumination.

The audit uses three independent targeting observations, all running on CUDA: legacy stop aiming, geometric stop aiming with aperture-free iterations followed by final physical aperture validation, and a raster of incident rays aimed directly at the known first-surface sag. Per-surface geometry/aperture losses, stop losses and chief visibility remain in each record. A transmitted ray found by a second targeting method can disprove a legacy zero-ray classification. Failure of all finite samplers alone is not proof of physical darkness.

The legacy aiming code accumulated aperture failures during intermediate guesses; its initial straight-line guess could therefore reject a physically valid final ray. The audited GPU production tracer uses paraxial front-matrix initialization, unclipped geometric stop-map iterations, final-root intersection validation, and then checks every actual clear aperture and the stop. EFL is independently recomputed on the GPU from the frozen exact 550 nm indices. The geometry and glass tables themselves are not changed.

Commands:

```bash
python validation/qfilmrw_field_audit_gpu.py --orchestrate
python validation/qfilmrw_real11.py --orchestrate --field-contract /path/to/field-audit-terminal.json
python -m evaluator.qfilmrw -g --prescription lens.json --field-angles-deg 0 15 30 --output rw.json
```

The last command's angles must come from an applicable field contract; its numbers are syntax examples, not certified lens settings. Both the optical tests and cross-GPU transform checks now execute on CUDA. When CUDA is unavailable, local checks only verify policy/schema and explicitly skip optical computation rather than run a CPU fallback.

The first detailed field audit runs from code 9dd35f42f87aeeca0ff55a8a0bbaf8f4e5fa42e9, request 000-000-020-qfilmrw-gpu-field-audit-20261031-01. Its pinned worktree is independent of later changes to the main branch. The subsequent GPU-only probe checks the new production tracer and field boundaries on both GPUs. Field calibration and ray visibility remain separate from PSF/OTF convergence and independent optical-model validation.
