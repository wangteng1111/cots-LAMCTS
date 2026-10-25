# Qfilm0.2 Workstation Smoke Validation — 2026-10-25

## Code
- Evaluator: `evaluator/qfilm02.py`
- Smoke test: `validation/qfilm02_smoke.py`
- Workstation runner: `scripts/qfilm02_server_validation.py`
- Test source commit: `760e290deaeb73b861e4b5229b6cb40b8a370e61`
- Bridge job: `qfilm02-server-cpu-dualgpu-20261025-01`
- Workstation: `aipss-Rack-Server`, 2 x NVIDIA RTX 5090
- Final status: `finished`, `returncode=0`

## Workload
100 radial image-field points, 7 wavelengths, 64x64 pupil samples per field/wavelength.

## Results (single synthetic spherical biconvex test prescription)
|Metric|CPU|GPU0|GPU1|
|---|---:|---:|---:|
|Eval seconds|8.1521|2.2046|2.1402|
|Total process wall seconds|9.7733|4.6295|4.5216|
|J_task|2.45693332014785|2.4569333201478445|2.4569333201478445|

- Maximum CPU/GPU J absolute difference: `5.33e-15`
- Maximum center T-MTF difference: `1.21e-13`
- Maximum edge T-MTF difference: `2.09e-13`
- Eval-only speedup: 3.70x (GPU0) and 3.81x (GPU1).
- Unaberrated Airy MTF maximum absolute error: `0.000398`.
- Exact result artifact: `workstation-results/remote/artifacts/qfilm02-server-cpu-dualgpu-20261025-01/qfilm02_server_summary.json`.

## Physics certification status
**NOT CERTIFIED.** This acceptance test establishes code execution and CPU/GPU consistency only.
The current implementation is `Qfilm0.2-provisional-fft-v1`, with an approximate
exit-pupil FFT, a cos^4 illumination proxy, and approximate dispersion.
Before authoritative pretrain labeling, complete exit-pupil/Huygens reference
verification, pupil sampling convergence, spectral/color response validation,
nonparaxial/off-axis ray and vignetting checks, and a diverse prescription suite.
