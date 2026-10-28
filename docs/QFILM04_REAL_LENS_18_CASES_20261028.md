# Qfilm0.4-c / Qfilm0.4-g: commercial lens prescription benchmark
**Prepared:** 2026-10-28. **Updated status: the 18-candidate CPU pilot, 90 common-ray cases and 22 two-solver screening comparisons have now executed. GPU seven-color acceptance remains pending.** See [current CPU evidence and sampling failures](QFILM04_REAL_LENS_CPU_VALIDATION_20261028.md). The remaining preparation and queue sections below are the historical test protocol, not a claim of completed GPU certification. Neither 5um detector sampling nor a successful solver return certifies photographic MTF.

## Version definition
Qfilm0.4 = versioned fixed-film physical exit-plane direct Huygens (ray OPL/OPD and physically mapped exit pupil, Jacobian area amplitude, 256² minimum pupil, complex polychromatic OTF, final scalar J only). `Qfilm0.4-g` uses CUDA GPUs with PyTorch float64, `Qfilm0.4-c` uses CPUs with the **same physical equations and numerical settings**. They are execution variants, **not** different physical models, and neither is certified with `training_eligible=True`. For speed, original `Qfilm0.3` default approximate stop-pupil FFT remains accessible by its original module; do not rename this fast implementation to Qfilm0.4.

- [Interface](../evaluator/qfilm04.py)
- [Physical propagation](../evaluator/qfilm03_physical_huygens.py)
- [Full-image final-score assembly](../evaluator/qfilm03_physical_mode.py)
- [Glass and surface import](../validation/qfilm04_real_prescription_import.py)

Usage, a Qfilm-native prescription:
```bash
python -m evaluator.qfilm04 -g --prescription candidate.json --fields 3 --pupil 256 --detector 128 --output results/qfilm04g.json
python -m evaluator.qfilm04 -c --prescription candidate.json --fields 3 --pupil 256 --detector 128 --output results/qfilm04c.json
```

## Source distinction: commercial product identity != patented optical prescription != actual shipped lens

Sources:
1. [Pinned DeepLens camera prescriptions](https://github.com/vccimaging/DeepLens/tree/911bb9d334fce7e20a4407c4506ac683b52ff246/datasets/lenses/camera). First 11 files below, named for commercially offered camera lenses. Several give a **published patent and Example index**; some lack primary patent provenance. Manufacturer actual factory prescription and unit production MTF are generally private.
2. [Amarthgul/ZemaxSims](https://github.com/Amarthgul/ZemaxSims), independent optical reconstructions of real models; the repository explicitly states that most are patent examples, and are **not optimized**. These seven are fetched from their attributed source ZMX URLs at runtime without storing wholesale unauthorized copies in this repo. Some Zemax files may have unavailable glass libraries or unsupported surface types and are allowed to FAIL.

The [Optical Bench Hub](https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/OpticalBenchHub.htm) emphasizes that a patent example is not guaranteed identical to the lens actually placed into production. Therefore all results must preserve provenance and `actual_manufactured_lens_verified=False`.

## First 11 photographed-lens model-associated prescriptions: independent geometric audit DONE
A separate script-independent reduced-slope ABCD paraxial ray propagation was performed directly on all 11 downloaded published prescriptions in their printed **nd** glass values. This is a preliminary structural audit, **NOT Qfilm0.4 wave-optical PSF/MTF**, and nd differs slightly from 550nm source glass. Raw results: [qfilm04_11_source_paraxial_audit.json](../validation/fixtures/qfilm04_11_source_paraxial_audit.json).

| Model-associated public file | Listed file EFL mm | Independent nd ABCD EFL mm | Surfaces incl stop | Source/reference evidence |
|---|---:|---:|---:|---|
| Canon EF 100mm f/2.8 Macro | 100.03 | 100.04 | 20 | JP 1991-141313 ex1 |
| Canon EF 35mm f/2 | 34.83 | 35.51 | 13 | JP 1992-015611 ex1 |
| Canon EF 40mm f/2.8 | 39.54 | 39.00 | 11 | model file; patent not cited |
| Canon EF 50mm f/1.8 | 50.52 | 51.42 | 12 | JP 1987-087922 ex1 |
| Canon EF 85mm f/1.8 USM | 83.87 | 85.00 | 17 | JP 1993-157964 ex1 |
| Canon RF 16mm f/2.8 STM | **13.05** | **16.48** | 17 | JP 2022-085382 ex3; file itself says example cannot cover full frame |
| Canon RF 24mm f/1.8 Macro | **39.54** | **24.72** | 21 | US 2022/0019061 ex3; file metadata inconsistent |
| Canon RF 35mm f/1.8 Macro | 36.01 | 36.01 | 21 | US 2019/0113711 ex1 |
| Canon RF 50mm f/1.8 STM | 49.57 | 49.56 | 12 | US 2021/0263286 ex1 |
| Sigma 70mm f/2.8 Macro EX DG | 69.17 | 68.91 | 20 | JP 2008-020656 ex1 |
| Yongnuo 50mm f/1.8 | 48.91 | 48.96 | 12 | model file; patent not cited |

**Necessary restrictions:** RF16 and RF24 have inconsistent file EFL metadata; they are retained as anomaly tests, not accepted as matching product optical truth on metadata alone. The imported Qfilm geometry must use the **recomputed physical EFL and original surface dimensions**, not any unreliable `foclen` field; no fitting to the retail nominal value. The RF16 patent example is explicitly NOT certified for full-frame. File-based glass indices are extracted wavelength-by-wavelength from the same pinned DeepLens material instance so the comparison does not conflate dispersion mismatch with propagation physics. Aspheric a2 (if nonzero), shifted/tilted elements and unsupported surfaces are rejected instead of being silently dropped.

## Seven independent cross-source product designs
- Canon EF 50mm f/1.2 L — `Canon/CanonEF50f1.2L/CanonEF50f1.2L.zmx`
- Nikon Nikkor 50mm f/1.4 AI-S — `Nikon/Nikkor50mmf1.4AI-S/Nikkor50mmf1.4AI-S.zmx`
- Carl Zeiss Planar T* 50mm f/1.4 — `Zeiss/PlanarT50f1.4/Zeiss50f1.4.zmx`
- Leica Summicron-M 50mm f/2 — `Leica/SummicronM50f2/SummicronM50f2.zmx`
- Canon FD 28mm f/2.8 S.C. — `Canon/CanonFD28f2.8SC/CanonFD28f2.8SC.zmx`
- Nikon Nikkor 28mm f/2.8 S — `Nikon/Nikkor28mmf2.8S/Nikkor28mmf2.8S.zmx`
- Carl Zeiss Contax Distagon T* 25mm f/2.8 — `Zeiss/DistagonT25f2.8Contax/DistagonT25f2.8Contax.zmx`

No claim made that all seven import successfully until the server runs and independently reports results.

## Full 18-case GPU research protocol

[Runner](../validation/qfilm04_full_18_real_case_benchmark.py) shards the 18 source files into three 6-case tasks. Each lens:
1. Parse ALL actual optical faces, explicit stop, positive physical film coordinate, aspheres and materials; no arbitrary representative or simplified replacement.
2. Preserve source index dispersion on all seven wavelengths 430/470/510/550/590/630/670nm and use the same film for all colors.
3. Qfilm0.4-g computes physical PSF at three 36×24mm radial points (center, 50% and 100% half diagonal), 256² pupil minimum, 128² ×5µm physical detector and complete final J. A field with invalid ray transmission or exit-pupil Jacobian must fail closed.
4. DeepLens independent Huygens computes 250,000 rays per color at exact identical prescription and field on the same fixed film. Its finite-window PSF is **always anchored on the same deterministic green physical chief**. No independent per-color stochastic recentering.
5. Output DeepLens vs Qfilm T/S spatial-frequency 5/10/15/20/30/40/50/60/70/80lp/mm, max/mean/P95 absolute MTF difference, 4px border PSF energy and any failures. A high 4px border fraction is recorded as a **window-truncation risk**, NOT an accepted oracle.
6. Product vs patent identity is tracked separately from physics. Published MTF figures (Canon/Zeiss), if available, require matched focal distance, aperture, spectral weighting and measured-vs-theoretical status. They are **not automatically commensurate** with a patent model's 5–80lp/mm synthetic MTF.

Proof levels:
- L0: filename and product identity exist.
- L1: physical prescription parsed and independently paraxial EFL/stop/film consistency checked.
- L2: Qfilm0.4 result and CPU/GPU parity passed.
- L3: sufficiently sampled *independent* optics simulator agrees in field/wavelength/MTF and reference window convergence.
- L4: actual product measured MTF agrees at same aperture, wavelength band and focus. L4 is required before claiming **empirical real lens accuracy**.

The `quality` output and final J remain *provisional* even on L3 because relative illumination, film response, coatings and manufacturing tolerance are not fully calibrated.

## Submitted but unclaimed job IDs (as of 2026-10-28 ~19:40 Beijing)
- `qfilm04-11-real-lens-pilot-20261028-01` — native camera-11 basic physical 550nm GPU run, two fields
- `qfilm04-seven-independent-patents-20261028-01` — independent Zemax-7 import + basic GPU run
- `qfilm04-real-lens-cpu-gpu-parity-20261028-01` — same Canon EF50 f/1.8 patent example on -c and -g with same math
- `qfilm04-full-18-real-shard0-20261028-01`
- `qfilm04-full-18-real-shard1-20261028-01`
- `qfilm04-full-18-real-shard2-20261028-01`

**Infra blocker:** latest known `workstation-results/remote/agent_status.json` heartbeat 2026-10-28T09:57:11Z; actual time approx 11:40Z, >100 minutes stale. Only permitted workstation access is via GitHub Bridge; writing GitHub queue files does **NOT** restart systemd service. No output artifact exists for any of the above jobs at preparation time. Bridge must resume polling or be restarted by someone with workstation OS access:
```bash
sudo systemctl status cots-lamcts-bridge.service
sudo journalctl -u cots-lamcts-bridge.service -n 100 --no-pager
sudo systemctl restart cots-lamcts-bridge.service
```
Service name/location should be verified on the workstation against current local installation and secret permissions. Do not fabricate GPU or CPU results.

## Additional physical benchmark references
- Canon reading manufacturer MTF and warning regarding design-simulated curves: https://www.usa.canon.com/learning/training-articles/training-articles-list/reading-and-understanding-lens-mtf-charts
- Carl Zeiss Planar 50mm f/1.4 manufacturer MTF at 10,20,40 lp/mm, nominal infinity and wide open: https://www.zeiss.com/content/dam/pno/downloads/industrial-lenses/datasheets/en/classic-lenses/datasheet-zeiss-planar-1450-industrial-lens.pdf
- Canon RF24 manufacturer product and lens-element specification: https://global.canon/en/c-museum/product/rf519.html
