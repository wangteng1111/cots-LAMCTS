# Alpha Lense — Upstream DeepLens vs Qfilm0.2 Reproduction (2026-10-27)

## Executive finding
We executed **the original upstream DeepLens `GeoLens.psf_huygens` and `GeoLens.psf_coherent` methods, not home-grown reimplementations**, on the workstation `aipss-Rack-Server` with RTX 5090. Upstream repository revision: `911bb9d334fce7e20a4407c4506ac683b52ff246`.

Same-lens physically matched experiments find **good agreement between Qfilm0.2 pupil-FFT MTF and DeepLens million-ray Huygens MTF** for two converging reference lenses. Prior large errors were partly caused by inadequate coherent Monte Carlo samples, by comparing T/S in different image orientations, and by using an inappropriate negative-EFL test lens.

DeepLens coherent pupil-field + ASM currently differs from BOTH Huygens and Qfilm; it improves noticeably with pixel grid resolution, hence is **not yet converged** at the tested settings.

## Source and reproducibility
- Upstream: https://github.com/vccimaging/DeepLens/tree/911bb9d334fce7e20a4407c4506ac683b52ff246
- Our bridge runner: `validation/qfilm_deeplens_same_prescription.py`
- Native prescription converter maps physical curvature, conic/aspheric, exact axial spacings and aperture position. DeepLens uses the same N-BK7 Sellmeier glass as Qfilm. The coordinate origin is shifted when the stop precedes the first surface; this does not change geometric physics.
- Single wavelength: 550 nm, common fixed film, 3 fields (center/mid/edge), 10 MTF frequencies [5,10,15,20,30,40,50,60,70,80] lp/mm, DeepLens Huygens `spp=1,000,000`, detector pixel pitch `0.00416667 mm`, 96-pixel patch (0.4 mm physical window).
- Qfilm pupil grid tested: 128², 256², 512²; FFT grid 4× pupil samples along each axis.
- Important: Qfilm field rotation = +X while DeepLens source field orientation = +Y. **T/S axes must be swapped for off-axis comparisons**; prior comparisons of raw pixel-x frequency were incorrect.
- Python extras required by upstream plotting / LPIPS modules cannot be installed from PyPI (network restricted) and Ubuntu binaries built for NumPy 1.x are incompatible with the workstation NumPy 2.x. The comparison wrapper supplies **import-time placeholders for optional plotting/image I/O/LPIPS/surrogate network modules**. **No upstream GeoLens optical ray tracer, Huygens or ASM function is modified.** Torch remains `2.14.0+cu130`, original shared environment unchanged.

## Geometry parity
| Design | DeepLens EFL mm | Qfilm EFL mm | Absolute difference mm |
|---|---:|---:|---:|
| N-BK7 biconvex singlet, front stop |49.0513934159|49.0513928700|0.000000546|
| Two-singlet positive EFL, internal stop, conic/aspheric surface |43.4488493234|43.4488488609|0.000000462|

An earlier 4-surface "rich lens" benchmark had **negative EFL ~ -470 mm** and was unsuitable for a real image-lens comparison; not used as positive-lens evidence.

## Maximum absolute MTF disagreement (max of aligned T/S over 10 frequencies per field)
Field coordinates 0=center, 0.5=mid, 1=edge.  Each result compares Qfilm and **DeepLens million-ray Huygens**.

### N-BK7 biconvex singlet
| Qfilm pupil | Center | Mid-field | Edge |
|---|---:|---:|---:|
|p128|0.00900|0.07873|0.21729|
|p256|0.00689|0.01350|0.01687|
|p512|0.00739|0.01329|0.01588|

### Internal stop, two COTS-like singlets, one non-spherical surface
| Qfilm pupil | Center | Mid-field | Edge |
|---|---:|---:|---:|
|p128|0.02667|0.02207|0.03527|
|p256|0.02514|0.02237|0.03076|
|p512|0.02453|0.02112|0.02725|

The 512² results support Qfilm consistency for these tested **positive EFL** systems; they are NOT a general physics certificate for all COTS topologies and 100×7 fields/wavelengths.

## DeepLens coherent ASM convergence (biconvex center, fixed 0.4mm PSF window)
| DeepLens sensor pupil grid | Coherent rays | Max ΔMTF vs Qfilm 128² | Coherent PSF runtime |
|---|---:|---:|---:|
|3072|1,000,000|0.37061|0.357 s|
|3072|4,000,000|0.37778|0.368 s|
|4096|4,000,000|0.18749|0.384 s|
|6144|4,000,000|0.07926|0.450 s|

At fixed grid 3072, increasing rays 1M→4M barely changes discrepancy. Increasing spatial grid 3072→4096→6144 significantly improves agreement, identifying a **wavefront discretization / sampling convergence issue** rather than simply ray-count insufficiency. It does not establish that Qfilm is universally correct.

DeepLens 1M-ray Huygens on these cases took approximately 2.3s/field; DeepLens coherent ASM with 1M or 4M rays took about 0.35–0.45s/center PSF at these grids (warm GPU; not comparable full 100×7 throughput).

## Remaining work before pretraining-label certification
1. Compare all 100 radial fields and all 7 wavelengths; not just 3 fields × one λ.
2. Independently converge Huygens spp, finite PSF physical window, pixel pitch, and Qfilm pupil n on hard lenses.
3. Derive physically correct film irradiance/illumination (current Qfilm cos⁴ proxy), spectral weights/film response, PSF flux and vignetting.
4. Test real supplier COTS singlets, achromats, aspheres, aperture placements and invalid/caustic topologies.
5. Verify DeepLens ASM/exit-pupil grid convergence at higher spatial resolution and compare normalized physical-plane PSF with common coordinates.
6. Complete FFT propagation error budget, GPU internal-stop throughput, and authoritative 100×7 convergence gates.

**Do NOT change `training_eligible=False` yet.** DeepLens validation greatly improves confidence in the current Qfilm wavefront-based MTF, but it is not yet a blanket certification.

## Bridge raw result artifacts
- [Qfilm128 internal lens](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm-deeplens-internal-positive-million-20261027-01/deeplens_qfilm_comparison.json)
- [Qfilm256 both positive lenses](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm-deeplens-qfilm256-million-20261027-01/deeplens_qfilm_comparison.json)
- [Qfilm512 both positive lenses](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm-deeplens-qfilm512-million-20261027-01/deeplens_qfilm_comparison.json)
- [DeepLens coherent 4M 3072 grid](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm-deeplens-coherent-4m-20261027-01/deeplens_qfilm_comparison.json)
- [DeepLens coherent 4096 grid](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm-deeplens-coherent-grid4096-20261027-01/deeplens_qfilm_comparison.json)
- [DeepLens coherent 6144 grid](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm-deeplens-coherent-grid6144-20261027-01/deeplens_qfilm_comparison.json)
