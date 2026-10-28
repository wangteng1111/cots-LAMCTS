# Qfilm0.3 vs upstream DeepLens Huygens — 36 × 24 mm full-frame image-plane accuracy audit
**Run date:** 2026-10-28  
**Code:** [full-frame 11×7 benchmark](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/qfilm03_fullframe_spectral_audit.py).  
**Raw machine-readable results:** [154 optical conditions](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-fullframe-11x7-20261028-02/qfilm03_fullframe_spectral/fullframe_11field_7wave_huygens.json).

## Scope and caveat
This is **solver-to-solver disagreement**, not a metrologically calibrated error versus a manufactured lens. DeepLens 1M Huygens samples and 0.64mm PSF crop have not been established as converged for strong-aberration/red-light cases. Optical material/model has no measured manufacturing tolerance, coatings or photometry. **Do not set training_eligible=True.**

## Common conditions
- Exactly matched physical sequential prescription, supplier N-BK7 Sellmeier dispersion and mechanical image plane.
- Two *synthetic, converging*, rotationally symmetric lenses: (a) N-BK7 spherical singlet, front stop, R=+50/-50mm, central 5mm glass, fixed film z=52.55mm, target ~50mm EFL; (b) two separated N-BK7 singlets with one conic/aspheric face, stop inside stack, fixed film z=51.4228mm, EFL ~43.45mm.
- **36×24mm physical image field**, radial half-diagonal 21.63330765mm, 11 normalized fields h=0..1 in steps .1. DeepLens uses larger *sensor metadata* 48mm/9600pixel to achieve 5µm detector pixel pitch; it does **not** propagate a 9600² ASM.
- Seven monochromatic wavelengths 430/470/510/550/590/630/670nm. Physical lens and film unchanged across wavelengths: **no per-λ refocus**.
- Qfilm pupil 512², FFT 2048², complex OTF read at [5,10,15,20,30,40,50,60,70,80] lp/mm, T/S directions aligned to field radial direction.
- Native unmodified DeepLens `GeoLens.psf_huygens()` upstream revision 911bb9d334fce7e20a4407c4506ac683b52ff246. 1 million rays per PSF, scalar Huygens integration, 128×128 detector patch = 0.64mm field diameter, PSF normalized **within crop**, detector sampling Nyquist 100lp/mm.
- Object is at depth -1,000,000mm (1km), nominal infinity. Internal DeepLens `calc_scale=-depth/foclen` converts normalized source position to physical source coordinates. This is checked against target field angle; exact-infinity difference requires independent depth-convergence test.
- DeepLens PSF `recenter=True` uses lens chief-centering; monochromatic PSF magnitude-MTF is mostly translation invariant within a sufficiently large physical window, **but polychromatic PSF centroid and absolute LCA/FLCA are not certified by these monochromatic checks**.
- Single GPU RTX5090; 77×2=154 conditions, each with T/S frequency probe (3080 MTF differences), zero tool execution errors, zero Qfilm-invalid conditions.

## Per-wavelength errors
`max_abs`: maximum single absolute MTF difference across 11 fields, 2 directions and 10 frequencies. `mean_per_MTF`: actual mean across those 220 values per wavelength.

| Wavelength (nm) | Singlet max abs | Internal stop max abs | Singlet mean per MTF | Internal stop mean per MTF |
|---|---:|---:|---:|---:|
|430|0.02156|0.03996|0.01431|0.01979|
|470|0.03454|0.04004|0.01926|0.02073|
|510|0.04170|0.04305|0.01841|0.02455|
|550|0.02958|0.03615|0.01851|0.02283|
|590|0.05317|0.07909|0.02002|0.02802|
|630|0.11090|0.33441|0.03108|0.06418|
|670|0.30825|0.27318|0.06796|0.06549|

## By spatial-frequency band (maximum MTF absolute difference)
| λ nm | Singlet 5–20 | Singlet 30–50 | Singlet 60–80 | Internal 5–20 | Internal 30–50 | Internal 60–80 |
|---|---:|---:|---:|---:|---:|---:|
|430|0.02156|0.00608|0.00569|0.03996|0.00776|0.01072|
|470|0.03454|0.00731|0.00741|0.04004|0.01021|0.00930|
|510|0.04170|0.00632|0.01219|0.04305|0.01305|0.00854|
|550|0.02958|0.01593|0.01644|0.03615|0.01511|0.01175|
|590|0.03209|0.03630|0.05317|0.05971|0.05750|0.07909|
|630|0.02183|0.03060|0.11090|0.13596|0.28374|0.33441|
|670|0.10010|0.25806|0.30825|0.14956|0.27318|0.20301|

## By radial image height (maximum across 7 λ × 20 MTF values)
| h | Physical image height mm | Singlet max | Internal stop max |
|---:|---:|---:|---:|
|0|0.00000|0.30825|0.33441|
|1|21.63331|0.01925|0.01690|
|0.1|2.16333|0.23408|0.27318|
|0.2|4.32666|0.04335|0.08512|
|0.3|6.48999|0.02287|0.02511|
|0.4|8.65332|0.01867|0.04004|
|0.5|10.81665|0.04170|0.04017|
|0.6|12.97998|0.02589|0.03162|
|0.7|15.14332|0.02503|0.03660|
|0.8|17.30665|0.02388|0.02511|
|0.9|19.46998|0.02295|0.01768|

## Whole-test statistics
- **singlet**: 77 monochromatic PSFs; mean over all individual T/S/frequency MTF absolute errors 0.00810; max 0.30825; P95 of *each per-PSF maximum difference* 0.05812; 8 of 77 conditions above 0.03 and 3 above 0.10; median native 1M-ray Huygens runtime ~4.052s per PSF.
- **internal_stop_asphere**: 77 monochromatic PSFs; mean over all individual T/S/frequency MTF absolute errors 0.01299; max 0.33441; P95 of *each per-PSF maximum difference* 0.10171; 28 of 77 conditions above 0.03 and 4 above 0.10; median native 1M-ray Huygens runtime ~4.073s per PSF.

## Interpretation
1. Blue and green wavelengths mostly agree within ~0.01–0.04 maximum per condition. But the 0.03 threshold is **not met everywhere**.
2. A severe axial red disagreement is reproducible: 670nm singlet h=0 max ~0.308; internal stop 630nm h=0 max ~0.334. This is a solver discrepancy, **not a certified absolute error**.
3. For many edge fields both models predict very low MTF due to strong aberrations, so small `|ΔMTF|` at the far corner **does not establish high image quality or accurate energy throughput**.
4. DeepLens Huygens Monte Carlo ray count, 0.64mm PSF crop/normalization and approximate finite object distance can bias MTF. A separate [1M vs 4M / larger-window test](https://github.com/wangteng1111/cots-LAMCTS/blob/main/validation/qfilm03_fullframe_reference_convergence.py) is required to isolate red disagreements.
5. Underlying Qfilm scalar paraxial pupil-FFT still uses unit pupil amplitude, proxy irradiance and approximate exit-pupil mapping. It may have systematic errors even where 512² pupil numerically converges.
6. Independent previous DeepLens-vs-TorchOptics ASM shared-input test agreed to 8.65e-14 MTF; this validates the ASM kernel, **not pupil-field reconstruction**.
7. Fixed-weight seven-wave integrated MTF is **not** an empirical photographic-film response. The physically co-registered monochromatic Huygens PSF sum is now measured and reported below.

## Incoherent polychromatic PSF and MTF (same physical fixed film)
7 spectral bands [430,470,510,550,590,630,670] nm, weights [0.075,0.125,0.16,0.28,0.16,0.125,0.075]. DeepLens outputs each wavelength's PSF on a common 5µm primary-550nm-chief-referenced 128² grid before per-color *incoherent intensity* summation; Qfilm sums complex monochromatic OTF with physical lateral-chromatic offsets. 1 million Huygens rays per color. Qfilm512, full 36×24mm.

| Radial field h | Singlet max absolute Δpoly-MTF | Internal-stop max absolute Δpoly-MTF |
|---:|---:|---:|
| 0.0 (center) | 0.04868 | 0.06723 |
| 0.5 (10.8167mm) | 0.02051 | 0.01881 |
| 1.0 (21.6333mm) | 0.01705 | 0.01258 |

While the worst single-color disagreements are ~0.31–0.33, integrated PSF errors are smaller due to color-weighted intensity addition. However the center-field errors of ~0.05–0.07 remain above proposed high-quality label criteria. Chromatic window truncation, source-spectrum weights and photographic emulsion spectral response remain unverified.

[Polychromatic reference raw data](https://github.com/wangteng1111/cots-LAMCTS/blob/workstation-results/remote/artifacts/qfilm03-fullframe-polychromatic-20261028-01/qfilm03_poly_fullframe/polychromatic_huygens_36x24.json).

## Raw data exports
- [154-row scene summary CSV](https://github.com/wangteng1111/cots-LAMCTS/blob/main/artifacts/qfilm03_fullframe_154_cases_20261028.csv)
- [3080-row per-frequency T/S CSV](https://github.com/wangteng1111/cots-LAMCTS/blob/main/artifacts/qfilm03_fullframe_3080_mtf_20261028.csv)

**Training-grade physics status: provisional; candidate PSF must pass adaptive pupil, reference wavefield, frequency-band and window convergence with the same physical detector model.**
