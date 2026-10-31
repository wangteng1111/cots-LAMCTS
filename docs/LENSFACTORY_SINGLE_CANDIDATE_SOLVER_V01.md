# Lens Factory Single-Candidate Solver v0.1

## Scope

This module implements the continuous optimization and optical evaluation stage for one fixed sequence of commercial optical elements. Element identities, surface radii, center thicknesses, glass data, clear apertures and orientations remain fixed. Only internal air gaps and the back focal length (BFL) are optimization variables.

It is not a reimplementation of Lens Factory's discrete catalog search or splitting strategy, and it is not the original authors' source code.

## Method

### Stage 1: geometric optics

For each configured field and wavelength, sequential rays are traced through the actual surfaces using the existing float64 Qfilm02 Snell-refraction kernel. For each point source, the objective is the mean squared spot error (MSSE) about the spot centroid, or the MSSE of optical path about its mean:

- Spot objective: mean of dx^2 + dy^2 over transmitted pupil rays.
- OPD objective: mean of (OPL - mean(OPL))^2 over transmitted pupil rays.

RGB channels are evaluated independently and averaged, following Lens Factory's stated choice not to combine lateral chromatic displacement into one spot.

For a fixed gap vector, rays exiting the final surface are cached. BFL search changes only the final air propagation and sensor intersection; it does not retrace the preceding surfaces. The paraxial focus centers the BFL search interval. Internal air gaps are seeded with equal 1, 2, 3, 4, 5 and 6 mm spacings plus the supplied catalog spacings. L-BFGS-B provides bounded continuous gap optimization, followed by a small coordinate grid search.

### Stage 2: diffraction MTF

The coarse MTF objective uses the existing Qfilm04 physical exit-plane ray bundle and its direct scalar Rayleigh-Sommerfeld-1 (RS1) integral. The two principal image-plane projections produce tangential (x, field plane) and sagittal (y) complex OTF samples. MTF is the magnitude of OTF. The area under each MTF curve from zero to the configured maximum frequency is maximized.

The default maximum frequency is 100 lp/mm. This frequency limit is an explicit implementation choice for comparability with the Lens Factory plots; it is not a uniquely specified constant in the paper. The dimensionless reported objective is:

    J_LF = 1 - mean(MTF_AUC / f_max)

Lower is better. Stage-2 coarse results are optimizer diagnostics only and never become J.

### Final evaluation

After MTF optimization, each field/wavelength condition is rerun through Qfilm04's convergence planner and solver with the direct RS1 kernel. A final J is returned only if every condition reaches the numerical convergence state and passes the geometry, pupil, support, pitch and joint refinement gates. Otherwise J is null and the condition-level failure states remain in the report.

The default final profile is configurable. Increase the pupil and support sequences for higher-confidence verification. A completed numerical gate is not a proof that the scalar physical model is valid for every thick-lens/high-NA system.

## Model boundaries

The declared illumination is uniform scalar flux over the aperture stop. Fresnel transmission, AR coating transmission, polarization, scatter and manufacturing tolerances are not included. The current Qfilm04 physical model is not independently certified; therefore this solver always reports:

- physics_certified = false
- model_scope_validated = false
- empirically_calibrated = false
- training_eligible = false

This keeps numerical convergence separate from physical-model validation and measured-lens calibration.

## Input prescription

The input is the canonical sequential prescription JSON used by the project:

- surfaces: ordered list of radius, thickness, n_after, v_after, clear_aperture; optional conic, asphere, n_by_lambda_nm, sellmeier.
- stop_z_mm: physical aperture-stop coordinate.
- image_z_mm: sum of sequential surface thicknesses.
- design_spec: positive efl_target_mm and stop_diameter_mm.

For legacy prescriptions with stop_z_mm inside an air interval, the parser splits that interval and inserts a zero-power aperture-stop element. The two adjacent air gaps are then independent optimization variables, matching the Lens Factory element sequence (where an element may be a lens or the stop). The stop moves with its sequence position rather than retaining an arbitrary fractional coordinate inside one gap. Missing clear apertures or missing glass-index data at requested wavelengths are rejected rather than guessed.

## Usage

    python -m evaluator.lensfactory -c --prescription lens.json --output result.json
    python -m evaluator.lensfactory -g --gpu-index 0 --prescription lens.json --output result.json

A fixture with multiple cases requires --case ef50mm_f1.8. Add --no-final-verify to obtain only optimization diagnostics; this always leaves J null. The --quick option is a smoke profile and never produces a final score.

## Validation

The first integration smoke test uses the EF50mm f/1.8 public prescription fixture. It verifies prescription parsing, stop-gap preservation, sequential geometric ray tracing, direct RS1 MTF, OTF normalization at zero frequency and the no-score boundary when final verification is skipped.

This fixture is a public product-associated/patent prescription, not a certified retail manufacturing prescription or measured MTF reference. Passing the smoke test proves software-path execution only, not general optical accuracy.