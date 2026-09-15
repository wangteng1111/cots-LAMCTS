# Alpha Lense v0.1 — 10k+ commercial/patent prescription corpus plan

## Objective
Build a provenance-preserving corpus of >=10,000 usable photographic-lens prescriptions for optical representation and goal-directed policy pretraining. A record is useful for goal-policy training only when the prescription is sufficiently complete to reconstruct surfaces/materials/spacings and infer a DesignSpec. Model names or calibration-only profiles are metadata, not prescriptions.

## Primary source identified
PhotonsToPhotos Optical Bench currently states that it contains about **10,000 optical prescriptions**, with a public hub table containing >1,280 prescriptions associated with known lenses. Many prescriptions originate from patents, and the site explicitly warns that a patent example is not guaranteed to equal the final production lens.

Therefore records must carry `source_type` and must never silently label every patent example as an exact commercial production prescription.

Target source classes:
1. known-production-associated Optical Bench records;
2. patent optical prescriptions linked to identifiable commercial lens families;
3. additional public patent examples with complete prescriptions, retained as patent-derived seeds;
4. manufacturer/patent/public optical-design sources added later after provenance and license review.

Lensfun is useful for lens identity, focal/aperture ranges, mounts and calibration metadata, but it is not by itself a full optical prescription source. It should be used for metadata matching/validation, not fabricated surface prescriptions.

## Canonical schema
Each seed should preserve:
- `seed_id`, source URL/reference, source type, patent/example identifier;
- maker/model when supported;
- production-match confidence: exact / likely / family / patent-only / unknown;
- focal range and aperture/T-stop information;
- format/image circle when known;
- complete ordered surfaces: radius/curvature, axial thickness/air gap, material/glass, refractive index/dispersion, clear aperture, asphere coefficients;
- stop position;
- element/group count and topology;
- BFD/total length where available;
- provenance per field and parser confidence;
- raw source checksum and normalized prescription checksum.

## Deduplication
Do not deduplicate merely by lens model name. Patent examples and zoom states may share names but differ optically. Use normalized prescription hashes plus patent/example/scenario metadata. Maintain aliases separately.

## Topology perturbation requirement
Alpha Lense v0.1 perturbations explicitly include **lens element count**. The mutation grammar must support:
- add an element in a legal axial interval;
- remove an element;
- split one element into a cemented/air-spaced pair;
- merge a compatible adjacent pair;
- plus continuous perturbations of curvature, spacing, thickness, refractive index, Abbe number, asphere terms and aperture.

Topology-changing mutations require a variable-length surface representation and legality checks (positive thickness/gaps, material validity, no impossible overlap, aperture/package constraints). The Transformer already supports masked variable surface counts; policy/action encodings must include topology deltas.

## Goal-directed labels
For a seed lens, first evaluate/refine a feasible basin goal `x*` under its inferred/declared DesignSpec. Generate multi-scale perturbation clouds up to nominal depth 10. Every training label points directly:

`(x_perturbed, DesignSpec) -> optical adjustment toward x*`

The random perturbation ancestry is NOT the policy target. Depth is only a curriculum/difficulty variable.

## Acquisition phases
- Phase A: inventory and parsers; collect source metadata and raw references.
- Phase B: normalize complete prescriptions and materials; compute element/group topology.
- Phase C: deduplicate and classify production-match confidence.
- Phase D: authoritative physics validation and DesignSpec inference; reject/flag non-reconstructable records.
- Phase E: create train/validation/test splits by lens family/patent family to prevent near-duplicate leakage.
- Phase F: generate goal-directed perturbation clouds on the workstation and pretrain.

## Data-quality gates
A 10k headline count is not sufficient. Report separately:
- raw prescription records;
- parseable complete prescriptions;
- physics-valid prescriptions;
- DesignSpec-feasible seeds;
- production-associated subset;
- unique optical families after deduplication.

The scientific benchmark must disclose these counts and must not call patent-only examples confirmed production lenses.
