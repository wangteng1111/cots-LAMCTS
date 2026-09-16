# Alpha Lense Dataset v0.1 — generation and review gates

## Scope
This stage prepares optical prescription data only. It MUST NOT start Transformer pretraining.

## Provenance classes
- `production_associated`: prescription URL is explicitly associated with a named real lens in the public known-lens manifest. This does **not** prove the patent example is byte-for-byte the final production prescription.
- `patent_only`: sibling example discovered from the same patent family; no production identity is asserted.

## Funnel
Report every stage separately:
`attempted -> downloaded -> raw unique -> normalized unique -> conservatively parseable -> reconstructed -> physics valid -> usable`.
Never report attempted/downloaded as training examples.

## Current generation job
The generator probes known prescriptions plus sibling Examples 01..12. It stores raw text, SHA256, normalized SHA256, provenance, patent family and conservative structural signals. `parseable_conservative` is only an ingestion gate; it is not a claim that surfaces/glasses/stops have been reconstructed correctly.

## Review before pretraining
1. Manually inspect representative prime, zoom, fisheye, telephoto, asphere and missing-glass prescriptions.
2. Derive and test an exact P2P format parser against those fixtures.
3. Reconstruct ordered surfaces, thickness/air gaps, stop, material n/V or glass identity, asphere coefficients, element/group topology, and zoom configurations where present.
4. Reject or quarantine ambiguous/incomplete records rather than guessing.
5. Deduplicate normalized prescriptions and split by patent/lens family to prevent train/validation leakage.
6. Run paraxial/physics sanity checks; later run the authoritative optical evaluator where compatible.
7. Only the final physics-valid manifest may feed Stage-I pretraining.

## Topology requirement
Dataset representation must preserve variable element count and group topology. Future perturbation/pretraining may include add/remove/split/merge element operations, so fixed-length lens-ID encodings are not acceptable as the canonical representation.
