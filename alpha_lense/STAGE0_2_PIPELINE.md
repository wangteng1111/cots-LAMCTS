# Alpha Lense v0.1 — Stage 0–2 Pretrain Data Pipeline

Principle: **Intelligence guides the search. Physics decides the result.**

## Stage 0 — Seed preparation

Input is the high-confidence real-lens-associated prescription corpus already acquired from the public known-lens manifest. Stage 0 is deliberately bounded: it does **not** blindly enumerate tens of thousands of nonexistent patent sibling URLs. It reconstructs/normalizes available seed prescriptions, preserves provenance, removes duplicate optical prescriptions, derives a DesignSpec where supported, and quarantines ambiguous records rather than guessing.

Output: `stage0_seeds.jsonl` with stable prescription hashes and family-safe split keys.

## Stage 1 — Physics-grounded perturbation dataset

For every usable seed prescription x*, generate a multi-scale cloud of physically legal perturbations. Perturbation dimensions are curvature, axial spacing, center thickness, material/index/Abbe, asphere terms, aperture, and topology (add/remove/split/merge element). The random ancestry is not supervision.

Every candidate x is authoritatively evaluated by Q4096. Candidate ranking is feasibility-first under the seed DesignSpec: feasible > infeasible; infeasible candidates are ordered by normalized constraint violation; feasible candidates by final image-quality merit J. Q4096 outputs and evaluator config hash are retained.

Output: `stage1_physics.jsonl` containing source seed, perturbation/edit script, prescription, DesignSpec, Q4096 metrics, feasibility/margins, and rank metadata.

## Stage 2 — Transformer pretrain target assembly

Stage 2 converts Stage 1 physics records into model targets. It creates optical/value/constraint targets and policy targets from physics-grounded local candidate rankings. Policy supervision points from a perturbed design toward actions/candidates that improve the physics ranking; it does not force a random perturbation chain and does not assume the production seed is globally optimal.

Splits are by patent/lens family, never by individual perturbed sample, preventing near-duplicate leakage.

Output: sharded `train/val/test` manifests plus `dataset_summary.json` and evaluator/config hashes. Only Stage 2 output is called the Alpha Lense pretrain dataset.

## Execution gates

1. Stage 0 must report exact usable/quarantine counts.
2. Stage 1 must run a small Q4096 smoke shard and verify units/config hash before scale-out.
3. Stage 1 is resumable and cached by prescription hash.
4. Stage 2 is deterministic from immutable Stage 1 records.
5. Transformer training is a separate later job and is not started by this pipeline.

## Current evaluator limitation

The existing `MultiGPUQ4096Pool` accepts the Mandler discrete state representation only. It cannot authoritatively evaluate arbitrary real-lens prescriptions. Therefore the production Stage 1 pipeline must use a prescription-level Q4096 adapter before large-scale perturbation generation. Falling back to Mandler would invalidate the real-lens pretraining corpus and is prohibited.
