# OPT1.0 — Optical Pretrained Transformer 1.0

## Definition

OPT1.0 is a physics-grounded active optical search model pretrained on human-designed optical prescriptions for structural representation learning, then iteratively improved exclusively by high-fidelity physical evaluation, with persistent out-of-distribution exploration to permit departure from the human design manifold.

Its governing principles are:

1. **Human data initializes.** Existing human-designed lens prescriptions are used only to learn reusable optical representations: surface sequences, curvature/power relationships, glass transitions, stop placement, spacing patterns, symmetry/asymmetry and other structural regularities. Human designs do not define the final search space or the final quality target.
2. **Physics judges.** Once the model is fine-tuned for the COTS task, all quality supervision comes from the authoritative Q4096 production evaluator. Human quality labels, catalog prestige, lens family names and SKU identity are excluded from the final optimization target.
3. **Search is never confined to the human manifold.** Persistent global/out-of-distribution exploration is retained so the system can discover physically superior structures that do not resemble conventional human lens forms.
4. **The Transformer is not the evaluator.** The Transformer predicts/ranks candidate designs and may later guide MCTS/LA-MCTS-style exploration; authoritative labels remain Q4096 final optical evaluations.
5. **Physical prescription, not identity.** Learner inputs are physical optical quantities. Vendor, product, patent-example and catalog IDs are audit/procurement metadata only.

## Intended training loop

### Stage A — optical representation pretraining

Train a variable-length surface-sequence Transformer on a large corpus of human-designed prescriptions using self-supervised objectives such as masked-surface/field reconstruction, next-surface prediction, contrastive structure learning and physics-informed perturbation ranking. No COTS task quality labels are required in this stage.

### Stage B — physics-grounded fine-tuning

Fine-tune the pretrained encoder on authoritative COTS pairs `(design, J_Q4096)`. The primary prediction/ranking target is monotonic with the single final merit J (lower is better). Tail-aware regression and pairwise ranking should emphasize low-J designs without introducing an alternate optical evaluator.

### Stage C — active self-improvement

Use an ensemble to rank a large candidate pool with an acquisition function combining predicted quality and uncertainty. Evaluate only selected candidates with Q4096, append those labels to the dataset and retrain. Candidate generation must permanently retain a nonzero global/OOD exploration component.

## Relationship to zero-human-knowledge search

OPT1.0 uses human optical knowledge as a warm start, not as a ceiling. A corresponding `OPT0-Zero` baseline should start from random model weights and learn only from self-generated Q4096 labels. Long-horizon experiments should compare whether OPT1.0 preserves its sample-efficiency advantage while still leaving the human prescription manifold when physics rewards doing so.

## Current authoritative evaluator

OPT1.0 quality supervision is grounded in `market_final_direct_q4096_v4`, evaluator hash:

`0dcde5089bd66deba991214558637fe906f6c2a52530619850e6c5bab0d684da`

The authoritative scalar is `J` (lower is better); search may use `-J`. Display-only transformed quality scores must never be used as training ground truth.

## Status

The first container-scale pretraining meta-test uses a deliberately tiny public/proxy corpus and therefore tests transfer feasibility only. It does not constitute validation of full-scale OPT1.0. Full validation requires a much larger and provenance-graded prescription corpus plus prospective Q4096 comparisons against scratch Transformer, random search and previous SVM/LA-MCTS baselines.
