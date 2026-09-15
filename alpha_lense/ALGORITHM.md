# Alpha Lense

## Definition
Alpha Lense is a physics-grounded, search-driven policy/value learning system for COTS optical design. It borrows the learning loop of AlphaZero, but replaces game self-play truth with an authoritative optical physics evaluator and replaces discrete human-labelled moves with physically meaningful COTS substitutions/adjustments.

Principle: **the network proposes and generalizes; MCTS improves the policy; physics decides the realised result.**

## State
A state `x` is a complete optical system. The neural input is the existing OPT surface sequence (up to 64 surfaces, 8 physical features per surface plus mask), not vendor/SKU IDs.

## Actions
An action is a legal optical modification. The intended production action space is physical: choose a slot/group and substitute a real COTS component whose optical signature follows a desired adjustment direction. IDs are identities only, never learned geometry.

Each legal action carries an `action_features` vector describing the physical delta between current and proposed component (curvature/power, thickness, refractive index, dispersion, topology/asphere/mechanical features as available). The catalog layer maps this physical action onto a concrete vendor+SKU.

## Network
`f_theta(x)` shares a Transformer prescription encoder and has:

1. `merit(x)`: dense surrogate of immediate `-log(J_physics(x))`.
2. `value(x)`: predicted best reachable future value under search, not merely current merit.
3. `policy(a|x)`: probability over legal physical adjustments/COTS substitutions.
4. `adjustment(x)`: latent continuous desired optical adjustment vector; used as an auxiliary representation and later for catalog projection.
5. `embedding(x)`: learned system representation.

## Search-improvement loop
For state `x_t`:

1. Encode full prescription.
2. Network predicts priors `p(a|x)` and reachable value `V(x)`.
3. PUCT MCTS explores legal COTS modifications.
4. Leaf expansion uses network value cheaply.
5. High-value/high-visit candidates are evaluated by the authoritative physics evaluator (currently production Q4096).
6. Physics result is transformed as `z=-log(J)` and backed up into the tree.
7. Root visit counts produce improved policy target
   `pi(a|x) = N(x,a)^(1/tau) / sum_b N(x,b)^(1/tau)`.
8. Select a root action and continue the design trajectory.
9. Store `(x, pi, z_search, q_immediate)` in replay.
10. Train Transformer from replay and repeat.

This makes MCTS a policy-improvement/training operator, rather than merely a consumer of the Transformer estimator.

## Training targets
- Policy target: MCTS visit distribution `pi`.
- Search-value target: best physics-grounded descendant value reachable from the state under the search budget.
- Immediate-merit target: dense authoritative `q=-log(J_physics(x))` whenever the state has been evaluated.

Loss (v0):

`L = lambda_pi * CE(pi,p_theta) + lambda_v * MSE(V_theta,z_search) + lambda_q * MSE(Q_theta,q_immediate) + regularization`

Later auxiliary losses can train the continuous adjustment head from successful local transitions.

## Difference from current OLMTA
OLMTA: `OPT -> rank/partition/propose -> Q4096 -> triggered estimator update`.

Alpha Lense: `Transformer policy/value -> MCTS -> Q4096-grounded tree improvement -> visit/value targets -> Transformer training -> stronger MCTS`.

The conceptual change is therefore from **surrogate-assisted black-box optimization** to **search-driven policy/value learning with a physics oracle**.

## Difference from AlphaZero
AlphaZero receives sparse terminal game outcome. Alpha Lense has a dense expensive oracle: Q4096 can evaluate realised optical systems directly. Therefore Alpha Lense should not copy AlphaZero literally. It combines dense immediate physics supervision with MCTS-derived reachable-value and policy supervision.

## Real-COTS requirement
The Mandler 72-state benchmark can be used only for algorithm validation. Production Alpha Lense must use the recovered multi-vendor COTS catalog with vendor+SKU identity and physical prescription/geometry. Procurement metadata (price, stock, coating, aperture, mounting envelope) should enter legality/cost constraints, not replace physical optical representation.

## v0 code
`alpha_lense/alpha_lense_v0.py` implements:
- Transformer shared encoder;
- immediate-merit/value/policy/adjustment heads;
- PUCT MCTS;
- selective authoritative physics grounding;
- visit-count policy targets;
- replay buffer;
- AlphaZero-like multi-head training loss.

## Next integration steps
1. Adapter for existing OPTv1 surface tokenizer/checkpoint.
2. Adapter for production Q4096 evaluator.
3. Catalog action layer for current Mandler benchmark as a controlled meta-test.
4. Reproducibility test against known OLMTA benchmark states.
5. Replace benchmark adapter with real multi-vendor COTS catalog.
6. Add trajectory-level best-descendant value backup and continuous adjustment supervision.
7. Compare physics evaluations-to-best-J against OLMTA, random/LA-MCTS, and Transformer-only baselines.
