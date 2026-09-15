# Alpha Lense

## Definition
Alpha Lense is a physics-grounded, search-driven policy/value learning system for **constrained COTS optical design**. It borrows the learning loop of AlphaZero, but replaces game self-play truth with an authoritative optical physics evaluator and replaces discrete human-labelled moves with physically meaningful COTS substitutions/adjustments.

Principle: **the network proposes and generalizes; MCTS improves the policy; physics decides the realised result; the design specification defines what counts as a legal/useful lens.**

## Design specification / constraints
Alpha Lense never optimizes image-quality J in isolation. Every run is conditioned on a design brief `c` describing the required lens. At minimum this contains:
- target focal length / allowed EFL interval;
- target maximum f-number or T-stop;
- entrance pupil / clear-aperture requirements and maximum component diameter;
- sensor/film format and required image circle / field of view;
- back focal distance / flange and shutter clearance;
- total optical/mechanical length and package envelope;
- permitted number/topology of groups/elements;
- optional distortion, illumination, chief-ray-angle, telecentricity and chromatic limits;
- real-COTS constraints: availability, coating band, mount/edge/mechanical envelope and optionally price.

Constraints are split into three classes:
1. **Hard legality constraints**: impossible designs/actions are masked before MCTS expansion (e.g. component diameter cannot pass the required beam; Copal/back-space/package collision; unavailable part).
2. **Physics feasibility constraints**: quantities requiring full tracing are checked by Q4096 (EFL, F/# or T-stop, image circle/illumination, BFD, distortion, etc.). A design violating a hard specification is infeasible regardless of excellent MTF.
3. **Soft design objectives**: once feasible, image quality and optional secondary objectives determine preference. Constraint violation can provide a shaped learning signal, but must never allow a severely infeasible design to win merely because its MTF/J is good.

The optimization target is therefore lexicographic/constrained, conceptually:

`feasible(x|c) first; then minimize J_quality(x|c) and optional cost objectives.`

A practical scalar backup may use a constrained merit `J_total = J_quality + penalties`, but final ranking must retain explicit feasibility flags and physical metrics so that penalty weights cannot silently redefine the requested lens.

## State
A state `x` is a complete optical system. The neural input is the existing OPT surface sequence (up to 64 surfaces, 8 physical features per surface plus mask), not vendor/SKU IDs. The design specification `c` is also an input/conditioning vector so the same prescription can be judged differently for, e.g., 80 mm T/3 versus 50 mm f/2 requirements.

## Actions
An action is a legal optical modification under `c`. The intended production action space is physical: choose a slot/group and substitute a real COTS component whose optical signature follows a desired adjustment direction. IDs are identities only, never learned geometry.

Each legal action carries an `action_features` vector describing the physical delta between current and proposed component (curvature/power, thickness, refractive index, dispersion, topology/asphere/mechanical features as available). The catalog layer maps this physical action onto a concrete vendor+SKU. Hard design constraints mask illegal substitutions before policy normalization.

## Network
`f_theta(x,c)` shares a Transformer prescription encoder and has:
1. `merit(x,c)`: dense surrogate of immediate feasible optical merit.
2. `value(x,c)`: predicted best reachable future constrained value under search, not merely current image quality.
3. `policy(a|x,c)`: probability over legal physical adjustments/COTS substitutions.
4. `adjustment(x,c)`: latent continuous desired optical adjustment vector.
5. `feasibility(x,c)`: predicted feasibility / constraint margins (auxiliary; physics remains authoritative).
6. `embedding(x,c)`: learned system representation.

## Search-improvement loop
For state `x_t` and design brief `c`:
1. Encode full prescription and design specification.
2. Generate legal actions using cheap geometric/catalog hard constraints.
3. Network predicts priors `p(a|x,c)`, reachable value `V(x,c)`, adjustment direction and feasibility hints.
4. PUCT MCTS explores legal COTS modifications.
5. Leaf expansion uses the network cheaply.
6. High-value/high-visit candidates are evaluated by authoritative Q4096.
7. Q4096 returns image-quality metrics **and explicit design metrics/constraint margins** (EFL, F/#/T-stop, pupil/aperture, field/image circle, BFD, distortion, illumination, package-related quantities where modeled).
8. Infeasible candidates cannot outrank feasible candidates solely through image quality. Physics-grounded constrained value is backed up.
9. Root visit counts form improved policy target `pi`.
10. Continue the design trajectory; store `(x,c,pi,z_search,q_immediate,constraint_targets)` in replay.
11. Train Transformer and repeat.

## Training targets
- Policy: MCTS visit distribution `pi` after constraint masking and physics improvement.
- Search value: best physics-grounded **feasible** descendant value reachable under budget.
- Immediate merit: authoritative optical quality for evaluated states.
- Feasibility/margins: physics-computed EFL/F/#/T-stop/aperture/BFD/image-circle/etc. constraint targets.

Conceptual loss:
`L = lambda_pi CE(pi,p) + lambda_v MSE(V,z_search) + lambda_q MSE(Q,q) + lambda_c L_constraints + regularization`.

## Difference from OLMTA
OLMTA is surrogate-assisted black-box optimization. Alpha Lense is search-driven policy/value learning for a **specified constrained lens design**, with MCTS as policy-improvement operator and Q4096 as physics authority.

## Difference from AlphaZero
AlphaZero has game legality and sparse terminal outcomes. Alpha Lense has engineering design legality, continuous constraint margins, and a dense expensive physics oracle. The design brief plays a role analogous to rules/objective context and must condition both policy and value.

## Real-COTS requirement
Mandler is only an algorithm-validation environment. Production Alpha Lense uses the recovered multi-vendor catalog with vendor+SKU, prescription, diameter, coating, mechanical dimensions, availability and price metadata. A part may be optically attractive but illegal for a design because of aperture, package, shutter clearance, coating or availability.

## v0 implementation direction
The code implements Transformer + PUCT + physics grounding + replay/training. The next code revision adds an explicit `DesignSpec`, constraint evaluation/masking and spec conditioning before the first scientific meta-test.
