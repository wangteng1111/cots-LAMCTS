# Alpha Lense v0.1 — algorithm and pretraining definition

## 1. Definition
Alpha Lense v0.1 is a constraint-conditioned optical-design policy/value system trained in three stages:

1. **Learn optics**: pretrain the Transformer prescription representation on optical structure and physics.
2. **Learn to design**: build stochastic degradation trees around known-good commercial lenses, evaluate local neighborhoods with the authoritative physics evaluator, and reverse improving edges into policy supervision.
3. **Learn beyond known lenses**: use the pretrained policy/value network inside physics-grounded MCTS; MCTS visit distributions and grounded descendant values become new training targets.

The central rule is: **network proposes; MCTS improves; physics decides; DesignSpec defines the lens.** Commercial lenses are seeds of known-good basins, not immutable ground truth.

## 2. State and design conditioning
A state `x` is a complete optical prescription represented by physical surface tokens, not arbitrary part IDs. A design brief `c=DesignSpec` contains focal-length tolerance, F/# or T-stop, aperture/diameter, image circle/field, BFD, package limits, distortion/illumination and real-COTS constraints where available.

The network is conditioned on `(x,c)`. The same prescription can therefore have different value and policy under different lens requirements.

## 3. Physical perturbation actions
Pretraining perturbations act on physically meaningful variables. v0.1 supports a generic `PerturbationAction` containing one or more deltas such as:

- surface curvature / radius;
- inter-surface or inter-group axial distance;
- element thickness;
- refractive index;
- Abbe/dispersion parameter;
- asphere coefficients;
- aperture-related variables where represented;
- discrete glass or COTS substitution through an adapter.

A perturbation step may change several parameters. The perturbation sampler must enforce cheap legality bounds before expensive tracing.

## 4. Stochastic degradation tree
For each known-good seed lens `x0`, generate stochastic trajectories rather than materializing the impossible full `100^10` tree.

Conceptually:

`x0 -> x1 -> ... -> xD`, with nominal depth `D=10` and local branching/proposal count `K≈100`.

In implementation, maintain a bounded frontier or sample trajectories. Every visited parent receives a local neighborhood of perturbation proposals. This yields arbitrarily large training corpora without exponential storage.

Generation ancestry alone is **not** a quality label. A random perturbation can improve a commercial design.

## 5. Physics relabeling and reverse improvement policy
For a parent state `x`, evaluate the parent and selected legal neighbors `xi` using the authoritative evaluator (Q4096 in the current environment). Apply `DesignSpec` constraints to every result.

Preference is feasibility-first:

1. feasible beats infeasible;
2. among feasible states, lower authoritative `J_quality` is better;
3. among infeasible states, lower **normalized dimensionless constraint violation** is better.

Define an improvement utility `u(xi|x,c)` from this ordered score. Only physics-supported improving moves receive positive policy mass. The policy-pretraining target is a soft distribution:

`pi_pre(i|x,c) ∝ exp(u_i / tau)` over improving legal actions.

If no neighbor improves, the sample may be used as a value/physics/feasibility example but is not forced into a false one-hot policy label.

Thus tree edges are reversed by **physics quality**, not blindly by ancestry. A child that physics shows to be better than its parent can become the target direction even if it lies farther from the original commercial prescription.

## 6. Network
`f_theta(x,c)` uses a shared Transformer optical encoder and predicts:

- `policy(a|x,c)`: physical improvement action distribution;
- `value(x,c)`: best reachable constrained value under search;
- `merit(x,c)`: immediate grounded optical merit surrogate;
- `adjustment(x,c)`: continuous desired physical adjustment direction;
- `feasibility(x,c)` and normalized constraint margins;
- optional physics-property heads during optical pretraining.

## 7. Stage I — optical representation pretraining
The backbone is not random when policy pretraining begins. Pretraining should combine:

`L_opt = λ_mask L_masked-prescription + λ_contrast L_contrastive + λ_phys L_physics`.

The first two inherit the OPTv1 idea. Physics heads can predict EFL, F/#, BFD, distortion, illumination, chromatic and image-quality descriptors when labels exist. This stage teaches the encoder optical structure rather than search strategy.

## 8. Stage II — reverse-tree policy pretraining
For each physics-relabelled neighborhood store:

`(x, c, actions, pi_pre, q_immediate, feasibility, normalized_margins, best_neighbor_value)`.

Train policy jointly with grounded auxiliary heads. Policy is therefore meaningful before the first MCTS run.

Recommended loss:

`L_pre = λ_pi CE(pi_pre,p) + λ_q L_merit + λ_f BCE(feasible) + λ_m L_margin + λ_adj L_adjust + λ_v L_bootstrap-value`.

`L_v` must use a physics-grounded reachable target from the sampled neighborhood/trajectory, not pretend that the commercial seed is always globally optimal.

## 9. Stage III — MCTS self-improvement
Initialize Alpha Lense MCTS from the Stage-II checkpoint. During search, selected leaves are evaluated by Q4096 within a physics budget and grounded values are backed up through PUCT. Root visits produce:

`pi_MCTS(a|x,c) = N(x,a)^(1/tau) / sum_b N(x,b)^(1/tau)`.

Replay stores MCTS-improved policy and grounded descendant value. Continued training transitions supervision from `pi_pre` to `pi_MCTS`.

This gives the curriculum:

`optical knowledge -> local design skill around known-good basins -> search-driven improvement beyond existing designs`.

## 10. Required safeguards
- Never instantiate the literal `100^10` tree; use bounded stochastic frontier/trajectory sampling.
- Never infer improvement solely from parameter distance to a commercial lens.
- Never allow image quality to override hard design infeasibility.
- Constraint violation used for learning/ranking must be dimensionless and normalized by a physically meaningful scale/tolerance.
- Required physics metrics missing from an evaluation must be marked unknown/invalid rather than silently treated as feasible.
- Preserve evaluator configuration hashes and seed/checkpoint manifests for reproducibility.
- Separate arbitrary database identity from learned optical features.

## 11. v0.1 validation plan
A scientific v0.1 experiment should report, in order:

1. optical-pretraining validation;
2. reverse-tree policy top-k accuracy / cross-entropy on held-out commercial-lens basins;
3. probability assigned to physics-improving actions versus random initialization/OPT-only baseline;
4. MCTS physics-evaluation efficiency: feasible rate, first-feasible Q4096 count, best feasible J versus Q4096 calls;
5. held-out seed-lens generalization;
6. later, real multi-vendor COTS design under a true T3/80 DesignSpec.

The earlier random-network smoke test is only an integration test and is not evidence of Alpha Lense design performance.
