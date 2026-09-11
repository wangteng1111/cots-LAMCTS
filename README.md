# COTS-LAMCTS

## Commercial-Off-The-Shelf Optical Design with Learning-Augmented Monte Carlo Tree Search

COTS-LAMCTS is a research framework for autonomous optical design by combining:

- Commercial-Off-The-Shelf (COTS) optical components;
- Large-scale black-box optimization;
- Learning-guided Monte Carlo Tree Search;
- Physics-based optical evaluation.

The goal is to transform optical design from a manually guided iterative process into an autonomous search problem over manufacturable optical design spaces.

---

# Research Idea

Modern optical design requires expert knowledge to repeatedly:

1. select candidate optical elements;
2. adjust configuration parameters;
3. evaluate optical performance;
4. refine the design.

However, optical search spaces are difficult because they are:

- highly non-convex;
- expensive to evaluate;
- constrained by available commercial components.

COTS-LAMCTS treats optical design as a structured search problem and combines learning with Monte Carlo Tree Search to efficiently explore this space.

---

# Core Methodology

## Transformer-Guided LA-MCTS

The current OPTv1 framework follows:

```
Optical design state
        |
        v
Transformer representation
        |
        v
Search guidance
(reward estimation + uncertainty)
        |
        v
LA-MCTS exploration
        |
        v
Q4096 physical evaluation
        |
        v
True objective update
```

A key principle:

> The Transformer guides search; it does not replace the physical evaluator.

All final optical decisions are validated by Q4096.

---

# Current Validation

## OPTv1 Transformer-only Meta Validation

Experiment:

```
optv1-transformer-only-meta-10k-20260910-1819
```

Configuration:

```
Q4096 budget: 10000 evaluations
Initial population: 96
Batch size: 16
Pool size: 60000
Transformer ensemble: 3
Workers: 32
```

The purpose is to validate whether Transformer-guided LA-MCTS can improve exploration efficiency while preserving physics-based evaluation.

---

# Execution Infrastructure

The project uses a GitHub relay workstation architecture.

```
Development

ChatGPT
   |
   v
Sandbox
   |
   v
Workstation repository
   |
   v
GitHub main


Execution

GitHub
   |
   v
Workstation bridge
   |
   v
Compute node
   |
   v
workstation-results
```

The bridge provides:

- experiment execution;
- heartbeat monitoring;
- result synchronization.

Development and execution are intentionally separated.

Detailed recovery guide:

```
docs/workstation_takeover_optv1_meta10k.md
```

---

# Repository Structure

```
cots-LAMCTS/

├── core/
├── validation/
├── evaluator/
├── infra/
│   ├── cots_workstation_bridge.py
│   └── cots-lamcts-bridge.service
├── remote/
│   ├── queue/
│   ├── control/
│   └── results/
└── docs/
```

---

# Research Roadmap

## Phase 1 — Transformer-Guided Search Validation

Demonstrate that learned search guidance improves LA-MCTS exploration.

## Phase 2 — Adaptive Learning

Future extension:

```
Q4096 evaluation
        |
        v
Dataset accumulation
        |
        v
Transformer update
        |
        v
Adaptive LA-MCTS
```

## Phase 3 — Autonomous COTS Optical Design

Target:

- large optical component catalogs;
- physical optical representations;
- manufacturable lens systems;
- autonomous optical discovery.

---

# Key Research Questions

1. Can learning-guided search outperform classical optimization heuristics?

2. Can a Transformer learn optical design representations rather than catalog identities?

3. Can autonomous search discover competitive optical systems under real manufacturing constraints?
