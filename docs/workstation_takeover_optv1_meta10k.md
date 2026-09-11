# COTS-LAMCTS Workstation Takeover Guide

## Purpose

This document preserves the development and execution workflow for the COTS-LAMCTS project, especially the OPTv1 Transformer-only LA-MCTS meta validation.

Target experiment:

```
optv1-transformer-only-meta-10k-20260910-1819
```

---

# 1. Development Workflow

The project uses a two-stage workflow.

## Development path

```
ChatGPT
   |
   v
Sandbox (/mnt/data)
   |
   v
Workstation repository
   |
   v
GitHub main branch
```

The sandbox is used for:

- generating new code;
- modifying experiments;
- preparing documentation;
- reviewing changes before integration.

## Execution path

```
GitHub main branch
        |
        v
workstation bridge daemon
        |
        v
aipss-Rack-Server
        |
        v
workstation-results branch
```

---

# 2. GitHub Bridge

Repository:

```
wangteng1111/cots-LAMCTS
```

Bridge implementation:

```
infra/cots_workstation_bridge.py
```

The bridge is responsible for:

- receiving experiment requests;
- running approved entrypoints;
- uploading heartbeat and results;
- exposing execution status through GitHub.

It is not the primary development path.

---

# 3. Monitoring Running Experiments

Workstation heartbeat:

```
workstation-results/remote/agent_status.json
```

Experiment result:

```
workstation-results/remote/results/<job_id>.json
```

For OPTv1 meta validation:

```
optv1-transformer-only-meta-10k-20260910-1819
```

Monitor:

- state;
- PID;
- log_tail;
- Q4096 evaluation count;
- best J;
- best state.

---

# 4. OPTv1 Meta Validation

Pipeline:

```
Optical design state
        |
        v
Transformer guidance
        |
        v
LA-MCTS search
        |
        v
Q4096 final optical evaluation
        |
        v
True objective update
```

Important principle:

The Transformer is a search guide, not the final optical evaluator.

Q4096 remains the authoritative physics-based evaluation.

---

# 5. Current Experiment Configuration

Entrypoint:

```
validation/optv1_transformer_only_lamcts_mandler_runner.py
```

Budget:

```
10000 Q4096 evaluations
```

Current validation purpose:

Verify whether Transformer-guided LA-MCTS improves exploration efficiency in a COTS optical design space.

---

# 6. Recovery Procedure

When starting a new session:

1. Read this document.
2. Check workstation heartbeat.
3. Locate job result JSON.
4. Extract progress from log_tail.
5. Continue analysis without restarting the experiment.

Expected report format:

```
Completed:
xxxx / 10000

Best J:
xxxx

Best state:
(...)

State:
RUNNING / COMPLETE / ERROR
```

---

# 7. Research Roadmap

Current phase:

```
Frozen Transformer-guided LA-MCTS validation
```

Future phase:

```
Q4096 feedback
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
