# GitHub Bridge Workstation Access Guide

## Purpose

This document describes how to access, monitor, and control the COTS-LAMCTS workstation through the GitHub relay bridge.

The workstation is controlled through GitHub as the communication plane rather than direct SSH access.

```
Controller
    |
    v
GitHub repository
wangteng1111/cots-LAMCTS
    |
    +----------------+
    |                |
    v                v
main branch     workstation-results branch
(queue/control) (status/results)
    |
    v
cots_workstation_bridge.py
    |
    v
Compute workstation
```

## Repository

Repository:

```
wangteng1111/cots-LAMCTS
```

Branches:

```
main
workstation-results
```

Important paths:

```
remote/
 ├── queue/              # job requests
 ├── control/            # stop/control requests
 ├── results/            # execution results
 ├── artifacts/          # uploaded artifacts
 └── agent_status.json   # workstation heartbeat

infra/
 ├── cots_workstation_bridge.py
 └── cots-lamcts-bridge.service
```

## Bridge Mechanism

The workstation daemon:

1. Polls GitHub main branch.
2. Reads `remote/queue/*.json`.
3. Validates the requested job.
4. Checks out the requested commit.
5. Runs the allowed entrypoint.
6. Uploads status and results to `workstation-results`.

No inbound network listener is required.

## Workstation Service

Systemd service:

```
User: cotslamcts
WorkingDirectory: /opt/cots-lamcts
EnvironmentFile: /etc/cots-lamcts/bridge.env
ExecStart: /usr/bin/python3 /opt/cots-lamcts/infra/cots_workstation_bridge.py
```

The service uses automatic restart.

## Reading Workstation Status

Heartbeat file:

```
workstation-results/remote/agent_status.json
```

Contains:

- hostname
- CPU information
- GPU utilization
- running job
- bridge version
- timestamp

Example:

```json
{
  "hostname": "aipss-Rack-Server",
  "running_job": "optv1-transformer-only-meta-10k-20260910-1819",
  "gpu": "RTX 5090"
}
```

## Monitoring a Job

For a job:

```
remote/results/<job_id>.json
```

Check:

- execution state
- completion
- errors
- evaluation count
- best score
- best state

## Job Submission

Create:

```
main/remote/queue/<job_id>.json
```

Example schema:

```json
{
  "schema": 1,
  "job_id": "example",
  "code_commit": "git_sha",
  "entrypoint": "search/run_large_meta.py",
  "args": [],
  "timeout_sec": 172800
}
```

The bridge automatically executes the job and writes results.

## Stop a Job

Create:

```
remote/control/<job_id>.json
```

with:

```json
{
  "action": "stop"
}
```

## Current Workstation

Known workstation:

```
aipss-Rack-Server
```

Hardware:

```
CPU: 128 cores
GPU: 2 x NVIDIA RTX 5090
```

## Debugging Procedure

1. Check:

```
workstation-results/remote/agent_status.json
```

Confirm heartbeat updates.

2. Check:

```
workstation-results/remote/results/
```

Confirm result generation.

3. Check:

```
main/remote/queue/
```

Confirm job request exists.

## Current Experiment

```
optv1-transformer-only-meta-10k-20260910-1819
```

Monitoring targets:

- Q4096 evaluation count
- best J
- best state
- errors
- completion

## Summary

The access model is:

```
GitHub = control plane

main branch
    |
    + queue
    + control

workstation daemon
    |
    v
compute node
    |
    v
workstation-results branch
    |
    + heartbeat
    + results
    + artifacts
```
