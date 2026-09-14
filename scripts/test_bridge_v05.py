#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
state_root = Path(tempfile.mkdtemp(prefix="bridge-v05-test-", dir=os.environ.get("COTS_JOB_DIR", "/tmp")))
os.environ["COTS_BRIDGE_ROOT"] = str(state_root)
sys.path.insert(0, str(repo / "infra"))

import cots_workstation_bridge_v05 as v5  # noqa: E402

job_id = "selftest-v05-immutable-job"
req = {
    "schema": 1,
    "job_id": job_id,
    "request_sha256": "a" * 64,
    "code_commit": "b" * 40,
    "entrypoint": "scripts/noop.py",
}

# Test durable local claim gate without touching remote GitHub.
rec = {
    "schema": 1,
    "job_id": job_id,
    "state": "claimed",
    "request_sha256": req["request_sha256"],
}
v5._atomic_json(v5._receipt_path(job_id), rec)
assert v5._read_local_receipt(job_id)["state"] == "claimed"
assert v5.already_claimed(job_id) is True

out = {
    "bridge_version": v5.VERSION,
    "root": str(state_root),
    "local_claim_persisted": True,
    "duplicate_gate": True,
    "terminal_states_include_recovered": "recovered" in v5.TERMINAL_STATES,
}
art = Path(os.environ.get("COTS_JOB_ARTIFACT_DIR", repo / "artifacts"))
art.mkdir(parents=True, exist_ok=True)
(art / "bridge_v05_smoke.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, sort_keys=True))
