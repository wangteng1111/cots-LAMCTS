#!/usr/bin/env python3
"""COTS-LAMCTS workstation bridge v0.5.

Adds durable exactly-once job-id consumption on top of v0.4.

Design invariants
-----------------
1. A job_id is an immutable execution attempt. Once claimed, it is never
   automatically executed again. A retry must use a new job_id.
2. Claim state is persisted locally *before* compute starts and mirrored to
   workstation-results. This prevents old queue files from being replayed.
3. Terminal results are additionally snapshotted under remote/finals/ so the
   mutable live-status file is not the sole historical record.
4. Infrastructure/network failures are deferred; they are not converted into
   scientific job rejections.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cots_workstation_bridge as v3
import cots_workstation_bridge_v04 as v4

VERSION = "0.5.0"
v3.VERSION = VERSION
v4.VERSION = VERSION

RECEIPTS = v3.STATE / "receipts_v05"
RECEIPTS.mkdir(parents=True, exist_ok=True)
CLAIM_PREFIX = "remote/claims/"
FINAL_PREFIX = "remote/finals/"
TERMINAL_STATES = {"finished", "failed", "timed_out", "stopped", "rejected", "recovered"}


def _atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _receipt_path(job_id: str) -> Path:
    return RECEIPTS / f"{job_id}.json"


def _read_local_receipt(job_id: str) -> dict[str, Any] | None:
    p = _receipt_path(job_id)
    if not p.is_file():
        return None
    try:
        x = json.loads(p.read_text())
        return x if isinstance(x, dict) else None
    except Exception:
        return None


def _remote_json(path: str) -> dict[str, Any] | None:
    x = v3.gh_content(path)
    if not x or not isinstance(x, dict) or not x.get("content"):
        return None
    obj = json.loads(base64.b64decode(x["content"]).decode("utf-8"))
    return obj if isinstance(obj, dict) else None


def _request_hash(raw_obj: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(raw_obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def already_claimed(job_id: str) -> bool:
    """Return True if this immutable job id has ever been claimed.

    Local durable state is checked first. Remote claim is the cross-host/fresh
    install fallback. Network errors are intentionally propagated so the caller
    can defer processing rather than risk duplicate execution.
    """
    if _read_local_receipt(job_id) is not None:
        return True
    return _remote_json(f"{CLAIM_PREFIX}{job_id}.json") is not None


def claim(req: dict[str, Any], queue_path: str) -> dict[str, Any]:
    rec = {
        "schema": 1,
        "job_id": req["job_id"],
        "request_sha256": req["request_sha256"],
        "queue_path": queue_path,
        "code_commit": req["code_commit"],
        "entrypoint": req["entrypoint"],
        "claimed_at": time.time(),
        "hostname": os.uname().nodename,
        "bridge_version": VERSION,
        "state": "claimed",
        "policy": "job_id_once; retry_requires_new_job_id",
    }
    # Local persistence MUST precede remote publication and process start.
    _atomic_json(_receipt_path(req["job_id"]), rec)
    v4.resilient_put_file(
        f"{CLAIM_PREFIX}{req['job_id']}.json",
        (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode(),
        f"claim {req['job_id']}",
    )
    return rec


def finish_receipt(req: dict[str, Any], result: dict[str, Any]) -> None:
    job_id = req["job_id"]
    rec = _read_local_receipt(job_id) or {"schema": 1, "job_id": job_id}
    rec.update({
        "state": result.get("state", "unknown"),
        "finished_at": result.get("finished_at", time.time()),
        "returncode": result.get("returncode"),
        "request_sha256": req["request_sha256"],
        "bridge_version": VERSION,
    })
    _atomic_json(_receipt_path(job_id), rec)

    # Historical terminal snapshot. v0.5 never re-executes a claimed job id,
    # therefore this path is written once by construction.
    final_obj = dict(result)
    final_obj["bridge_version"] = VERSION
    final_obj["immutable_job_id"] = True
    final_obj["retry_policy"] = "new_job_id_required"
    v4.resilient_put_file(
        f"{FINAL_PREFIX}{job_id}.json",
        (json.dumps(final_obj, indent=2, sort_keys=True) + "\n").encode(),
        f"final snapshot {job_id}: {result.get('state')}",
    )


def run_job(req: dict[str, Any]) -> dict[str, Any]:
    result = v4.run_job(req)
    finish_receipt(req, result)
    v4.flush_pending()
    return result


def _record_invalid(job_id: str, raw_obj: dict[str, Any], queue_path: str, error: Exception) -> None:
    rec = {
        "schema": 1,
        "job_id": job_id,
        "state": "rejected",
        "request_sha256": _request_hash(raw_obj),
        "queue_path": queue_path,
        "timestamp": time.time(),
        "error": str(error),
        "hostname": os.uname().nodename,
        "bridge_version": VERSION,
        "policy": "job_id_once; retry_requires_new_job_id",
    }
    _atomic_json(_receipt_path(job_id), rec)
    v4.resilient_put_file(
        f"{CLAIM_PREFIX}{job_id}.json",
        (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode(),
        f"reject claim {job_id}",
    )
    # Keep backward-compatible live result, plus immutable terminal snapshot.
    v4.resilient_put_file(
        f"{v3.RESULT_PREFIX}{job_id}.json",
        (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode(),
        f"job {job_id}: rejected",
    )
    v4.resilient_put_file(
        f"{FINAL_PREFIX}{job_id}.json",
        (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode(),
        f"final snapshot {job_id}: rejected",
    )


def daemon(once: bool = False) -> None:
    if not v3.TOKEN:
        raise RuntimeError("COTS_GITHUB_TOKEN is not set")
    v3.ensure_repo()
    v3.ensure_results_branch()
    print(f"COTS bridge {VERSION} watching {v3.REPO_FULL}:{v3.CONTROL_BRANCH}", flush=True)
    v4.flush_pending()
    v4.recover_legacy_orphans()

    while True:
        try:
            v4.flush_pending()
            v3.put_json(v3.AGENT_STATUS_PATH, v3.agent_status(None), "workstation heartbeat")
            did_work = False
            try:
                paths = v4.queue_paths()
            except Exception as e:
                print(f"queue fetch deferred: {e}", file=sys.stderr, flush=True)
                paths = []

            for path in paths:
                obj: dict[str, Any] | None = None
                try:
                    raw = v3.git_show(f"origin/{v3.CONTROL_BRANCH}", path)
                    obj = json.loads(raw)
                    job_id = str(obj.get("job_id") or Path(path).stem)

                    # Exactly-once gate occurs before validation/fetch-heavy work.
                    # If the remote claim check itself cannot be performed, defer.
                    if already_claimed(job_id):
                        continue

                    req = v3.validate_request(obj, path)
                    claim(req, path)
                except ValueError as e:
                    job_id = str((obj or {}).get("job_id") or Path(path).stem)
                    # Only deterministic request/schema errors are rejections.
                    if _read_local_receipt(job_id) is None:
                        _record_invalid(job_id, obj or {}, path, e)
                    continue
                except Exception as e:
                    # Network/Git/infrastructure errors must never become a
                    # scientific rejection or overwrite prior history.
                    print(f"request deferred for {path}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    continue

                print(f"starting {req['job_id']} @ {req['code_commit'][:12]}", flush=True)
                try:
                    result = run_job(req)
                    print(f"finished {req['job_id']}: {result['state']} rc={result.get('returncode')}", flush=True)
                except Exception as e:
                    # Claim remains durable. Automatic replay is forbidden.
                    rec = _read_local_receipt(req["job_id"]) or {}
                    rec.update({"state": "bridge_error", "bridge_error_at": time.time(), "error": repr(e)})
                    _atomic_json(_receipt_path(req["job_id"]), rec)
                    print(f"bridge error after claim for {req['job_id']}: {e}", file=sys.stderr, flush=True)
                did_work = True
                break

            if once:
                return
            if not did_work:
                time.sleep(v3.POLL_SEC)
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"bridge loop error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            if once:
                raise
            time.sleep(v3.POLL_SEC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--recover-only", action="store_true")
    ns = ap.parse_args()
    if ns.self_test:
        print(f"bridge version: {VERSION}")
        v4.retry_fetch()
        v3.ensure_results_branch()
        v3.put_json(v3.AGENT_STATUS_PATH, v3.agent_status(None), "workstation v0.5 self-test")
        print("GitHub read/write test: OK")
        return
    if ns.recover_only:
        v4.flush_pending()
        v4.recover_legacy_orphans()
        v4.flush_pending()
        return
    daemon(once=ns.once)


if __name__ == "__main__":
    main()
