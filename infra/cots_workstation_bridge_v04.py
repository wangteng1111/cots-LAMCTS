#!/usr/bin/env python3
"""COTS-LAMCTS workstation bridge v0.4 resilience shim.

Wraps v0.3 and fixes the failure mode where a transient GitHub write/fetch
error could escape run_job(), orphan an already-started compute process, and
then overwrite its live result with state='rejected'.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cots_workstation_bridge as v3

VERSION = "0.4.0"
v3.VERSION = VERSION
PENDING = v3.STATE / "pending_publish"
JOURNALS = v3.STATE / "jobs_v04"
PENDING.mkdir(parents=True, exist_ok=True)
JOURNALS.mkdir(parents=True, exist_ok=True)
_ORIG_PUT_FILE = v3.put_file


def _atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _journal(job_id: str, phase: str, **extra: Any) -> None:
    _atomic_json(JOURNALS / f"{job_id}.json", {"job_id": job_id, "phase": phase, "updated_at": time.time(), **extra})


def _pending_name(path: str, data: bytes) -> Path:
    key = hashlib.sha256(path.encode() + b"\0" + data).hexdigest()[:24]
    return PENDING / f"{key}.json"


def resilient_put_file(path: str, data: bytes, message: str) -> None:
    try:
        _ORIG_PUT_FILE(path, data, message)
        return
    except Exception as e:
        rec = {"path": path, "message": message, "data_b64": base64.b64encode(data).decode("ascii"),
               "queued_at": time.time(), "last_error": repr(e)}
        _atomic_json(_pending_name(path, data), rec)
        print(f"publish queued locally: {path}: {e}", file=sys.stderr, flush=True)
        return


v3.put_file = resilient_put_file


def flush_pending() -> tuple[int, int]:
    ok = 0
    failed = 0
    for p in sorted(PENDING.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
            _ORIG_PUT_FILE(rec["path"], base64.b64decode(rec["data_b64"]), rec["message"])
            p.unlink()
            ok += 1
        except Exception as e:
            failed += 1
            try:
                rec["last_error"] = repr(e)
                rec["last_attempt_at"] = time.time()
                _atomic_json(p, rec)
            except Exception:
                pass
    return ok, failed


def retry_fetch() -> None:
    last: Exception | None = None
    for i in range(5):
        try:
            v3.fetch()
            return
        except Exception as e:
            last = e
            time.sleep(min(30, 2 ** i))
    raise RuntimeError(f"git fetch still unavailable after retries: {last}")


def queue_paths() -> list[str]:
    retry_fetch()
    out = v3.run(["git", "ls-tree", "-r", "--name-only", f"origin/{v3.CONTROL_BRANCH}", v3.QUEUE_PREFIX.rstrip("/")],
                 cwd=v3.MIRROR, timeout=120)
    return sorted(p for p in out.splitlines() if p.startswith(v3.QUEUE_PREFIX) and p.endswith(".json"))


def run_job(req: dict[str, Any]) -> dict[str, Any]:
    job_id = req["job_id"]
    _journal(job_id, "starting", request=req)
    try:
        result = v3.run_job(req)
    except Exception as e:
        _journal(job_id, "bridge_error", error=repr(e))
        raise
    _journal(job_id, "terminal_local", status=result)
    flush_pending()
    return result


def _proc_matches_job(job: Path) -> bool:
    needle = str(job / "src")
    proc_root = Path("/proc")
    if not proc_root.exists():
        return False
    for p in proc_root.iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().decode("utf-8", errors="ignore")
            if needle in cmd:
                return True
        except Exception:
            pass
    return False


ROUND_RE = re.compile(r"ROUND\s+(\d+)\s+n\s+(\d+)\s+bestJ\s+([^\s]+)\s+best\s+(\([^\n]+?\))\s+eligible\s+(\d+)")


def _last_round(log: Path) -> dict[str, Any] | None:
    text = v3.tail(log, 5000, 5_000_000)
    matches = list(ROUND_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return {"round": int(m.group(1)), "n": int(m.group(2)), "bestJ": float(m.group(3)),
            "best_state": m.group(4), "eligible": int(m.group(5))}


def recover_legacy_orphans() -> None:
    if not v3.JOBS.exists():
        return
    for job in sorted(v3.JOBS.iterdir()):
        log = job / "job.log"
        if not job.is_dir() or not log.is_file() or _proc_matches_job(job):
            continue
        job_id = job.name
        remote = None
        try:
            x = v3.gh_content(f"{v3.RESULT_PREFIX}{job_id}.json")
            if x and isinstance(x, dict) and x.get("content"):
                remote = json.loads(base64.b64decode(x["content"]).decode())
        except Exception:
            pass
        if remote and remote.get("state") in {"finished", "failed", "timed_out", "stopped", "recovered"}:
            continue
        last = _last_round(log)
        status = {"schema": 1, "job_id": job_id, "state": "recovered", "hostname": os.uname().nodename,
                  "recovered_at": time.time(), "recovery_source": "legacy_local_job_dir",
                  "last_complete_round": last, "log_tail": v3.tail(log, 300),
                  "note": "Legacy compute process ended outside bridge supervision; original exit code is unavailable."}
        resilient_put_file(f"{v3.ARTIFACT_PREFIX}{job_id}/job.log", log.read_bytes()[-v3.MAX_LOG_UPLOAD_BYTES:],
                           f"job {job_id}: recovered log")
        v3.put_json(f"{v3.RESULT_PREFIX}{job_id}.json", status, f"job {job_id}: recovered")
        _journal(job_id, "recovered", status=status)


def daemon(once: bool = False) -> None:
    if not v3.TOKEN:
        raise RuntimeError("COTS_GITHUB_TOKEN is not set")
    v3.ensure_repo()
    v3.ensure_results_branch()
    print(f"COTS bridge {VERSION} watching {v3.REPO_FULL}:{v3.CONTROL_BRANCH}", flush=True)
    flush_pending()
    recover_legacy_orphans()
    while True:
        try:
            flush_pending()
            v3.put_json(v3.AGENT_STATUS_PATH, v3.agent_status(None), "workstation heartbeat")
            did_work = False
            try:
                paths = queue_paths()
            except Exception as e:
                print(f"queue fetch deferred: {e}", file=sys.stderr, flush=True)
                paths = []
            for path in paths:
                obj: dict[str, Any] | None = None
                try:
                    raw = v3.git_show(f"origin/{v3.CONTROL_BRANCH}", path)
                    obj = json.loads(raw)
                    job_id = str(obj.get("job_id") or Path(path).stem)
                    if v3.final_result_exists(job_id):
                        continue
                    req = v3.validate_request(obj, path)
                except ValueError as e:
                    job_id = str((obj or {}).get("job_id") or Path(path).stem)
                    v3.reject(job_id, e, path)
                    continue
                except Exception as e:
                    print(f"request validation deferred for {path}: {e}", file=sys.stderr, flush=True)
                    continue
                print(f"starting {req['job_id']} @ {req['code_commit'][:12]}", flush=True)
                try:
                    result = run_job(req)
                    print(f"finished {req['job_id']}: {result['state']} rc={result.get('returncode')}", flush=True)
                except Exception as e:
                    print(f"bridge error after start for {req['job_id']}: {e}", file=sys.stderr, flush=True)
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
        retry_fetch()
        v3.ensure_results_branch()
        v3.put_json(v3.AGENT_STATUS_PATH, v3.agent_status(None), "workstation v0.4 self-test")
        print("GitHub read/write test: OK")
        return
    if ns.recover_only:
        flush_pending()
        recover_legacy_orphans()
        flush_pending()
        return
    daemon(once=ns.once)


if __name__ == "__main__":
    main()
