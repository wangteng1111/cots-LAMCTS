#!/usr/bin/env python3
"""GitHub relay bridge for the COTS-LAMCTS workstation.

This daemon turns a dedicated workstation into a compute node controlled through
one fixed GitHub repository:

    https://github.com/wangteng1111/cots-LAMCTS

Why GitHub relay instead of a remote shell/MCP endpoint?
---------------------------------------------------------
The controller writes immutable job-request JSON files to `main`.  This daemon
polls those files, checks out the requested exact code commit into an isolated
worktree, runs only an allow-listed Python entrypoint, and writes status/results
to a separate `workstation-results` branch through the GitHub Contents API.

There is no inbound network listener and no general shell command interface.
The workstation therefore does not need to be publicly exposed.

Required on the workstation
---------------------------
* Linux / POSIX
* Python 3.10+
* git
* A fine-grained GitHub token with **Contents: read/write** access ONLY to
  `wangteng1111/cots-LAMCTS`, provided in environment variable
  `COTS_GITHUB_TOKEN`. Never put the token in this file or in Git.

Usage
-----
    export COTS_GITHUB_TOKEN='...'
    python cots_workstation_bridge.py --self-test
    python cots_workstation_bridge.py

Optional environment variables
------------------------------
    COTS_BRIDGE_ROOT=~/.cots_lamcts_bridge
    COTS_BRIDGE_POLL_SEC=20
    COTS_BRIDGE_HEARTBEAT_SEC=180
    COTS_BRIDGE_PYTHON=/path/to/venv/bin/python
    COTS_BRIDGE_DEFAULT_MEMORY_GB=0       # 0 = no RLIMIT_AS cap
    COTS_BRIDGE_MAX_MEMORY_GB=256

Job request schema (committed by controller under remote/queue/<job_id>.json)
-----------------------------------------------------------------------------
{
  "schema": 1,
  "job_id": "meta-20260909-001",
  "code_commit": "<40 hex sha reachable from main>",
  "entrypoint": "search/run_large_meta.py",
  "args": ["--config", "configs/meta.json"],
  "env": {"NUMBA_NUM_THREADS": "1"},
  "timeout_sec": 172800,
  "memory_gb": 0
}

To stop a running job, commit remote/control/<job_id>.json to main:
    {"action": "stop"}
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


VERSION = "0.3.0"
OWNER = "wangteng1111"
REPO = "cots-LAMCTS"
REPO_FULL = f"{OWNER}/{REPO}"
REPO_URL = f"https://github.com/{REPO_FULL}.git"
CONTROL_BRANCH = "main"
RESULTS_BRANCH = "workstation-results"
QUEUE_PREFIX = "remote/queue/"
CONTROL_PREFIX = "remote/control/"
RESULT_PREFIX = "remote/results/"
ARTIFACT_PREFIX = "remote/artifacts/"
AGENT_STATUS_PATH = "remote/agent_status.json"

ROOT = Path(os.environ.get("COTS_BRIDGE_ROOT", "~/.cots_lamcts_bridge")).expanduser().resolve()
MIRROR = ROOT / "repo"
JOBS = ROOT / "jobs"
STATE = ROOT / "state"
TOKEN = os.environ.get("COTS_GITHUB_TOKEN", "")
PYTHON = os.environ.get("COTS_BRIDGE_PYTHON", sys.executable)
POLL_SEC = max(5, int(os.environ.get("COTS_BRIDGE_POLL_SEC", "20")))
HEARTBEAT_SEC = max(30, int(os.environ.get("COTS_BRIDGE_HEARTBEAT_SEC", "180")))
DEFAULT_TIMEOUT_SEC = int(os.environ.get("COTS_BRIDGE_DEFAULT_TIMEOUT_SEC", str(48 * 3600)))
MAX_TIMEOUT_SEC = int(os.environ.get("COTS_BRIDGE_MAX_TIMEOUT_SEC", str(48 * 3600)))
DEFAULT_MEMORY_GB = int(os.environ.get("COTS_BRIDGE_DEFAULT_MEMORY_GB", "0"))
MAX_MEMORY_GB = int(os.environ.get("COTS_BRIDGE_MAX_MEMORY_GB", "256"))

ALLOWED_ENTRY_PREFIXES = tuple(
    x.strip().strip("/") + "/"
    for x in os.environ.get(
        "COTS_BRIDGE_ENTRY_PREFIXES", "search,validation,scripts,benchmarks"
    ).split(",")
    if x.strip()
)
ALLOWED_JOB_ENV = {
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
}
TEXT_ARTIFACT_SUFFIXES = {
    ".json", ".jsonl", ".csv", ".txt", ".md", ".log", ".yaml", ".yml"
}
MAX_ARTIFACT_BYTES = 20 * 1024 * 1024
MAX_LOG_UPLOAD_BYTES = 5 * 1024 * 1024

for p in (ROOT, JOBS, STATE):
    p.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 600, check: bool = True) -> str:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        shell=False,
    )
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd!r}\n{p.stdout[-8000:]}")
    return p.stdout.strip()


def ensure_repo() -> None:
    if (MIRROR / ".git").exists():
        return
    if MIRROR.exists():
        shutil.rmtree(MIRROR)
    run(["git", "clone", "--origin", "origin", REPO_URL, str(MIRROR)], timeout=900)


def fetch() -> None:
    ensure_repo()
    run(["git", "fetch", "--prune", "--tags", "origin"], cwd=MIRROR, timeout=900)


def git_show(ref: str, path: str) -> str:
    return run(["git", "show", f"{ref}:{path}"], cwd=MIRROR, timeout=120)


def queue_paths() -> list[str]:
    fetch()
    out = run(
        ["git", "ls-tree", "-r", "--name-only", f"origin/{CONTROL_BRANCH}", QUEUE_PREFIX.rstrip("/")],
        cwd=MIRROR,
        timeout=120,
    )
    return sorted(p for p in out.splitlines() if p.startswith(QUEUE_PREFIX) and p.endswith(".json"))


def exact_commit(sha: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", sha or ""):
        raise ValueError("code_commit must be an exact 40-hex SHA")
    fetch()
    resolved = run(["git", "rev-parse", "--verify", f"{sha}^{{commit}}"], cwd=MIRROR)
    if resolved != sha:
        raise ValueError("code_commit did not resolve exactly")
    p = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, f"origin/{CONTROL_BRANCH}"],
        cwd=str(MIRROR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
    )
    if p.returncode != 0:
        raise ValueError("code_commit is not reachable from origin/main")
    return sha


def gh_request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    if not TOKEN:
        raise RuntimeError("COTS_GITHUB_TOKEN is not set")
    url = f"https://api.github.com/repos/{REPO_FULL}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "cots-lamcts-workstation-bridge")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw.decode("utf-8")) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        if e.code == 404:
            return None
        raise RuntimeError(f"GitHub API {method} {path} failed: HTTP {e.code}: {raw[:2000]}") from e


def ensure_results_branch() -> None:
    ref = gh_request("GET", f"/git/ref/heads/{urllib.parse.quote(RESULTS_BRANCH, safe='')}")
    if ref is not None:
        return
    main = gh_request("GET", f"/git/ref/heads/{CONTROL_BRANCH}")
    if not main:
        raise RuntimeError("main branch does not exist yet")
    sha = main["object"]["sha"]
    gh_request("POST", "/git/refs", {"ref": f"refs/heads/{RESULTS_BRANCH}", "sha": sha})


def gh_content(path: str, branch: str = RESULTS_BRANCH) -> Any:
    q = urllib.parse.urlencode({"ref": branch})
    return gh_request("GET", f"/contents/{urllib.parse.quote(path, safe='/')}?{q}")


def put_file(path: str, data: bytes, message: str) -> None:
    ensure_results_branch()
    old = gh_content(path)
    body: dict[str, Any] = {
        "message": message[:200],
        "content": base64.b64encode(data).decode("ascii"),
        "branch": RESULTS_BRANCH,
    }
    if old and isinstance(old, dict) and old.get("sha"):
        body["sha"] = old["sha"]
    gh_request("PUT", f"/contents/{urllib.parse.quote(path, safe='/')}", body)


def put_json(path: str, obj: Any, message: str) -> None:
    put_file(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8"), message)


def final_result_exists(job_id: str) -> bool:
    x = gh_content(f"{RESULT_PREFIX}{job_id}.json")
    if not x or not isinstance(x, dict) or "content" not in x:
        return False
    try:
        obj = json.loads(base64.b64decode(x["content"]).decode("utf-8"))
        return obj.get("state") in {"finished", "failed", "timed_out", "stopped", "rejected"}
    except Exception:
        return False


def validate_request(obj: dict[str, Any], queue_path: str) -> dict[str, Any]:
    if obj.get("schema") != 1:
        raise ValueError("unsupported schema")
    job_id = str(obj.get("job_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9._-]{6,80}", job_id):
        raise ValueError("invalid job_id")
    if queue_path != f"{QUEUE_PREFIX}{job_id}.json":
        raise ValueError("queue filename must match job_id")

    commit = exact_commit(str(obj.get("code_commit", "")))
    entry = str(obj.get("entrypoint", "")).replace("\\", "/").lstrip("/")
    if not entry.endswith(".py") or ".." in Path(entry).parts:
        raise ValueError("invalid Python entrypoint")
    if not any(entry.startswith(p) for p in ALLOWED_ENTRY_PREFIXES):
        raise ValueError(f"entrypoint must be under {ALLOWED_ENTRY_PREFIXES}")

    args = obj.get("args") or []
    if not isinstance(args, list) or len(args) > 128:
        raise ValueError("invalid args")
    args = [str(x) for x in args]
    if any(len(x) > 4096 or "\x00" in x for x in args):
        raise ValueError("invalid argv item")

    env = obj.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("invalid env")
    env2: dict[str, str] = {}
    for k, v in env.items():
        if k not in ALLOWED_JOB_ENV:
            raise ValueError(f"env variable not allowed: {k}")
        s = str(v)
        if len(s) > 256 or "\x00" in s:
            raise ValueError(f"invalid env value: {k}")
        env2[k] = s

    timeout = int(obj.get("timeout_sec", DEFAULT_TIMEOUT_SEC))
    if timeout < 60 or timeout > MAX_TIMEOUT_SEC:
        raise ValueError(f"timeout_sec must be in [60,{MAX_TIMEOUT_SEC}]")
    memory = int(obj.get("memory_gb", DEFAULT_MEMORY_GB))
    if memory < 0 or memory > MAX_MEMORY_GB:
        raise ValueError(f"memory_gb must be in [0,{MAX_MEMORY_GB}]")

    return {
        "schema": 1,
        "job_id": job_id,
        "code_commit": commit,
        "entrypoint": entry,
        "args": args,
        "env": env2,
        "timeout_sec": timeout,
        "memory_gb": memory,
        "request_sha256": hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest(),
    }


def preexec(memory_gb: int) -> None:
    os.setsid()
    if memory_gb > 0:
        lim = memory_gb * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))


def tail(path: Path, lines: int = 100, max_bytes: int = 1_000_000) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        text = f.read().decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def stop_requested(job_id: str) -> bool:
    fetch()
    path = f"{CONTROL_PREFIX}{job_id}.json"
    p = subprocess.run(
        ["git", "cat-file", "-e", f"origin/{CONTROL_BRANCH}:{path}"],
        cwd=str(MIRROR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False
    )
    if p.returncode != 0:
        return False
    try:
        obj = json.loads(git_show(f"origin/{CONTROL_BRANCH}", path))
        return obj.get("action") == "stop"
    except Exception:
        return False


def gpu_info() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    out = run([
        "nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits"
    ], timeout=20, check=False)
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 5:
            rows.append({"name": p[0], "memory_total_mb": p[1], "memory_free_mb": p[2],
                         "utilization_pct": p[3], "temperature_c": p[4]})
    return rows


def agent_status(running: str | None = None) -> dict[str, Any]:
    return {
        "schema": 1,
        "bridge_version": VERSION,
        "timestamp": time.time(),
        "hostname": socket.gethostname(),
        "python": PYTHON,
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "gpu": gpu_info(),
        "running_job": running,
        "repo": REPO_FULL,
        "control_branch": CONTROL_BRANCH,
        "results_branch": RESULTS_BRANCH,
    }


def upload_artifacts(job_id: str, artifacts: Path) -> list[dict[str, Any]]:
    uploaded = []
    if not artifacts.exists():
        return uploaded
    for p in sorted(artifacts.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_ARTIFACT_SUFFIXES:
            continue
        size = p.stat().st_size
        rel = p.relative_to(artifacts).as_posix()
        if size > MAX_ARTIFACT_BYTES:
            uploaded.append({"path": rel, "size": size, "uploaded": False, "reason": "too_large"})
            continue
        data = p.read_bytes()
        put_file(f"{ARTIFACT_PREFIX}{job_id}/{rel}", data, f"result {job_id}: {rel}")
        uploaded.append({"path": rel, "size": size, "uploaded": True})
    return uploaded


def run_job(req: dict[str, Any]) -> dict[str, Any]:
    job_id = req["job_id"]
    job = JOBS / job_id
    src = job / "src"
    artifacts = job / "artifacts"
    log = job / "job.log"
    if job.exists():
        shutil.rmtree(job)
    job.mkdir(parents=True)
    artifacts.mkdir()

    run(["git", "worktree", "prune"], cwd=MIRROR)
    run(["git", "worktree", "add", "--detach", str(src), req["code_commit"]], cwd=MIRROR)
    entry = (src / req["entrypoint"]).resolve()
    if src.resolve() not in entry.parents or not entry.is_file():
        raise ValueError("entrypoint not present in requested commit")

    env = os.environ.copy()
    env.update(req["env"])
    env.update({
        "COTS_JOB_ID": job_id,
        "COTS_JOB_DIR": str(job),
        "COTS_JOB_ARTIFACT_DIR": str(artifacts),
        "COTS_GIT_COMMIT": req["code_commit"],
        "PYTHONUNBUFFERED": "1",
    })

    started = time.time()
    status = {**req, "state": "running", "hostname": socket.gethostname(), "started_at": started}
    put_json(f"{RESULT_PREFIX}{job_id}.json", status, f"job {job_id}: started")

    with log.open("ab", buffering=0) as f:
        proc = subprocess.Popen(
            [PYTHON, "-u", str(entry), *req["args"]], cwd=str(src), env=env,
            stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
            shell=False, preexec_fn=lambda: preexec(req["memory_gb"]), close_fds=True,
        )

    deadline = started + req["timeout_sec"]
    next_heartbeat = 0.0
    stopped = False
    timed_out = False
    while proc.poll() is None:
        now = time.time()
        if now >= deadline:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            break
        if now >= next_heartbeat:
            if stop_requested(job_id):
                stopped = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                break
            status.update({"pid": proc.pid, "updated_at": now, "log_tail": tail(log, 120)})
            put_json(f"{RESULT_PREFIX}{job_id}.json", status, f"job {job_id}: heartbeat")
            put_json(AGENT_STATUS_PATH, agent_status(job_id), "workstation heartbeat")
            next_heartbeat = now + HEARTBEAT_SEC
        time.sleep(2)

    if proc.poll() is None:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    rc = proc.returncode
    finished = time.time()
    state = "stopped" if stopped else "timed_out" if timed_out else "finished" if rc == 0 else "failed"

    log_data = log.read_bytes() if log.exists() else b""
    if len(log_data) > MAX_LOG_UPLOAD_BYTES:
        log_data = log_data[-MAX_LOG_UPLOAD_BYTES:]
    put_file(f"{ARTIFACT_PREFIX}{job_id}/job.log", log_data, f"job {job_id}: final log")
    uploaded = upload_artifacts(job_id, artifacts)

    status.update({
        "state": state,
        "returncode": rc,
        "finished_at": finished,
        "elapsed_sec": finished - started,
        "log_tail": tail(log, 300),
        "uploaded_artifacts": uploaded,
    })
    put_json(f"{RESULT_PREFIX}{job_id}.json", status, f"job {job_id}: {state}")
    put_json(AGENT_STATUS_PATH, agent_status(None), "workstation idle")

    try:
        run(["git", "worktree", "remove", "--force", str(src)], cwd=MIRROR, timeout=120)
    except Exception as e:
        print(f"warning: worktree cleanup failed: {e}", flush=True)
    return status


def reject(job_id: str, error: Exception | str, queue_path: str) -> None:
    obj = {
        "schema": 1,
        "job_id": job_id,
        "state": "rejected",
        "queue_path": queue_path,
        "timestamp": time.time(),
        "error": str(error),
        "hostname": socket.gethostname(),
    }
    put_json(f"{RESULT_PREFIX}{job_id}.json", obj, f"job {job_id}: rejected")


def self_test() -> None:
    print(f"bridge version: {VERSION}")
    print(f"repo: {REPO_FULL}")
    if not TOKEN:
        raise RuntimeError("COTS_GITHUB_TOKEN is not set")
    fetch()
    sha = run(["git", "rev-parse", f"origin/{CONTROL_BRANCH}"], cwd=MIRROR)
    print(f"origin/{CONTROL_BRANCH}: {sha}")
    ensure_results_branch()
    put_json(AGENT_STATUS_PATH, agent_status(None), "workstation self-test")
    print(f"results branch: {RESULTS_BRANCH}")
    print("GitHub read/write test: OK")
    print("entrypoint prefixes:", ALLOWED_ENTRY_PREFIXES)


def daemon(once: bool = False) -> None:
    if not TOKEN:
        raise RuntimeError("COTS_GITHUB_TOKEN is not set")
    ensure_repo()
    ensure_results_branch()
    print(f"COTS bridge {VERSION} watching {REPO_FULL}:{CONTROL_BRANCH}", flush=True)
    while True:
        try:
            put_json(AGENT_STATUS_PATH, agent_status(None), "workstation heartbeat")
            did_work = False
            for path in queue_paths():
                raw = git_show(f"origin/{CONTROL_BRANCH}", path)
                try:
                    obj = json.loads(raw)
                    guessed_id = str(obj.get("job_id") or Path(path).stem)
                    if final_result_exists(guessed_id):
                        continue
                    req = validate_request(obj, path)
                    print(f"starting {req['job_id']} @ {req['code_commit'][:12]}", flush=True)
                    result = run_job(req)
                    print(f"finished {req['job_id']}: {result['state']} rc={result['returncode']}", flush=True)
                    did_work = True
                    break
                except Exception as e:
                    job_id = str((locals().get("obj") or {}).get("job_id") or Path(path).stem)
                    print(f"rejecting {path}: {e}", file=sys.stderr, flush=True)
                    try:
                        reject(job_id, e, path)
                    except Exception as e2:
                        print(f"could not publish rejection: {e2}", file=sys.stderr, flush=True)
            if once:
                return
            if not did_work:
                time.sleep(POLL_SEC)
        except KeyboardInterrupt:
            print("stopped by operator", flush=True)
            return
        except Exception as e:
            print(f"bridge loop error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            if once:
                raise
            time.sleep(POLL_SEC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="verify git and GitHub result write access")
    ap.add_argument("--once", action="store_true", help="process at most one queue scan, then exit")
    ns = ap.parse_args()
    if ns.self_test:
        self_test()
    else:
        daemon(once=ns.once)


if __name__ == "__main__":
    main()
