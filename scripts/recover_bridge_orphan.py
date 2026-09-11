#!/usr/bin/env python3
"""Read-only recovery probe for orphaned COTS bridge jobs.

Inspects a previous bridge job directory and Linux PID without modifying or
terminating the target process. Writes a compact recovery JSON artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

ROUND_RE = re.compile(r"ROUND\s+(\d+)\s+n\s+(\d+)\s+bestJ\s+([^\s]+)\s+best\s+(\([^\n]+?\))\s+eligible\s+(\d+)")
Q_RE = re.compile(r"^Q\s+round(\d+)-s(\d+)\s+\d+\s+(\([^\n]+?\))\s+([^\s]+)$", re.M)


def tail(path: Path, lines: int = 200) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    text = data[-2_000_000:].decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def pid_info(pid: int) -> dict:
    proc = Path(f"/proc/{pid}")
    out = {"pid": pid, "alive": proc.exists()}
    if not proc.exists():
        return out
    try:
        out["cmdline"] = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except Exception as e:
        out["cmdline_error"] = repr(e)
    try:
        status = {}
        for line in (proc / "status").read_text(errors="replace").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                if k in {"Name", "State", "PPid", "Threads", "VmRSS", "VmSize"}:
                    status[k] = v.strip()
        out["status"] = status
    except Exception as e:
        out["status_error"] = repr(e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-job", required=True)
    ap.add_argument("--pid", type=int, default=0)
    ns = ap.parse_args()

    root = Path(os.environ.get("COTS_BRIDGE_ROOT", "~/.cots_lamcts_bridge")).expanduser().resolve()
    target = root / "jobs" / ns.target_job
    log = target / "job.log"
    artifacts = target / "artifacts"
    log_tail = tail(log, 400)

    rounds = []
    for m in ROUND_RE.finditer(log_tail):
        rounds.append({
            "round": int(m.group(1)),
            "n": int(m.group(2)),
            "bestJ": float(m.group(3)),
            "best_state": m.group(4),
            "eligible": int(m.group(5)),
        })

    q_matches = []
    for m in Q_RE.finditer(log_tail):
        try:
            j = float(m.group(4))
        except Exception:
            continue
        q_matches.append({"round": int(m.group(1)), "seed": int(m.group(2)), "state": m.group(3), "J": j})

    files = []
    if artifacts.exists():
        for p in sorted(artifacts.rglob("*")):
            if p.is_file():
                files.append({"path": p.relative_to(artifacts).as_posix(), "size": p.stat().st_size, "mtime": p.stat().st_mtime})

    result = {
        "schema": 1,
        "probe_time": time.time(),
        "bridge_root": str(root),
        "target_job": ns.target_job,
        "target_dir_exists": target.exists(),
        "log_exists": log.is_file(),
        "log_size": log.stat().st_size if log.is_file() else None,
        "log_mtime": log.stat().st_mtime if log.is_file() else None,
        "pid": pid_info(ns.pid) if ns.pid else None,
        "last_complete_round": rounds[-1] if rounds else None,
        "best_q_in_tail": min(q_matches, key=lambda x: x["J"]) if q_matches else None,
        "artifact_files": files,
        "log_tail": log_tail,
    }

    outdir = Path(os.environ["COTS_JOB_ARTIFACT_DIR"])
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "recovery.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ["target_job", "target_dir_exists", "log_exists", "log_size", "log_mtime", "pid", "last_complete_round", "best_q_in_tail", "artifact_files"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
