#!/usr/bin/env python3
"""Copy an existing bridge job log into this job's artifact directory."""
from __future__ import annotations
import argparse, json, os, shutil, time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-job", required=True)
    ns = ap.parse_args()
    root = Path(os.environ.get("COTS_BRIDGE_ROOT", "~/.cots_lamcts_bridge")).expanduser().resolve()
    src = root / "jobs" / ns.target_job / "job.log"
    out = Path(os.environ["COTS_JOB_ARTIFACT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    info = {"target_job": ns.target_job, "source": str(src), "exists": src.is_file(), "snapshot_at": time.time()}
    if src.is_file():
        dst = out / "orphan_job.log"
        shutil.copyfile(src, dst)
        info.update({"size": dst.stat().st_size, "mtime": src.stat().st_mtime})
    (out / "export_info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
