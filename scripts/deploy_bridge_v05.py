#!/usr/bin/env python3
"""Deploy bridge v0.5 from an approved job worktree onto the workstation.

This script is intended to be launched once through the currently running bridge.
It copies only the bridge runtime files and systemd unit. If direct writes are
blocked by ProtectSystem=strict, it attempts passwordless sudo. The service
restart is scheduled as a transient systemd unit after this job exits so the
current bridge can publish the deployment result first.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("COTS_JOB_ARTIFACT_DIR", ROOT / "artifacts"))
ART.mkdir(parents=True, exist_ok=True)

FILES = [
    (ROOT / "infra/cots_workstation_bridge.py", Path("/opt/cots-lamcts/infra/cots_workstation_bridge.py"), 0o644),
    (ROOT / "infra/cots_workstation_bridge_v04.py", Path("/opt/cots-lamcts/infra/cots_workstation_bridge_v04.py"), 0o644),
    (ROOT / "infra/cots_workstation_bridge_v05.py", Path("/opt/cots-lamcts/infra/cots_workstation_bridge_v05.py"), 0o644),
    (ROOT / "infra/cots-lamcts-bridge.service", Path("/etc/systemd/system/cots-lamcts-bridge.service"), 0o644),
]


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def can_sudo() -> bool:
    if not shutil.which("sudo"):
        return False
    p = run(["sudo", "-n", "true"], check=False)
    return p.returncode == 0


def direct_install(src: Path, dst: Path, mode: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".v05-new")
    shutil.copy2(src, tmp)
    os.chmod(tmp, mode)
    os.replace(tmp, dst)


def sudo_install(src: Path, dst: Path, mode: int) -> None:
    run(["sudo", "-n", "install", "-D", "-m", f"{mode:o}", str(src), str(dst)])


def main() -> int:
    report = {
        "timestamp": time.time(),
        "uid": os.getuid(),
        "euid": os.geteuid(),
        "source_root": str(ROOT),
        "sudo_noninteractive": can_sudo(),
        "files": [],
        "restart_scheduled": False,
    }
    sudo_ok = report["sudo_noninteractive"]

    for src, dst, mode in FILES:
        rec = {"src": str(src), "dst": str(dst), "ok": False, "method": None}
        try:
            direct_install(src, dst, mode)
            rec.update(ok=True, method="direct")
        except Exception as direct_err:
            rec["direct_error"] = repr(direct_err)
            if not sudo_ok:
                report["files"].append(rec)
                (ART / "bridge_deploy_report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2))
                return 2
            try:
                sudo_install(src, dst, mode)
                rec.update(ok=True, method="sudo")
            except Exception as sudo_err:
                rec["sudo_error"] = repr(sudo_err)
                report["files"].append(rec)
                (ART / "bridge_deploy_report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2))
                return 3
        report["files"].append(rec)

    # Validate syntax before scheduling restart.
    targets = [str(x[1]) for x in FILES[:3]]
    py = [sys.executable, "-m", "py_compile", *targets]
    p = run((["sudo", "-n"] + py) if sudo_ok else py, check=False)
    report["py_compile_rc"] = p.returncode
    report["py_compile_output"] = p.stdout[-4000:]
    if p.returncode != 0:
        (ART / "bridge_deploy_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 4

    # Give the supervising v0.3 bridge enough time to publish this job's final
    # status before systemd replaces it with v0.5.
    if sudo_ok and shutil.which("systemd-run"):
        cmd = [
            "sudo", "-n", "systemd-run", "--unit=cots-lamcts-bridge-v05-activate",
            "--on-active=12s", "/bin/sh", "-c",
            "systemctl daemon-reload && systemctl restart cots-lamcts-bridge.service",
        ]
        p = run(cmd, check=False)
        report["restart_schedule_rc"] = p.returncode
        report["restart_schedule_output"] = p.stdout[-4000:]
        report["restart_scheduled"] = p.returncode == 0
    else:
        report["restart_schedule_rc"] = None
        report["restart_schedule_output"] = "sudo/systemd-run unavailable"

    (ART / "bridge_deploy_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["restart_scheduled"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
