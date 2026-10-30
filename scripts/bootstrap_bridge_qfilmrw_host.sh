#!/usr/bin/env bash
# Host administrator recovery path; QfilmRW migration: Bridge 0.6.1, verified public source, no credentials read.
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then
  echo 'Run this reviewed script with sudo on the workstation.' >&2
  exit 1
fi
bootstrap_dir=/var/lib/cots-lamcts/bootstrap-bridge-v06-7b8766c9d6e4
install -d -m 755 "$bootstrap_dir/infra" "$bootstrap_dir/scripts" "$bootstrap_dir/install-report"
curl --fail --show-error --location --retry 3 --retry-delay 2 --connect-timeout 10 --max-time 90 'https://raw.githubusercontent.com/wangteng1111/cots-LAMCTS/7b8766c9d6e4da5ddebbe46397d5c792b8508bf9/infra/cots_workstation_bridge.py' --output "$bootstrap_dir/infra/cots_workstation_bridge.py"
curl --fail --show-error --location --retry 3 --retry-delay 2 --connect-timeout 10 --max-time 90 'https://raw.githubusercontent.com/wangteng1111/cots-LAMCTS/7b8766c9d6e4da5ddebbe46397d5c792b8508bf9/infra/cots_workstation_bridge_v04.py' --output "$bootstrap_dir/infra/cots_workstation_bridge_v04.py"
curl --fail --show-error --location --retry 3 --retry-delay 2 --connect-timeout 10 --max-time 90 'https://raw.githubusercontent.com/wangteng1111/cots-LAMCTS/7b8766c9d6e4da5ddebbe46397d5c792b8508bf9/infra/cots_workstation_bridge_v05.py' --output "$bootstrap_dir/infra/cots_workstation_bridge_v05.py"
curl --fail --show-error --location --retry 3 --retry-delay 2 --connect-timeout 10 --max-time 90 'https://raw.githubusercontent.com/wangteng1111/cots-LAMCTS/7b8766c9d6e4da5ddebbe46397d5c792b8508bf9/infra/cots_workstation_bridge_v06.py' --output "$bootstrap_dir/infra/cots_workstation_bridge_v06.py"
curl --fail --show-error --location --retry 3 --retry-delay 2 --connect-timeout 10 --max-time 90 'https://raw.githubusercontent.com/wangteng1111/cots-LAMCTS/7b8766c9d6e4da5ddebbe46397d5c792b8508bf9/infra/cots_bridge_worker.py' --output "$bootstrap_dir/infra/cots_bridge_worker.py"
curl --fail --show-error --location --retry 3 --retry-delay 2 --connect-timeout 10 --max-time 90 'https://raw.githubusercontent.com/wangteng1111/cots-LAMCTS/7b8766c9d6e4da5ddebbe46397d5c792b8508bf9/infra/qfilm_convergence_status.py' --output "$bootstrap_dir/infra/qfilm_convergence_status.py"
curl --fail --show-error --location --retry 3 --retry-delay 2 --connect-timeout 10 --max-time 90 'https://raw.githubusercontent.com/wangteng1111/cots-LAMCTS/7b8766c9d6e4da5ddebbe46397d5c792b8508bf9/scripts/deploy_bridge_v06.py' --output "$bootstrap_dir/scripts/deploy_bridge_v06.py"
curl --fail --show-error --location --retry 3 --retry-delay 2 --connect-timeout 10 --max-time 90 'https://raw.githubusercontent.com/wangteng1111/cots-LAMCTS/7b8766c9d6e4da5ddebbe46397d5c792b8508bf9/scripts/test_bridge_v06.py' --output "$bootstrap_dir/scripts/test_bridge_v06.py"
cat > "$bootstrap_dir/SHA256SUMS" <<'QFILM_CHECKSUMS'
ef04d9aedfe37d553d850abe7f26c812b3ea046825e2778cacd238a5f1e776fc  infra/cots_workstation_bridge.py
3d53e368e3ac59e104b966497d7ee1af9a6a1fc9881f0cf8b037f8ff5289cdbf  infra/cots_workstation_bridge_v04.py
7bd93fb2325f71ad37f24ebcfbd822d862a73595ce11970bbee3157618a0e9ee  infra/cots_workstation_bridge_v05.py
0bd1e7a462a6592234c44606eac7bfc740b2c7b0105db416b06e70dfdccd0ef5  infra/cots_workstation_bridge_v06.py
3356d46db69a2cf31b164c60986c0c0887733b03bd59c91c7df6eee621df123b  infra/cots_bridge_worker.py
83ebc632f1b4e7dc4a9112b59da974be12b0183698e1dd274227492469b0d302  infra/qfilm_convergence_status.py
f3ebe9a909cc8325747d8b93a1ec10bf8c2943d342b34dbeb5a053e6a4f55530  scripts/deploy_bridge_v06.py
42765e564f343a6a0ca499f1192af4e16faefa59ab36bd3723057212442f4ca5  scripts/test_bridge_v06.py
QFILM_CHECKSUMS
(cd "$bootstrap_dir" && sha256sum --check SHA256SUMS)
export COTS_BRIDGE_ROOT=/var/lib/cots-lamcts
export COTS_GIT_COMMIT=7b8766c9d6e4da5ddebbe46397d5c792b8508bf9
export COTS_JOB_ARTIFACT_DIR="$bootstrap_dir/install-report"
/usr/bin/python3 "$bootstrap_dir/scripts/deploy_bridge_v06.py"
activation_status=$(/usr/bin/python3 -c 'import json,os;from pathlib import Path;print(json.loads((Path(os.environ["COTS_JOB_ARTIFACT_DIR"])/"bridge-v06-deployment.json").read_text())["status"])')
if [ "$activation_status" = staged_requires_host_activation ]; then
  /usr/bin/python3 /var/lib/cots-lamcts/runtime/bridge-v06-7b8766c9d6e4/scripts/deploy_bridge_v06.py --activate
elif [ "$activation_status" != activation_scheduled ]; then
  echo "Activation stopped: $activation_status" >&2
  exit 1
fi
systemctl show cots-lamcts-bridge.service --property=ActiveState,SubState,MainPID,KillMode,ExecStart
