#!/usr/bin/env bash
# Run from project root on Ubuntu 22.04 GPU ECS with an NVIDIA driver preinstalled.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux ]]; then echo 'Requires Linux ECS'; exit 1; fi
if ! command -v nvidia-smi >/dev/null; then
  echo 'NVIDIA driver missing. Install the driver for this ECS GPU/image, reboot, then retry.'; exit 1
fi
nvidia-smi
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip ffmpeg libsndfile1 curl tmux
python3 -m venv .venv
# This mirror serves CUDA-enabled Linux torch wheels and their pinned CUDA dependencies.
# Override for another region or an approved package mirror without editing this script.
package_index="${VOICEGUARD_PIP_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple/}"
.venv/bin/python -m pip install --index-url "$package_index" --upgrade pip
.venv/bin/python -m pip install --index-url "$package_index" -r deploy/requirements.txt
.venv/bin/python -m pip check
.venv/bin/python tools/preflight.py --device cuda --out outputs/cloud/runtime.json
echo 'Runtime ready. Next: .venv/bin/python scripts/reproduce.py models'
