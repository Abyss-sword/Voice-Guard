#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export GRADIO_ANALYTICS_ENABLED=False
export MPLCONFIGDIR="$PWD/.mplcache"
.venv/bin/python tools/preflight.py --scope experiment --out outputs/cloud/preflight.json
exec .venv/bin/python frontend/app.py --model-dir checkpoints/adain-vc-official \
  --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz \
  --vctk data/VCTK22 --device cuda --server-name 127.0.0.1 --server-port 7860
