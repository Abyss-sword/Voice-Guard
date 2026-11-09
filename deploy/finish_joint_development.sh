#!/usr/bin/env bash
# A bounded one-off dependency chain, not a recurring automation.
set -euo pipefail
cd /root/voiceguard
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
for attempt in $(seq 1 360); do
  state=$(.venv/bin/python -c 'import json; print(json.load(open("outputs/joint_recovery/2026-11-09/development-state.json"))["state"])')
  case "$state" in
    complete) break ;;
    running) sleep 5 ;;
    *) echo "Development failed: $state"; exit 1 ;;
  esac
done
if [ "$state" != complete ]; then echo 'Development dependency timed out'; exit 1; fi
if [ ! -f docs/evidence/2026-11-09-joint/selection.json ]; then
  .venv/bin/python scripts/run_joint_recovery.py --split select > outputs/joint_recovery/selection.log 2>&1
fi
if [ ! -f docs/evidence/2026-11-09-joint/clean-chain-diagnostic.json ]; then
  timeout --signal=INT --kill-after=20s 300 .venv/bin/python -u tools/diagnose_joint_controls.py \
    > outputs/joint_recovery/diagnostic.log 2>&1
fi
bash deploy/run_joint_recovery.sh validation
.venv/bin/python tools/analyze_joint_recovery.py > outputs/joint_recovery/analysis.log 2>&1
