#!/usr/bin/env bash
set -euo pipefail
cd /root/voiceguard
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
for attempt in $(seq 1 112); do
  state=$(.venv/bin/python -c 'import json; print(json.load(open("outputs/joint_recovery/2026-11-09/validation-state.json"))["state"])')
  case "$state" in
    complete) break ;;
    running) sleep 45 ;;
    *) echo "First joint validation failed: $state"; exit 1 ;;
  esac
done
if [ "$state" != complete ]; then echo 'First validation dependency timed out'; exit 1; fi
bash deploy/run_anchored_joint.sh development
.venv/bin/python scripts/run_anchored_joint.py --split select > outputs/joint_anchored/selection.log 2>&1
if .venv/bin/python tools/decide_anchored_validation.py > outputs/joint_anchored/decision.log 2>&1; then
  bash deploy/run_anchored_joint.sh validation
  .venv/bin/python tools/analyze_joint_recovery.py --evidence docs/evidence/2026-11-09-joint-anchored \
    > outputs/joint_anchored/analysis.log 2>&1
else
  decision_status=$?
  if [ "$decision_status" -ne 3 ]; then exit "$decision_status"; fi
  echo 'Development gate failed: extension ends without validation'
fi
