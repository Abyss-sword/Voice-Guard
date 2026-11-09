#!/usr/bin/env bash
set -euo pipefail
cd /root/voiceguard
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
phase="${1:?development or validation required}"
case "$phase" in
  development) seconds=1290; hours=0.35 ;;
  validation) seconds=2400; hours=0.65 ;;
  *) exit 2 ;;
esac
mkdir -p outputs/joint_anchored
timeout --signal=INT --kill-after=20s "$seconds" .venv/bin/python -u \
  scripts/run_anchored_joint.py --split "$phase" --max-hours "$hours" \
  > "outputs/joint_anchored/${phase}.log" 2>&1
