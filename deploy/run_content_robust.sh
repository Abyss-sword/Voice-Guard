#!/usr/bin/env bash
set -euo pipefail
cd /root/voiceguard
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
phase="${1:?development or validation required}"
case "$phase" in
  development) seconds=3630; hours=1.0 ;;
  validation) seconds=4710; hours=1.3 ;;
  *) exit 2 ;;
esac
mkdir -p outputs/content_robust
timeout --signal=INT --kill-after=20s "$seconds" .venv/bin/python -u \
  scripts/run_content_robust.py --split "$phase" --max-hours "$hours" \
  > "outputs/content_robust/${phase}.log" 2>&1
