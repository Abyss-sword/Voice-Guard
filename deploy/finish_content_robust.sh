#!/usr/bin/env bash
set -euo pipefail
cd /root/voiceguard
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
bash deploy/run_content_robust.sh development
.venv/bin/python scripts/run_content_robust.py --split select > outputs/content_robust/selection.log 2>&1
if .venv/bin/python tools/decide_content_validation.py > outputs/content_robust/decision.log 2>&1; then
  bash deploy/run_content_robust.sh validation
else
  decision_status=$?
  if [ "$decision_status" -ne 3 ]; then exit "$decision_status"; fi
  echo 'Development gate failed: fresh validation is not scored'
fi
.venv/bin/python tools/audit_content_results.py --evidence docs/evidence/2026-11-09-content-robust \
  > outputs/content_robust/analysis.log 2>&1
