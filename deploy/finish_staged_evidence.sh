#!/usr/bin/env bash
# Run only after the frozen validation has ended. Never changes its selection.
set -euo pipefail
cd /root/voiceguard
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
while tmux has-session -t vg-final-validation 2>/dev/null; do sleep 10; done
.venv/bin/python -u tools/audit_native_validation.py
.venv/bin/python -u scripts/build_registry.py --mode vc \
  --model-dir checkpoints/adain-vc-official --vctk data/VCTK22 \
  --speakers p225,p226,p227,p228,p229,p230,p231,p232,p233,p234,p236,p237,p238,p239,p240,p241,p243,p244,p245,p246 \
  --per-speaker 30 --device cuda \
  --out outputs/staged_recovery/2026-11-09/production-registry-native.npz
.venv/bin/python tools/verify_native_registry.py
.venv/bin/python tools/analyze_staged_recovery.py
initial=outputs/staged_recovery/2026-11-09
evidence=docs/evidence/2026-11-09-staged
cp "$initial"/development-{state,results}.json "$evidence"/
cp "$initial"/strong-audit/results.json "$evidence"/strong-stage-development-audit.json
cp "$initial"/final/{development,validation}-{state,results}.json "$evidence"/final/
cp "$initial"/final/native-validation-audit.json "$evidence"/final/
cp "$initial"/production-registry-check.json "$evidence"/final/
.venv/bin/python tools/archive_staged_recovery.py
