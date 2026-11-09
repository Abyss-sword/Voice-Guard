#!/usr/bin/env bash
set -euo pipefail
cd /root/voiceguard
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
phase="${1:?development or validation required}"
if [[ "$phase" != development && "$phase" != validation ]]; then exit 2; fi
mkdir -p outputs/paper_quality
args=(--protocol docs/evidence/2026-11-09-quality/recovery-protocol.json --split "$phase" --max-hours 0.98)
if [[ "$phase" == validation ]]; then
  args+=(--selection docs/evidence/2026-11-09-quality/recovery-selection.json)
fi
.venv/bin/python -c 'import hashlib,json,pathlib; p=json.loads(pathlib.Path("docs/evidence/2026-11-09-quality/recovery-protocol.json").read_text()); assert hashlib.sha256((pathlib.Path.home()/".cache/whisper/base.pt").read_bytes()).hexdigest()==p["whisper_base_sha256"]'
timeout --signal=TERM --kill-after=20s 3600s .venv/bin/python -u scripts/run_quality_recovery.py "${args[@]}" > "outputs/paper_quality/${phase}.log" 2>&1
