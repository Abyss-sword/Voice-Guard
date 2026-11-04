#!/usr/bin/env bash
# Run the two independent experiment groups, then quality evaluation and tables.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p outputs/cloud
exec >>outputs/cloud/campaign.log 2>&1
max_hours="${VOICEGUARD_STAGE_MAX_HOURS:-2}"
.venv/bin/python tools/preflight.py --scope experiment --out outputs/cloud/preflight.json
date -u
finish() {
  campaign_code=$?
  .venv/bin/python tools/build_cloud_report.py --exit-code "$campaign_code"
}
trap finish EXIT
# Separate processes preserve independent random states and result directories.
.venv/bin/python -u scripts/reproduce.py main --max-hours "$max_hours" >outputs/cloud/main.log 2>&1 &
main_pid=$!
.venv/bin/python -u scripts/reproduce.py eps --max-hours "$max_hours" >outputs/cloud/eps.log 2>&1 &
eps_pid=$!
set +e
wait "$main_pid"
main_code=$?
wait "$eps_pid"
eps_code=$?
set -e
if (( main_code != 0 || eps_code != 0 )); then
  echo "Experiment failure: main=$main_code eps=$eps_code; inspect group logs and resume."
  exit 1
fi
for directory in outputs/cloud/main_* outputs/cloud/eps_*; do
  [[ -f "$directory/results.csv" ]] || continue
  .venv/bin/python scripts/compute_mos.py --exp-dir "$directory" --out "$directory/mos.csv" --device cuda
  .venv/bin/python scripts/compute_wer.py --exp-dir "$directory" --vctk data/VCTK22 \
    --source-speaker p226 --source-utt 2 --out "$directory/wer.csv" --device cuda
done
date -u
