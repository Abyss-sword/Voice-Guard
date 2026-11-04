#!/usr/bin/env bash
# Install from the uploaded project after experiment preflight succeeds.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(id -u)" != 0 ]]; then
  echo 'Run with sudo bash deploy/install_ecs_service.sh'; exit 1
fi
project_dir="$PWD"
service_user="${SUDO_USER:-$(id -un)}"
.venv/bin/python tools/preflight.py --scope experiment --out outputs/cloud/preflight.json
cat > /etc/systemd/system/voiceguard.service <<EOF
[Unit]
Description=Voice Guard GPU demonstration
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$service_user
WorkingDirectory=$project_dir
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/bash "$project_dir/deploy/start_ecs.sh"
Restart=on-failure
RestartSec=5
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now voiceguard
systemctl status voiceguard --no-pager
