#!/usr/bin/env bash
# The course frontend reaches this loopback service through an SSH tunnel.
set -euo pipefail
cd "$(dirname "$0")/.."
project_dir="$(pwd)"
service_user="${VOICEGUARD_SERVICE_USER:-$(id -un)}"
cat > /etc/systemd/system/voiceguard-platform.service <<UNIT
[Unit]
Description=Voice Guard course platform adapter
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$service_user
WorkingDirectory=$project_dir
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=PYTHONUNBUFFERED=1
ExecStart=$project_dir/.venv/bin/python platform/service/adapter.py --model-dir checkpoints/adain-vc-official --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz --vctk data/VCTK22 --device cuda --host 127.0.0.1 --port 8020 --iters1 3000 --iters2 1500
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now voiceguard-platform.service
systemctl status voiceguard-platform.service --no-pager
