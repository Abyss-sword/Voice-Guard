# C18 云端成本控制：自动关机与留证模板
#
# ⚠️ 这是模板，不会自动执行。请在 ECS 上按需复制。
#
# 背景：实验脚本的 --max-hours 只限制脚本运行时间，**不停止 ECS 计费**。
# 本模板提供三种「跑完自动关机」的方式，避免忘记关机产生费用。
#
# ---------------------------------------------------------------------------
# 方式 A：跑实验时用 timeout 包裹 + 结束自动关机（最直接）
# ---------------------------------------------------------------------------
#   cd /root/voiceguard
#   nohup bash -c '
#     .venv/bin/python deploy/run_cloud_enhancements.py --max-hours 6 \
#       && sudo shutdown -h +2 "auto shutdown after campaign"
#   ' > outputs/cloud/nohup.log 2>&1 &
#
# ---------------------------------------------------------------------------
# 方式 B：空闲自动关机（当 GPU 空闲超过阈值时）
# ---------------------------------------------------------------------------
#   cat > /usr/local/bin/idle_shutdown.sh <<'EOF'
#   #!/usr/bin/env bash
#   set -euo pipefail
#   THRESHOLD_MB=${1:-50}
#   used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
#   if (( used < THRESHOLD_MB )); then
#     echo "$(date -u) GPU idle (${used}MiB < ${THRESHOLD_MB}MiB), shutting down"
#     # 留证：先打包输出
#     cd /root/voiceguard && tar -czf /root/voiceguard-$(date -u +%Y%m%dT%H%M%SZ).tar.gz outputs/cloud || true
#     shutdown -h now
#   fi
#   EOF
#   chmod +x /usr/local/bin/idle_shutdown.sh
#   # 每 15 分钟检查一次
#   (crontab -l 2>/dev/null; echo '*/15 * * * * /usr/local/bin/idle_shutdown.sh 50 >> /var/log/idle_shutdown.log 2>&1') | crontab -
#
# ---------------------------------------------------------------------------
# 方式 C：定时关机（例如每天凌晨 4 点，防止通宵 idle 计费）
# ---------------------------------------------------------------------------
#   sudo systemctl enable --now systemd-timer  # 或直接在 cron:
#   (crontab -l 2>/dev/null; echo '0 4 * * * /sbin/shutdown -h now "scheduled auto shutdown"') | crontab -
#
# ---------------------------------------------------------------------------
# 关机前留证清单（建议写入同一 cron：先 tar 再关机）
# ---------------------------------------------------------------------------
#   tar -czf /root/voiceguard-evidence-$(date -u +%Y%m%dT%H%M%SZ).tar.gz \
#       outputs/cloud outputs/enhancements outputs/quality outputs/staged_recovery \
#       docs/evidence 2>/dev/null || true
#   # 然后 scp 下载到本机存档（tar 体积可能数百 MB）
#   # scp -i C:\keys\voiceguard.pem ubuntu@<IP>:/root/voiceguard-evidence-*.tar.gz ./
#
# ---------------------------------------------------------------------------
# 恢复开机后自检
# ---------------------------------------------------------------------------
#   cd /root/voiceguard && git pull
#   .venv/bin/python tools/preflight.py --scope experiment --out outputs/cloud/preflight.json
#   systemctl start voiceguard voiceguard-platform   # 若使用 systemd 服务
echo "C18 模板：请按上面注释选择一种方式配置；本文件不会自动执行。"
