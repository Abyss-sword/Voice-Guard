# 华为云部署入口

本轮以 **GPU ECS + Python venv** 为首选：同一台机器用于完整攻击实验和 Gradio 演示，
避免把长时攻击当作在线推理请求。逐步操作见 [部署操作手册](../docs/07_部署操作手册.md)。

## ECS 原生运行

Ubuntu 22.04 + NVIDIA GPU 驱动，进入项目根目录：

```bash
bash deploy/ecs_bootstrap.sh
.venv/bin/python scripts/reproduce.py models
.venv/bin/python scripts/reproduce.py data --max-hours 2
.venv/bin/python scripts/reproduce.py registry
.venv/bin/python tools/preflight.py --scope experiment --out outputs/cloud/preflight.json
.venv/bin/python scripts/reproduce.py smoke
.venv/bin/python scripts/reproduce.py benchmark --max-hours 2
bash deploy/start_ecs.sh
```

服务绑定 ECS 的 127.0.0.1:7860。本地使用 `ssh -N -L 7860:127.0.0.1:7860 用户名@公网IP`，
浏览器打开 `http://127.0.0.1:7860`。安全组只需允许本人 IP 访问 SSH 22。

## Docker（可选）

镜像只含代码；权重、注册库与数据在运行时挂载。构建上下文排除大文件和本地凭证，
补齐 `third_party/ada_in_vc`、ffmpeg 与 libsndfile，避免原镜像启动后缺模块。

```bash
docker build -f deploy/Dockerfile -t voiceguard:local .
# 需要 NVIDIA Container Toolkit；Compose >= 2.30 才支持 gpus: all
docker compose -f deploy/compose.yaml up -d
docker compose -f deploy/compose.yaml ps
docker compose -f deploy/compose.yaml logs --tail 100
```

实验命令也可在同一镜像中运行（无需启动网页，输出挂载到宿主机）：

```bash
docker run --rm --gpus all -v "$PWD/checkpoints:/app/checkpoints" \
  -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" \
  voiceguard:local python scripts/reproduce.py smoke
```

CPU 仅用于开发验证：手动 `docker run` 时省略 `--gpus all` 并设置 `-e VG_DEVICE=cpu`。
镜像不会在未识别 GPU 时静默降级。

## SWR / ModelArts

镜像可推 SWR，但本轮没有在账号中执行推送或创建服务。ModelArts 的鉴权、路径代理、
Gradio 队列与 WebSocket/SSE、超时和权重挂载需单独实测；不得把 REST 推理地址直接
当作已验证的完整 Gradio 网站。旧文档的“T4 固定规格名 / 公网端口直接可用”未核实，已移除。

## 验收证据

保留 ECS 区域/规格与订单截图、`nvidia-smi`、`runtime.json`、`preflight.json`、
实验 `manifest.json/result.json/results.csv/summary.json`、服务日志和一次真实音频操作截图。
资源价格、峰值显存、单句耗时应填写该 ECS 的实测值，不沿用历史 4060 数据。
