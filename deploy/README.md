# 部署说明（华为云）

## 方案 A：ModelArts 自定义镜像在线服务（推荐）

1. 本地/云端构建镜像（见 `Dockerfile` 顶部命令），推到 SWR（容器镜像服务）：
   ```
   docker tag voiceguard:latest swr.cn-north-4.myhuaweicloud.com/<组织>/voiceguard:latest
   docker push swr.cn-north-4.myhuaweicloud.com/<组织>/voiceguard:latest
   ```
2. ModelArts → AI 应用管理 → 创建 AI 应用：
   - 元模型来源：容器镜像 → 选择 SWR 里的 voiceguard 镜像
   - 运行环境调优：GPU 规格（如 `modelarts.vm.gpu.t4`）
   - 容器调用接口：选择"HTTP"，服务端口 7860
3. 部署 → 在线服务：选 GPU 实例、启动。
4. 验收现场直接访问服务地址（公网映射）使用 Gradio 界面；或用 REST：
   - Gradio 页面：`http://<服务地址>:7860`
   - 健康检查：`curl http://<服务地址>:7860/` 返回 200。

## 方案 B：ECS + Docker（兜底）

1. 买一台带 GPU 的 ECS（或用无 GPU 实例跑 CPU 版，`--device cpu`，速度慢但可演示）。
2. 装 docker + nvidia-container-toolkit，`docker run --gpus all -p 7860:7860 voiceguard:latest`。
3. 安全组放行 7860。

## 资源核验记录（README 要求）

| 项 | 值（本地 4060 Laptop 8GB 实测，待补） |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU（8 GB） |
| 峰值显存 | ~3.7 GB（batch=1，4.3s 音频，1500 步） |
| 单条耗时 | ~20-40 min @ 3000+1500 步（随音频长度线性） |
| CPU 版 | 可用（`--device cpu`），仅演示 |
