# 部署说明（华为云）

> 完整逐字段操作见 **[docs/07_部署操作手册.md](../docs/07_部署操作手册.md)**（路线 A：ECS 直跑；路线 B：镜像 + SWR + ModelArts）。
> 本机**没有 Docker 也能部署**：路线 A 不需要 Docker，路线 B 在 ECS 上构建镜像。

## 快速步骤

**0）生成部署精简包（本机）**

```powershell
cd "D:\Voice Guard"
.\.venv\Scripts\python.exe tools\make_cloud_bundle.py --zip
# 产物 deploy\cloud_bundle.zip（约 123 MB：代码+权重+注册库+每说话人 1 句演示音频）
```

**A）ECS 直接运行（最快，推荐验收演示）**

1. 买一台 ECS（Ubuntu 22.04，CPU 或 T4 GPU），安全组放行 7860。
2. `scp deploy\cloud_bundle.zip ubuntu@<公网IP>:/home/ubuntu/` → 解压 → 建 venv。
3. 按 `docs/07` §2.5 安装依赖（固定版本链），§2.6 启动：
   ```bash
   MPLCONFIGDIR=/tmp/mpl nohup python frontend/app.py \
     --model-dir checkpoints/adain-vc-official \
     --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz \
     --vctk data/VCTK22 --device cuda --server-name 0.0.0.0 --server-port 7860 > app.log 2>&1 &
   ```
4. 浏览器访问 `http://<公网IP>:7860`。

**B）镜像 + SWR + ModelArts（正式在线服务地址）**

1. ECS 上 `sudo docker build -f deploy/Dockerfile -t voiceguard:latest .`
2. 推到 SWR（组织 `voiceguard`，区域与 ECS 一致）：
   ```bash
   sudo docker tag voiceguard:latest swr.cn-north-4.myhuaweicloud.com/voiceguard/voiceguard:latest
   sudo docker push swr.cn-north-4.myhuaweicloud.com/voiceguard/voiceguard:latest
   ```
3. ModelArts → AI 应用管理 → 创建 AI 应用：镜像选 SWR 里的 voiceguard，
   容器调用接口 HTTP，端口 **7860**，健康检查 **HTTP GET `/`、超时 60s**，GPU T4。
4. 部署在线服务 → 拿地址 → 浏览器打开。

## 资源核验记录（课件「资源核验」加分材料）

| 项 | 值（RTX 4060 Laptop 8 GB 实测） |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU（8 GB） |
| 峰值显存 | ~4.4 GB（batch=1，4.3s 音频，1500 步） |
| 单条耗时 | 1.2s 句 ~17 min；4.3s 句 ~40 min（@3000+1500 / 1500+750 步） |
| 演示快速模式 | 300/200 步：2s 句约 2–3 分钟 |
| CPU 版 | 可用（`--device cpu`），仅适合演示快速模式 |
