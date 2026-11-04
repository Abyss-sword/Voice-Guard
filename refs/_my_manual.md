# 华为云部署详细操作手册（Voice Guard）

> 目标：把 Gradio 交互界面部署到华为云，验收现场可直接打开链接演示。
> 编写原则：**本机没有 Docker 也能做**；每一步都写到「点哪个按钮 / 填什么字段 / 跑什么命令」。
> 预计总耗时：路线 A 约 40 分钟；路线 B 约 60–90 分钟（含镜像构建）。
> 所有命令可在 Windows PowerShell 或 ECS 的 SSH 终端执行，标注了 `[本机]` / `[ECS]`。

---

## 0. 路线总览：先选一条

| 路线 | 做法 | 是否需要本地 Docker | 拿到什么 | 耗时 | 计费 | 推荐度 |
|---|---|---|---|---|---|---|
| **A. ECS 直接运行** | 买一台 ECS → 上传部署包 → 装依赖 → `python app.py` | ❌ 不需要 | 公网 IP:7860 可访问的界面 | ~40 min | 按量 ≈ 1–2 元/时（CPU） | ⭐⭐⭐ 验收演示首选 |
| **B. 镜像 + ModelArts 在线服务** | ECS 上构建镜像 → 推 SWR → ModelArts AI 应用 → 在线服务 | ❌ 不需要（在云上构建） | 正式「在线服务」地址（贴合课件评分） | ~60–90 min | 镜像存储便宜；在线服务按 T4 计费 | ⭐⭐ 冲满分用 |
| C. 本地 Docker Desktop | 本机装 Docker Desktop → 构建 → 推 SWR | ✅ 需要 | 同 B | +安装时间 | 本机资源 | 仅当已装 Docker |

**建议**：先走 **路线 A** 保证验收当天一定能演示；有余力再走 **路线 B** 拿正式在线服务地址（评分项「部署到华为云平台」）。
两条路线都用同一个部署包，互不影响。

---

## 1. 准备工作（一次性，约 10 分钟）

### 1.1 账号与实名
1. 打开 https://www.huaweicloud.com ，登录你的账号（没有就注册，需**实名认证**才能买 ECS）。
2. 右上角「费用中心 → 优惠与折扣」看看有没有**代金券 / 学生优惠**，有就领。
3. 确认账号**不欠费**（费用中心 → 总览）。

### 1.2 选区域
买机器前先定区域，后面所有资源（ECS、SWR、ModelArts）**尽量在同一区域**，内网互通且便宜。
建议 **华北-北京四（cn-north-4）**。本手册示例统一用它。

### 1.3 费用估算（按量付费，验收当天用完就删）

| 资源 | 规格 | 价格量级 | 说明 |
|---|---|---|---|
| ECS CPU 版 | 4 vCPU / 8 GB，Ubuntu 22.04 | ≈ 0.5–1.5 元/小时 | 够跑演示（CPU 推理慢，见 §3.6 说明） |
| ECS GPU 版 | T4（如 `g6.xlarge.2` 或 `pi2.xlarge`） | ≈ 8–15 元/小时 | 需 GPU 配额，攻击速度快 |
| SWR 镜像存储 | 私有仓库 | 几毛/月 | 镜像约 6–8 GB |
| ModelArts 在线服务 | T4 单实例 | 按小时计费，较高 | **用完立即删**，否则持续扣费 |

> ⚠️ 三项最容易「忘记删」：ECS 实例、ModelArts **在线服务**、SWR 镜像。验收结束按 §6 清理。

### 1.4 生成部署包（[本机] 执行）

```powershell
cd "D:\Voice Guard"
.\.venv\Scripts\python.exe tools\make_cloud_bundle.py --zip
```

产物：`deploy\cloud_bundle\`（目录）与 `deploy\cloud_bundle.zip`（约 123 MB）。
包内含：代码 + 权重（129 MB）+ 注册库 + 每个说话人 1 句演示音频，**不含** 12 GB 全量 VCTK。

---

## 2. 路线 A：ECS 直接运行（推荐，无需 Docker）

### 2.1 购买 ECS（[控制台]）

1. 控制台搜索「弹性云服务器 ECS」→ **购买弹性云服务器**。
2. 按下面填（其余默认）：

| 字段 | 填法 |
|---|---|
| 计费模式 | **按需计费**（用完删，别选包年包月） |
| 区域 | 华北-北京四 |
| 可用区 | 随机 / 任意 |
| 规格 | CPU 演示：`c6.large.2`（2vCPU/4G）或 `c6.xlarge.2`（4vCPU/8G）<br>GPU：筛选「GPU 加速型」，选 **T4**（如 `g6.xlarge.2`），需先有配额 |
| 镜像 | **公共镜像 → Ubuntu 22.04 64bit**<br>（GPU 机器选镜像时勾选「自动安装 GPU 驱动」） |
| 系统盘 | 高 IO / 通用型 SSD，**默认 40 GB 够用**（镜像构建路线建议 60 GB） |
| 网络 | 默认 VPC；**勾选「弹性公网 IP」→ 按带宽计费 → 5 Mbit/s**（够用且便宜） |
| 安全组 | 新建一个 `voiceguard-sg`（后面再加规则） |
| 登录方式 | **密码**（自己设一个，记下来）或密钥对（要保存 .pem） |
| 云服务器名称 | `voiceguard-demo` |

3. 勾选协议 → **立即购买** → 等待 1–2 分钟 → 在「ECS 列表」看到 **运行中**，记下 **弹性公网 IP**。

### 2.2 安全组放行 7860 端口（[控制台]）

1. ECS 列表 → 点实例名 → 「安全组」页签 → 点安全组名 → 「入方向规则」→ **添加规则**：

| 字段 | 填法 |
|---|---|
| 优先级 | 1 |
| 协议端口 | **TCP**，端口填 `7860` |
| 源地址 | `0.0.0.0/0`（演示用；若只想自己看，填你本机公网 IP） |
| 描述 | gradio-web |

2. 点「确定」。也可顺便放行 `22`（SSH，一般默认已有）。

### 2.3 SSH 登录（[本机] PowerShell）

```powershell
ssh ubuntu@<弹性公网IP>
# 首次连接输入 yes，再输入你设置的密码（或指定密钥：ssh -i path\to\key.pem ubuntu@IP）
```

> 华为云 Ubuntu 镜像默认用户是 `ubuntu`（有 sudo 权限）；如果是 CentOS 镜像则用 `root`。

### 2.4 上传部署包（[本机] 另开一个 PowerShell 窗口）

```powershell
cd "D:\Voice Guard\deploy"
scp cloud_bundle.zip ubuntu@<弹性公网IP>:/home/ubuntu/
# 1 分钟以内（123 MB，5 Mbit/s 约 4 分钟；慢就先建 OBS 中转，见 §7 排错）
```

上传完成后回到 SSH 窗口：

```bash
cd /home/ubuntu
sudo apt update && sudo apt install -y unzip python3-venv
unzip -q cloud_bundle.zip -d voiceguard
ls voiceguard        # 应看到 vguard/ scripts/ frontend/ checkpoints/ data/ CLOUD_README.md ...
```

### 2.5 安装依赖（[ECS]，10–20 分钟）

```bash
cd /home/ubuntu/voiceguard
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip -i https://repo.huaweicloud.com/repository/pypi/simple/

# ① 先装 torch（GPU 机器要 CUDA 版；CPU 机器装 CPU 版）
#   GPU（需已装 NVIDIA 驱动）：
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
#   CPU（不想装驱动的演示机用这个）：
# pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# ② 其余依赖（华为云源，境内快）
pip install -i https://repo.huaweicloud.com/repository/pypi/simple/ \
  numpy scipy librosa soundfile pyyaml tqdm matplotlib \
  "gradio==4.44.1" "gradio-client==1.3.0" "fastapi==0.112.4" "starlette==0.37.2" "jinja2==3.1.2" \
  uvicorn python-multipart speechbrain "huggingface-hub<0.30"
```

> 版本必须照抄：`gradio==4.44.1` + `gradio-client==1.3.0` + `fastapi==0.112.4` + `starlette==0.37.2` + `jinja2==3.1.2`
> 是本地验证过的组合；装成最新版会出现「localhost 无法访问 / 模板渲染报错」。
> MPCONFIGDIR 等环境变量在启动命令里用 `MPLCONFIGDIR=/tmp/mpl` 临时指定即可。

### 2.6 启动服务（[ECS]）

**GPU 机器：**

```bash
MPLCONFIGDIR=/tmp/mpl nohup python frontend/app.py \
  --model-dir checkpoints/adain-vc-official \
  --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz \
  --vctk data/VCTK22 --device cuda \
  --server-name 0.0.0.0 --server-port 7860 > app.log 2>&1 &
```

**CPU 机器（演示可用，慢）：**

```bash
MPLCONFIGDIR=/tmp/mpl nohup python frontend/app.py \
  --model-dir checkpoints/adain-vc-official \
  --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz \
  --vctk data/VCTK22 --device cpu \
  --server-name 0.0.0.0 --server-port 7860 > app.log 2>&1 &
```

查看启动日志（等 20–60 秒模型加载）：

```bash
tail -f app.log
# 出现 "Running on local URL:  http://0.0.0.0:7860" 即成功；Ctrl+C 退出日志查看（服务仍在跑）
```

### 2.7 验证（[本机]）

```powershell
curl -I http://<弹性公网IP>:7860/
# 期望：HTTP/1.1 200 OK
```

浏览器打开 **http://<弹性公网IP>:7860** → 应看到 Voice Guard 界面。

自检一遍：点「💡 载入 VCTK 样例音频」→ 勾选「⚡ 演示快速模式」→ 点「🛡️ 一键保护」→ 等结果卡出现。

> CPU 机器上演示快速模式（300/200 步）约 5–10 分钟/句；GPU 约 30–60 秒/句。
> 验收时间紧的话，**提前在 GPU 上生成好案例音频**，现场只讲结果（见 §8）。

### 2.8 保持运行（可选）

- `nohup ... &` 启动的进程在 SSH 断开后**继续运行**，但**机器重启会丢**。
- 需要开机自启时（[ECS]）：

```bash
sudo tee /etc/systemd/system/voiceguard.service > /dev/null <<'EOF'
[Unit]
Description=Voice Guard Gradio
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/voiceguard
Environment=MPLCONFIGDIR=/tmp/mpl
ExecStart=/home/ubuntu/voiceguard/.venv/bin/python frontend/app.py \
  --model-dir checkpoints/adain-vc-official \
  --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz \
  --vctk data/VCTK22 --device cuda --server-name 0.0.0.0 --server-port 7860
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now voiceguard
systemctl status voiceguard --no-pager
```

---

## 3. 路线 B：镜像 + SWR + ModelArts 在线服务（正式服务地址）

> 目的：产出「在线服务地址」，对应课件评分项「部署到华为云平台」。
> 全程**在 ECS 上构建镜像**，本机不需要 Docker。

### 3.1 准备一台 ECS（可复用路线 A 的机器）

- 系统盘 **≥ 60 GB**（镜像 + 构建缓存约 20 GB）。
- 若复用 CPU 机器构建镜像没关系——**构建不需要 GPU**。
- 安装 Docker（[ECS]）：

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER    # 重新登录 SSH 生效，之后可免 sudo
docker --version
```

### 3.2 构建镜像（[ECS]，10–25 分钟）

```bash
cd /home/ubuntu/voiceguard          # 部署包解压目录（内含 deploy/Dockerfile）
sudo docker build -f deploy/Dockerfile -t voiceguard:latest .
# 构建完成后确认：
sudo docker images | grep voiceguard
```

> Dockerfile 已固定全部关键版本（gradio 4.44.1 / fastapi 0.112.4 / starlette 0.37.2 / jinja2 3.1.2 等），
> 并用华为云 pip 源加速；不要随意改版本。
> 想在云上先自测：`sudo docker run -d -p 7860:7860 voiceguard:latest`，再 `curl -I http://127.0.0.1:7860/`。

### 3.3 推送到 SWR（[控制台 + ECS]）

**控制台（[控制台]）：**
1. 搜索并进入 **SWR 容器镜像服务** → 左侧「组织管理」→ **创建组织**：
   - 组织名称：`voiceguard`（小写）
   - 若提示「共享」选私有。
2. 左侧「我的镜像」→ 右上角点 **「登录指令」/「客户端上传」**：
   - 复制生成的 **`docker login -u ... -p ... swr.cn-north-4.myhuaweicloud.com`**（临时指令有效期 24 小时）。
   - 同一弹窗里会显示**镜像地址前缀**，形如 `swr.cn-north-4.myhuaweicloud.com/voiceguard`。

**ECS（[ECS]）：**

```bash
sudo docker login -u cn-north-4@<你的AK> -p <登录指令里的密码> swr.cn-north-4.myhuaweicloud.com
sudo docker tag voiceguard:latest swr.cn-north-4.myhuaweicloud.com/voiceguard/voiceguard:latest
sudo docker push swr.cn-north-4.myhuaweicloud.com/voiceguard/voiceguard:latest
# 推送约 6–8 GB，视带宽 10–30 分钟；完成后到 SWR 控制台「我的镜像」能看到该镜像即成功
```

> 区域域名：北京四 `swr.cn-north-4.myhuaweicloud.com`；上海一 `swr.cn-east-3.myhuaweicloud.com`；
> 广州 `swr.cn-south-1.myhuaweicloud.com`。与你 ECS/SWR 所在区域保持一致。

### 3.4 创建 ModelArts AI 应用（[控制台]）

1. 搜索进入 **ModelArts**（确保左上角区域与其他资源一致，如华北-北京四）。
2. 左侧菜单 **「AI 应用管理」→「AI 应用」→ 创建 AI 应用**（部分账号在「ModelArts Studio → 应用管理」下，界面名称略有差异）。
3. 逐字段填写：

| 字段 | 填法 |
|---|---|
| 名称 | `voiceguard` |
| 应用描述 | Voice Guard 语音隐私主动防御（Gradio） |
| 元模型来源 | **容器镜像** |
| 镜像来源 | **SWR** → 组织 `voiceguard` → 镜像 `voiceguard` → 版本 `latest` |
| 容器调用接口 | **HTTP** |
| 端口/服务端口 | **7860** |
| 运行环境/规格 | **GPU：T4**（如 `modelarts.vm.gpu.t4`）；若只想演示界面也可选 CPU 规格（慢） |
| 启动命令 | 留空（用镜像自带 CMD） |
| 健康检查 | **HTTP GET `/`**，超时 60 秒（模型加载慢，务必要长） |
| 环境变量 | 可选：`MPLCONFIGDIR=/tmp/mpl` |
| 其他 | 默认 |

4. 点「立即创建」，等状态变为 **「已创建」**（解析镜像需 1–3 分钟）。

### 3.5 部署为在线服务（[控制台]）

1. AI 应用列表 → 该应用 → **「部署」→「在线服务」**。
2. 填写：

| 字段 | 填法 |
|---|---|
| 服务名称 | `voiceguard-service` |
| 资源池 | **公共资源池** |
| 规格 | **T4 GPU 单实例**（或 CPU） |
| 实例数 | **1** |
| 是否自动停止 | 建议选「**是**，1 小时后自动停止」防止忘记删 |
| 其余 | 默认 |

3. 提交后等待：**部署中 → 运行中**（通常 3–8 分钟，冷启动含镜像拉取）。
4. 「运行中」后在服务详情页复制 **调用地址 / 公网访问地址**（形如 `https://xxx.modelarts...` 或 `http://<IP>:7860`）。

### 3.6 验证

```powershell
curl -I <在线服务地址>/
# 200 即通；浏览器打开同样的地址即可看到 Gradio 界面
```

现场演示就用这个地址。若返回 5xx 或超时：

| 现象 | 原因与处理 |
|---|---|
| 部署失败 / 一直「部署中」 | 镜像太大拉取慢（6–8 GB），等待 10 分钟；仍失败看「事件/日志」 |
| 服务运行但 502/504 | 健康检查路径/端口不对：确认端口 7860、路径 `/`、超时 ≥60s |
| 打开是 JSON 报错页 | 容器内 app 没起来：查看在线服务日志（`tail app.log` 等价物），确认 CMD 参数与权重路径 |
| 页面能开但点保护报错 | 容器内 `data/VCTK22` 缺失 → 检查构建上下文是否包含 data/（部署包自带 §1.4） |

### 3.7 释放资源（重要）

1. ModelArts → 在线服务 → **停止并删除** `voiceguard-service`。
2. AI 应用 → 删除 `voiceguard`。
3. SWR → 我的镜像 → 删除镜像（可选，占空间小）。
4. ECS → 删除或关机（关机仍收少量费用，按量付费建议直接删除）。

---

## 4. 路线 C：本地 Docker Desktop（仅当已装 Docker）

```powershell
cd "D:\Voice Guard"
docker build -f deploy/Dockerfile -t voiceguard:latest .
docker run -d --gpus all -p 7860:7860 voiceguard:latest
# 本机验证 http://127.0.0.1:7860
docker tag voiceguard:latest swr.cn-north-4.myhuaweicloud.com/voiceguard/voiceguard:latest
docker login swr.cn-north-4.myhuaweicloud.com
docker push swr.cn-north-4.myhuaweicloud.com/voiceguard/voiceguard:latest
```

之后接 §3.4（ModelArts 创建 AI 应用）。
> 本机装 Docker Desktop 需要 WSL2 且占 4–8 GB 空间；没有的话用路线 A/B 在云上构建，效果相同。

---

## 5. 验收当天的检查清单

**验收前 1 天**

- [ ] 部署包已生成并上传到 ECS（`deploy/cloud_bundle.zip`）
- [ ] ECS 上服务能起来：`curl -I http://127.0.0.1:7860/` 返回 200
- [ ] 安全组放行 7860；从自己手机/另一台电脑访问 `http://<公网IP>:7860` 成功
- [ ] 用「演示快速模式」跑通一次完整流程，确认判定卡正常显示
- [ ] 准备 2–3 个**提前生成好的案例**（`outputs/exp03_eps005/p225_040/` 等目录里有 original/defended/conv 音频），
      现场若网络或算力不稳，直接播放这些音频讲解

**验收当天（提前 30 分钟）**

- [ ] ECS 开机 → `cd voiceguard && tail -f app.log` 确认服务在跑（重启过机器则重新执行 §2.6）
- [ ] 浏览器打开 `http://<公网IP>:7860`，确认页面加载
- [ ] ModelArts 在线服务（若已部署）提前「启动」，确认运行中并复制地址
- [ ] 手机热点备份：演示机自带浏览器缓存打不开时，用手机 4G 访问

**演示结束后**

- [ ] 停止 ModelArts 在线服务；ECS 关机/删除（按 §3.7）

---

## 6. 费用与资源释放清单

| 资源 | 位置 | 处理 |
|---|---|---|
| ECS 实例 | 控制台 → 弹性云服务器 | 停止或**删除**（删前先解绑弹性公网 IP） |
| 弹性公网 IP | 控制台 → 弹性公网 IP | 解绑后**释放**，否则继续计费 |
| ModelArts 在线服务 | ModelArts → 在线服务 | **停止并删除** |
| ModelArts AI 应用 | ModelArts → AI 应用 | 删除 |
| SWR 镜像 | SWR → 我的镜像 | 可保留（费用极低）或删除 |
| OBS（若用过中转） | OBS → 桶 | 删除桶内文件 |

---

## 7. 排错速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `scp` 上传很慢/中断 | 带宽 5 Mbit/s 太小 | ECS 带宽临时升到 20 Mbit/s；或先传 OBS（控制台拖拽上传），ECS 内 `obsutil cp` 下载 |
| `ssh: connect to host ... timed out` | 安全组没放行 22 / 密码错 | 安全组加 22 入方向；确认用户名 `ubuntu` |
| 页面打不开（浏览器一直转） | 服务没起 / 端口没放行 | `tail -f app.log` 看是否「Running on local URL」；检查安全组 7860 |
| `ValueError: When localhost is not accessible...` | gradio 版本链不对 | 按 §2.5 固定 `gradio 4.44.1 + gradio-client 1.3.0 + fastapi 0.112.4 + starlette 0.37.2 + jinja2 3.1.2` |
| `TemplateResponse` / `unhashable type: dict` | fastapi/starlette 太新 | 同上，`pip install "fastapi==0.112.4" "starlette==0.37.2"` |
| `CUDA out of memory` | GPU 显存不足 | 用 `--device cpu` 或换大显存实例；页面用演示模式 |
| CPU 上保护要十几分钟 | CPU 算力弱 | 用演示快速模式；或提前生成案例 |
| ModelArts 部署失败（拉镜像超时） | 镜像 6–8 GB，冷启动慢 | 等 10–15 分钟；或删镜像中不必要的文件重建（见 Dockerfile 注释） |
| `docker push` 401/403 | 登录指令过期（24h） | 重新生成登录指令再 `docker login` |
| 在线服务 5xx | 健康检查/端口/CMD 配置错 | 端口 7860、路径 `/`、超时 60s；看服务日志 |

---

## 8. 附录

### 8.1 常用命令速查（[ECS]）

```bash
tail -f app.log                       # 看服务日志
pkill -f "frontend/app.py"            # 停服务
curl -I http://127.0.0.1:7860/        # 本机健康检查
sudo docker images                    # 看本地镜像
sudo docker ps                        # 看运行中的容器
ss -lntp | grep 7860                  # 确认端口监听
```

### 8.2 镜像地址格式

```
swr.<区域域名>/<组织名>/<镜像名>:<标签>
例：swr.cn-north-4.myhuaweicloud.com/voiceguard/voiceguard:latest
```

### 8.3 交付证据（截图清单）

1. ECS 列表（运行中）与安全组 7860 规则
2. `http://<公网IP>:7860` 浏览器界面（含判定卡结果）
3. ModelArts 在线服务「运行中」页面 + 服务地址
4. 本手册与 `deploy/Dockerfile`（代码仓证据）
