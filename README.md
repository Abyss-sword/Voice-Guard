# Voice Guard 复现与探索

> 论文：*Voice Guard: Protecting Voice Privacy with Strong and Imperceptible
> Adversarial Perturbation in the Time Domain*（IJCAI 2023, pp. 4812–4820）
>
> 选题 A036 / Audio / 音频主动防御。方案见 `docs/01_方案思路.md`，环境与进度见
> `docs/02_环境与进度.md`，实验协议见 `docs/03_实验计划.md`，报告见 `docs/04_报告.md`。

## 是什么

复现 Voice Guard 的**时域对抗扰动 + 心理声学掩蔽**方法：给目标说话人的音频注入人耳
不可感知的扰动（约束在掩蔽阈值 θx 之下），使零样本语音转换（AdaIN-VC）无法再伪造其
声纹，实现语音隐私的主动防御。含可运行系统（Gradio 前端 + Docker/华为云部署）。

## 目录结构

```
vguard/        核心库（features/masking/attack/baselines/channel/speaker/asv/vc/metrics/pipeline）
scripts/       protect / convert / run_experiment / reevaluate / build_registry / build_asv_probe
               + compute_wer / compute_mos / channel_sweep / channel_sim / run_eps_sweep / run_post_sweep / make_figures
frontend/      Gradio 前端
deploy/        Dockerfile + 华为云部署说明（docs/07_部署操作手册.md）
tests/         数值对齐与单元测试
tools/         curl 下载器 / 依赖解析安装器 / 官方对照脚本 / 图表与表格生成器
docs/          方案 / 进度 / 实验计划 / 报告 / 结果表 / PPT 大纲与草稿
third_party/   上游源码参考（attack-vc / Voice-Guard / AdaIN-VC 官方）
checkpoints/   权重（git 忽略，云盘分发）
data/          VCTK 语料（git 忽略）
outputs/       攻击产物与实验表（git 忽略）
```

## 环境搭建（Windows, CUDA）

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# 1) CUDA 版 torch（wheel 已备在 refs/wheels，也可直装）
.\.venv\Scripts\python.exe -m pip install --no-index "refs\wheels\torch-2.5.1+cu124-cp312-cp312-win_amd64.whl" "refs\wheels\torchaudio-2.5.1+cu124-cp312-cp312-win_amd64.whl"
# 2) 其余依赖（网络受限时用 tools/resolve_and_install.py 的 curl+--no-index 方案）
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> 本机网络特殊（pip 间歇性卡死、Google 不可达），所有大文件均用 curl 断点续传下载，
> 依赖安装走「curl 抓 wheel → pip --no-index」。见 `docs/02_环境与进度.md`。

## 权重与数据下载

> ⚠️ **验收前请把 `checkpoints/`（约 35 MB）与注册库 `outputs/registry*.npz` 打包上传云盘，
> 并把分享链接替换到下面两行**（课件要求权重数据经云盘分发，GitHub 不入库）。

```powershell
# 云盘分发（待填）：百度网盘/阿里云盘分享链接
# 链接1：checkpoints 打包  -> <待填>
# 链接2：VCTK22 数据（可选，约 10.7 GB） -> <待填>

# AdaIN-VC 官方权重（NTU 直链，curl 可续传；云盘链接失效时用）
curl -L -C - -o checkpoints\adain-vc-official\vctk_model.ckpt "http://speech.ee.ntu.edu.tw/~jjery2243542/resource/model/is19/vctk_model.ckpt"
curl -L -C - -o checkpoints\adain-vc-official\attr.pkl "http://speech.ee.ntu.edu.tw/~jjery2243542/resource/model/is19/attr.pkl"
# config.yaml 已随仓库提供

# VCTK 22kHz（hf-mirror，约 10.7 GB）
curl -L -C - -o data\VCTK_22kHz_raw.zip "https://hf-mirror.com/datasets/arampacha/vctk_22kHz/resolve/main/raw_data.zip"
tar -xf data\VCTK_22kHz_raw.zip -C data\VCTK22

# ECAPA-TDNN（本地加载用 5 个文件）
# hyperparams.yaml 已随仓库改好本地路径；ckpt 文件从 hf-mirror 下载到 checkpoints\ecapa-local\

# MOSNet（HF 移植版，已随 tools/download_models.py 支持 curl 下载到 checkpoints\mosnet-hf\）
# Whisper base.pt（WER 评测用）
# 放到 C:\Users\<you>\.cache\whisper\base.pt（或首次使用自动下载）
```

## 自检

```powershell
.\.venv\Scripts\python.exe -m tests.test_features    # H 数值对齐（ALL PASS）
.\.venv\Scripts\python.exe -m tests.test_masking     # θx / Transform（ALL PASS）
```

## 使用

```powershell
# 1) 构建注册库（攻击用 Es 128 维 / 评估用 ECAPA 192 维）
python scripts/build_registry.py --mode vc  --model-dir checkpoints\adain-vc-official --vctk data\VCTK22 --speakers p225,...,p246 --out outputs\registry_vc.npz
python scripts/build_registry.py --mode ecapa --vctk data\VCTK22 --speakers p225,...,p246 --out outputs\registry.npz

# 2) 训练闭集 ASV 探头（Es 嵌入 + 线性分类头，对应论文 GE2E）
python scripts/build_asv_probe.py --model-dir checkpoints\adain-vc-official --vctk data\VCTK22 --speakers p225,...,p246 --out outputs\asv_probe.npz

# 3) 单条保护（完整 3000+1500 步）
python scripts/protect.py --model-dir checkpoints\adain-vc-official --defend <x.wav> --adv-target <y.wav> --registry outputs\registry.npz --attack-registry outputs\registry_vc.npz --target-speaker p225 --eps 0.1 --out-dir outputs\case1

# 4) 批量实验（可断点续跑；--method vg|vg-nop|emb|fb|e2e）
python scripts/run_experiment.py --model-dir checkpoints\adain-vc-official --vctk data\VCTK22 --speakers p225,p226,p227,p228,p229 --per-speaker 2 --utt-offset 30 --registry outputs\registry.npz --attack-registry outputs\registry_vc.npz --source-speaker p226 --source-utt 2 --eps 0.1 --iters1 1500 --iters2 750 --out-dir outputs\exp01

# 5) 离线重评（换 ASV 不必重跑攻击）
python scripts/reevaluate.py --exp-dir outputs\exp01 --model-dir checkpoints\adain-vc-official --registry outputs\registry.npz --probe outputs\asv_probe.npz --out outputs\exp01\results_v2.csv

# 6) 前端
python frontend\app.py --model-dir checkpoints\adain-vc-official --registry outputs\registry.npz --attack-registry outputs\registry_vc.npz --vctk data\VCTK22
```

## 部署（华为云）

`deploy/Dockerfile`（pytorch CUDA 基础镜像 + 华为云 pip 源）→ 推到 SWR → ModelArts
自定义镜像在线服务（端口 7860，GPU 规格）。详见 `deploy/README.md`。

## 主要结果（抽样复现，详见 docs/04_报告.md 与 docs/06_结果表.md）

**ε 扫描**（5 句 × 4 档，ECAPA 口径 + 论文 clean-VC 筛选协议）：
防御率 0.02→75%、0.05→75%、0.08→50%、0.10→50%（曲线 `docs/fig_eps_curve.png`）。
低 ε 平台由 F_str"顽固句"限制，高 ε 由 F_imp 退化主导。

**方法对比**（ε=0.05）：emb 20%、fb 40%、e2e 20%、VG w/o P（消融）、
**VG 75%** —— 时域 + 掩蔽显著优于频域基线。

**加分项（真实信道）**：mp3/aac 对防御几乎无影响；8k/G.711 双向分化；
F_imp 是信道下最薄弱的一环。攻击侧 EOT 使防御全面变差（G.711 上 60%→20%），
为如实报告的负面结果（机理 + 正确改进方向见报告 §5）。

## ⏳ 待完成事项（成员协作清单）

| # | 事项 | 负责人 | 说明 |
|---|---|---|---|
| 1 | 华为云 ModelArts 部署 | 任一成员 | 按 `docs/07_部署操作手册.md` 执行（SWR→AI应用→在线服务，端口 7860）；完成后把**服务地址**贴到这里并同步报告 §3.1 |
| 2 | 组员姓名与分工 | 两位成员 | 报告 `docs/04_报告.md` §6 与 PPT 封面（现为「成员 A/B」占位） |
| 3 | 权重云盘分发 | 任一成员 | `checkpoints/`（约 35MB）+ `outputs/registry*.npz` 打包上传云盘，链接替换下方「<待填>」两处 |
| 4 | 验收 PPT 演练 | 两位成员 | 9 分钟（大纲 `docs/05_验收PPT大纲.md`、预演问答 `docs/09_答辩预演.md`）；PPT 文件 `docs/验收PPT_草稿.pptx` |
| 5 | 最终校对 | 任一成员 | 报告/README 过一遍（数字与 `docs/06_结果表.md` 一致）；更新本表状态 |

**进度速览**（W1–W6 全部完成，见下方「状态」节）：实验、系统、部署物料均已就绪并同步 GitHub；
当前卡点仅在①②③三项「需本人操作」的外部动作。

## 状态

- [x] W1：环境 / 权重 / H 对齐 / θx / 两阶段攻击 / 双注册库 / 端到端冒烟
- [x] W2：批量实验 / Es 探头 ASV / ε 扫描（75/75/50/50）/ 基线对比
- [x] W3：严格指标（MCD / WER-Whisper / MOS-MOSNet）全部落地并跑通
- [x] W5：Gradio 前端构建测试通过 + Dockerfile + 华为云操作手册（云端实例待部署）
- [x] W6：加分项（信道失效分析 + EOT 对照，含负面结果与机理）
- [~] 报告（约 80%，缺基线终值与分工签名）+ PPT 草稿（数据已填）+ GitHub 持续推送
