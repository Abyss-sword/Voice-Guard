# Voice Guard 复现与探索

论文：*Voice Guard: Protecting Voice Privacy with Strong and Imperceptible Adversarial
Perturbation in the Time Domain*，IJCAI 2023，4812–4820。

实现时域对抗扰动、心理声学掩蔽、AdaIN-VC 转换、独立 ECAPA 身份评估和 Gradio 界面。
目标是可核验的抽样复现；数据、ASV、声码器及 MCD 定义与原论文有差异，不能宣称精确复现论文数值。

## 当前状态（2026-11-04）

- 主仓库 `master` 与审计分支 `codex/huawei-ecs-reproduction` 已合并；本文件综合两台机器的记录。
- 审计修复重采样、批量断点续跑、随机种子、筛选分母和 WER 参考文本错误，补齐模型下载与 ECS 部署工具。
- 用户尚未创建华为云实例，有 500 元代金券；尚无云端部署地址或新一轮云端实验结果。
- 本机（4060 8GB）留有实验 `outputs/` 逐句 CSV：ε 扫描与基线表可按 `docs/06_结果表.md` 复核；
  审计机快照没有这些产物，故其 README 曾标注"待重新验证"——逐句记录本机存在，见下文「主要结果」。
- 本轮验证记录见 [复现审计](docs/10_复现审计.md)，实际部署见 [华为云操作手册](docs/07_部署操作手册.md)。

## 华为云 GPU ECS

建议 Ubuntu 22.04 x86_64、NVIDIA CUDA GPU（至少 8GB，优先 16GB 显存）、至少 100GB 可用磁盘。
先确认代金券覆盖产品/区域、GPU 配额和结算价格，先做单句 benchmark，再决定批量实验范围。

上传项目后，在 ECS 的项目根目录：

```bash
bash deploy/ecs_bootstrap.sh
.venv/bin/python scripts/reproduce.py models
.venv/bin/python scripts/reproduce.py data --max-hours 2
.venv/bin/python scripts/reproduce.py registry
.venv/bin/python tools/preflight.py --scope experiment --out outputs/cloud/preflight.json
.venv/bin/python scripts/reproduce.py smoke --max-hours 1
.venv/bin/python scripts/reproduce.py benchmark --max-hours 2
bash deploy/start_ecs.sh
```

本机打开 SSH 隧道 `ssh -N -L 7860:127.0.0.1:7860 用户名@公网IP`，浏览器访问
`http://127.0.0.1:7860`。安全组只放行本人 IP 到 SSH；完整步骤与 Windows 打包上传命令见部署手册。
`--max-hours` 控制脚本耗时，不停止 ECS 计费。

Docker 是可选路线，见 [deploy/README.md](deploy/README.md)。镜像运行时挂载模型/数据/输出；
ModelArts 的 Gradio 长任务、队列和代理兼容性尚未验证，本轮优先使用 ECS。

## 环境与资源

推荐 Python 3.10–3.12。torch/torchaudio 2.5.1 同版本安装，CUDA 版与驱动需匹配：

```bash
python -m venv .venv
# Linux 下使用 .venv/bin/python；Windows 下使用 .venv/Scripts/python.exe
.venv/bin/python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r deploy/requirements.txt
.venv/bin/python -m pip check
.venv/bin/python tools/download_models.py models
.venv/bin/python tools/download_models.py vctk --extract
```

`models` 下载 NTU AdaIN-VC（自动生成 24kHz 配置）和固定提交的 SpeechBrain ECAPA；
下载记录包含 URL/文件大小/SHA256。VCTK22 压缩包约 10.69GB，按发布摘要校验，解压到 `data/VCTK22/pXXX/`。
`requirements.txt` 另含 Whisper/报告工具；部署与身份维实验用固定版本 `deploy/requirements.txt`。
Whisper/MOSNet 和完整数据需额外准备，不能因命令存在就标记评测完成。

## 分阶段实验

```bash
python scripts/reproduce.py main --dry-run
python scripts/reproduce.py main --max-hours 2
python scripts/reproduce.py eps --max-hours 2
python tools/build_tables.py \
  --exp-dirs outputs/cloud/main_vg,outputs/cloud/main_vg-nop,outputs/cloud/main_emb,outputs/cloud/main_fb,outputs/cloud/main_e2e \
  --out outputs/cloud/results.md
```

`smoke` 使用 1 句、60/30 步；`benchmark` 使用 1 句、3000/1500 步。
`main` 使用 5 人各 1 句 × 5 方法，`eps` 使用同一选句规则 × 4 档 ε；正式步数 3000/1500。
样本从注册用前 30 句之后选择，最长 2.5 秒；论文规模更大，这只是抽样协议。
每句的种子不依赖执行顺序；`manifest.json` 记录参数和权重摘要，`result.json` 及完整音频标记完成，
`results.csv/summary.json` 原子写入并支持续跑。改变参数应使用新目录。

主表使用 **ECAPA 身份维**：ASV(defended)=X 且 ASV(conv(defended))≠X，
同时报告 clean-VC 筛选通过率、筛选后成功率和未筛选成功率。空分母记 N/A。
MOS/WER 需要独立评测，不把身份维成功率当作论文联合质量指标。

## 本地验证

```bash
python tests/test_features.py
python tests/test_masking.py
python -m unittest discover -s tests -p test_reproduction.py
python tools/preflight.py --device cpu --out outputs/local-runtime.json
python tools/verify_local.py --frontend  # 需先下载 AdaIN 权重；合成输入，非防御率评测
```

没有真实权重/数据时，单元测试只能核验数值与流程，不能证明防御有效。

## 历史结果口径更正

[历史逐句表](docs/06_结果表.md) 中 p229_033 的 clean-VC 已失败，不能计入筛选后分母。
由该表重新统计：

| 方法（ε=0.05） | 未筛选 | clean-VC 通过 | 筛选后 |
|---|---:|---:|---:|
| emb | 1/5，20% | 4/5 | **0/4，0%** |
| fb | 2/5，40% | 4/5 | **1/4，25%** |
| e2e | 1/5，20% | 4/5 | **0/4，0%** |
| VG w/o P | 3/5，60% | 4/5 | **3/4，75%** |

这些是对已存 Markdown 行的重新计数，不是本轮运行结果。ε 扫描 75/75/50/50% 的逐句 CSV
本机保留（`outputs/exp03_eps*/results.csv`），可作为历史声明依据；正式复跑以修复后的管线为准。

## 目录

`vguard/` 核心算法与评估；`scripts/` 注册、保护、转换与实验；`frontend/` Gradio；
`deploy/` ECS/Docker；`tools/` 下载、预检和统计；`tests/` 数值与回归；
`third_party/` 上游参考；`docs/` 方案、历史报告、实验协议与审计。

权重/数据/输出不入源码包。组员姓名、真实云端证据、数据分发链接与更新后的 PPT 仍需补齐。

## 课程平台接入

保留仓库 `platform/` 适配服务、方法配置与三份样例音频。启动及接口说明见
[课程平台接入说明](docs/10_课程平台接入说明.md) 和 [小组 README](platform/README-group.md)。
平台 Fork 与集成**已完成**：`Abyss-sword/algorithm_research_frontend` 分支 `group_xx_submission`
（README/方法配置/索引/样例均已推送，endpoint 本地 `http://127.0.0.1:8020/process`）。
剩余：替换真实组号 → `npm test && npm run build` → 发 PR（标题 `第XX组-音频主动防御`，暂缓）。
已有的含资源打包工具 `tools/make_cloud_bundle.py` 同样保留；新 ECS 流程使用源码包并在云端下载资源。

## 本地使用（本机验证环境，历史命令仍可用）

```powershell
# 1) 构建注册库（攻击用 Es 128 维 / 评估用 ECAPA 192 维）
python scripts/build_registry.py --mode vc  --model-dir checkpoints\adain-vc-official --vctk data\VCTK22 --speakers p225,...,p246 --out outputs\registry_vc.npz
python scripts/build_registry.py --mode ecapa --vctk data\VCTK22 --speakers p225,...,p246 --out outputs\registry.npz

# 2) 单条保护（完整 3000+1500 步）
python scripts/protect.py --model-dir checkpoints\adain-vc-official --defend <x.wav> --adv-target <y.wav> --registry outputs\registry.npz --attack-registry outputs\registry_vc.npz --target-speaker p225 --eps 0.1 --out-dir outputs\case1

# 3) 批量实验（可断点续跑；--method vg|vg-nop|emb|fb|e2e）
python scripts/run_experiment.py --model-dir checkpoints\adain-vc-official --vctk data\VCTK22 --speakers p225,p226,p227,p228,p229 --per-speaker 2 --utt-offset 30 --registry outputs\registry.npz --attack-registry outputs\registry_vc.npz --source-speaker p226 --source-utt 2 --eps 0.1 --iters1 1500 --iters2 750 --out-dir outputs\exp01
```

### 🚀 本地启动交互界面（Gradio）

```powershell
cd "D:\Voice Guard"
$env:MPLCONFIGDIR="D:\Voice Guard\.mplcache"
.\.venv\Scripts\python.exe frontend\app.py `
  --model-dir checkpoints\adain-vc-official `
  --registry outputs\registry.npz --attack-registry outputs\registry_vc.npz `
  --vctk data\VCTK22 --device cuda
```

浏览器打开 **http://127.0.0.1:7860** （中英双语 / 昼夜双模；演示快速模式约 30–90 秒/句）。
端口占用加 `--server-port 7861`；无 GPU 用 `--device cpu`。

## 主要结果（本机实验记录，逐句 CSV 在 outputs/）

**ε 扫描**（5 句 × 4 档，ECAPA 口径 + clean-VC 筛选协议）：防御率 0.02→75%、0.05→75%、
0.08→50%、0.10→50%（曲线 `docs/fig_eps_curve.png`）。低 ε 平台由 F_str"顽固句"限制，
高 ε 由 F_imp 退化主导。

**方法对比**（ε=0.05，筛选后）：emb 0%（0/4）、fb 25%（1/4）、e2e 0%（0/4）、
VG w/o P 75%（3/4）、**VG 75%**（3/4）——时域 + 掩蔽显著优于频域基线。

**加分项（真实信道）**：mp3/aac 对防御几乎无影响；8k/G.711 双向分化；F_imp 是信道下
最薄弱的一环。攻击侧 EOT 使防御全面变差（G.711 上 60%→20%），为如实报告的负面结果。

## ⏳ 待完成事项（成员协作清单）

| # | 事项 | 负责人 | 说明 |
|---|---|---|---|
| 0 | 课程平台 PR | 任一成员 | 替换真实组号 → `npm test && npm run build` → 发 PR（标题 `第XX组-音频主动防御`，暂缓） |
| 1 | 华为云部署与云端实验 | 任一成员 | 500 元代金券可用；按 `docs/07_部署操作手册.md` 建 GPU ECS，跑 `scripts/reproduce.py` 云端主表/ε 扫描，补云端显存/耗时证据 |
| 2 | 组员姓名与分工 | 两位成员 | 报告 `docs/04_报告.md` §6 与 PPT 封面（现为「成员 A/B」占位） |
| 3 | 权重/数据分发链接 | 任一成员 | `checkpoints/` 与注册库打包上传云盘，链接替换部署手册中的占位 |
| 4 | 验收 PPT 更新 | 任一成员 | 以修复后的管线重跑结果更新 `docs/验收PPT_v2.pptx` 数据页 |
| 5 | 最终校对 | 任一成员 | 报告/README 过一遍（数字与 `docs/06_结果表.md` 一致） |

## 状态

- [x] 复现审计分支合并：重采样/续跑/种子/筛选分母/WER 参考等修复 + ECS 工具 + 预检/回归测试
- [x] 本机实验：ε 扫描（75/75/50/50）、基线对比（筛选后 emb 0/fb 25/e2e 0/vg-nop 75/VG 75）、严格指标（MCD/WER/MOS）、信道分析 + EOT 对照
- [x] 系统：Gradio 前端（双语/昼夜）+ 课程平台适配服务 + Docker/ECS 部署物料
- [~] 待用户侧：真实组号、组员姓名、云端部署与云端实验、云盘链接、PPT 最终更新
