# Voice Guard 复现与探索

论文：*Voice Guard: Protecting Voice Privacy with Strong and Imperceptible Adversarial
Perturbation in the Time Domain*，IJCAI 2023，4812–4820。

实现时域对抗扰动、心理声学掩蔽、AdaIN-VC 转换、独立 ECAPA 身份评估和 Gradio 界面。
目标是可核验的抽样复现；数据、ASV、声码器及 MCD 定义与原论文有差异，不能宣称精确复现论文数值。

## 当前状态（2026-11-09）

第三轮原生参考与两阶段实验已完成：在开发集比较嵌入、反馈、解码频谱和软掩蔽 alpha 搜索后，冻结唯一配置，再完成 20 个新句、10 位注册人物、2 个内容源的同句配对验证。独立 GE2E/small.en 严格联合由基线 **9/31（29.0%）提高至 15/31（48.4%）**；完整干净控制下的 Eq.10 均为 **2/6（33.3%）**。保护原音预测 MOS 为 3.990、WER 为 1.27%。按说话人计算的改进区间跨过零，尚未建立稳定改进或原文总体质量复现，详见 [最新实验报告](docs/15_两阶段与原生参考修正.md) 与 [7 页 PDF](deliverables/Voice_Guard_原生参考与两阶段报告.pdf)。

已修复常规 VC 注册构建器的静音上下文不一致，现有 39 项测试全部通过，新增归档包含可核验的 302 个 WAV。原生参考修正本身不算防护提升；ECAPA/base 只筛身份的 81.5% 属次评估，完整控制仅 1/2，不能代替独立主结果。实际执行快照、协议、转录和哈希收据见 [第三轮证据](docs/evidence/2026-11-09-staged/README.md)，原有注册库、音频与负结果保留。

第二轮质量实验已完成：冻结 15 句、两个内容源，保护原音原始 MOSNet 预测均值 4.063、Whisper base 逐句 WER 4.61%；严格身份联合 8/21，Eq.10 结构 16/21，完整干净控制下 4/7。独立 GE2E/small.en 复核分别为 11/27、16/27、5/12，不能宣称达到论文 80.06% 的总体复现。旧结果保留，详见 [补充实验](docs/14_论文质量追赶.md) 与 [补充报告 PDF](deliverables/Voice_Guard_论文质量补充报告.pdf)。

- 主仓库 `master` 已三次合并审计分支（第三次：云端战役 + 论文质量追赶 + 两阶段原生参考）；本文件综合两台机器记录。
- 已修复重采样、批量断点续跑、随机种子、筛选分母、WER 参考文本，以及平台上传采样率、缺失 ASV 判定与请求串行处理；补齐模型下载与 ECS 部署工具。
- **华为云 T4 已部署并验证**：Ubuntu 22.04、Tesla T4（15GB 可用显存）、32GB 内存、100GB 系统盘；
  双 systemd 服务 `voiceguard`（127.0.0.1:7860）与 `voiceguard-platform`（8020）+ SSH 隧道；
  真实课程浏览器（Vite 5173）、FastAPI 与 Gradio 完整上下文请求均通过。
- 本机（4060 8GB）留有实验 `outputs/` 逐句 CSV：ε 扫描与基线表可按 `docs/06_结果表.md` 复核；
  历史表/旧 PPT 仅作历史材料。新报告与图表使用云端归档实测数据。
- 本轮验证记录见 [复现审计](docs/10_复现审计.md)，实际部署见 [华为云操作手册](docs/07_部署操作手册.md)。
- 11 月 9 日已完成全部 32 个冻结阶段：原主表/ε 的 9 组各 5 句、全部 MOS/WER、新 10 句、80 次信道和 60 次内容源配对；追加完整上下文与 ASR 重复性诊断也已完成。服务的双声道 API、6 文件下载、真实课程浏览器和 Gradio 完整上下文请求通过，见 [华为云实测](docs/11_华为云实测.md)。
- 新增投影后优化实验：同 5 句平均 MOS 1.397→3.192、平均逐句 WER 62.76%→5.71%；PCM16 导出后有 2 个样例同时通过质量与身份防护门槛。固定参数验证仅 1/2（筛选后），尚不能宣称达到论文质量，见 [质量改进实验](docs/12_质量改进实验.md)。
- 冻结新句中，原法/改进 MOSNet 为 1.936/3.055、WER 为 44.60%/14.69%，两者联合通过均为 0/8。完整上下文使改进 WER 降至 3.10%，但联合仍为 0/7。原 VG 首批联合 0/4；改进首批通过 2/4，且部分压缩/内容源下失败，不能宣称普遍有效。

报告正文：[可编辑 Markdown](docs/04_报告.md)；[PDF 报告](deliverables/Voice_Guard_课程报告.pdf)。
汇报：[14 页 / 9 分钟 PPT](deliverables/Voice_Guard_9分钟汇报.pptx)、[讲稿](docs/05_验收PPT大纲.md)。历史 PPT 已移至 `docs/history/`。
技术实验与材料已补齐；真实组号/成员、权重云盘链接、正式平台 PR、问卷和课程提交仍待本人完成。课程要求与加分对应见
[交付清单](docs/13_课程要求与交付清单.md)、[验收自查](docs/08_验收自查.md)。
本篇按用户确认的非开源选题范围验收；无需再增加两篇论文。

## 冻结的扩展实验

ECS 开机、同步本轮代码并检查 CUDA/资源后运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python deploy/run_course_campaign.py --max-hours 3
```

依次续跑原主表/ε、补齐全部 MOS/WER、10 个新句的原法/固定改进对照、
8 种配对编码/重采样信道、3 种内容源敏感性。协议与输入摘要事前冻结于
`docs/evidence/2026-11-09/frozen-protocol.json`，输出 `outputs/course/2026-11-09/`。
逐句续跑，并保留已完整的原表 MOS/WER（在阶段记录中写分数文件 SHA256），避免重新采样解码覆盖已有观测。
修改参数/输入/评估代码使用新目录；任务时限不会关机或停止实例计费。
余额约 470 元，首批最多 3 小时，按旧单价估计计算费约 22 元，磁盘/EIP 另计。

下载云端原目录至 `outputs/course/remote/` 后，重建报告：

```bash
python tools/build_course_figures.py
python tools/analyze_course_extensions.py
python tools/build_course_report.py
# 需要 reportlab、pypdfium2、Pillow 与支持中文的字体
python tools/render_course_pdf.py --font /path/to/simsun.ttc
```

PPT 源码 `tools/build_course_slides.mjs` 使用 `@oai/artifact-tool`；讲稿含来源与时长。
CPU 单元测试及前端 schema/构建不替代真实 GPU API 与质量实验。
完整归档已下载并校验，含 1559 份证据文件和 70 份云端源码快照，700 个实验 WAV 逐项匹配记录；
收据为 `docs/evidence/2026-11-09/final-archive.receipt.json`。
事后诊断可通过 `deploy/run_context_diagnostic.py` 和 `scripts/audit_asr_repeatability.py` 复查；
它们不替代原冻结结果。默认 Whisper 温度回退存在解码随机性，报告单列固定音频控制。

## 华为云 GPU ECS

建议 Ubuntu 22.04 x86_64、NVIDIA CUDA GPU（至少 8GB，优先 16GB 显存）、至少 100GB 可用磁盘。
先确认代金券覆盖产品/区域、GPU 配额和结算价格，先做单句 benchmark，再决定批量实验范围。

上传项目后，在 ECS 的项目根目录：

```bash
bash deploy/ecs_bootstrap.sh
.venv/bin/python scripts/reproduce.py models
.venv/bin/python scripts/reproduce.py data --max-hours 2
# 也可只取本项目抽样协议需要的原始 WAV，节省下载流量：
# .venv/bin/python scripts/reproduce.py data-subset --max-hours 2
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
`data-subset` 通过 HTTP Range 提取同一 ZIP 中的原始 WAV：5 位测试人各 80 句，其余 15 人各 31 句。
保留完整语料排序后的前 30 句注册集；逐文件校验 ZIP CRC32 并记录 SHA256，**不声称验证完整压缩包摘要**。
`requirements.txt` 另含 Whisper/报告工具；部署与身份维实验用固定版本 `deploy/requirements.txt`。
Whisper/MOSNet 和完整数据需额外准备，不能因命令存在就标记评测完成。
可选质量评测依赖见 `deploy/evaluation-requirements.txt`，MOS 模型下载命令为
`python tools/download_models.py mosnet`；该第三方 MOSNet 移植的分数不等同于人工 MOS。

## 权重分发（✅ 已发布：GitHub Release + 三个云盘）

代码仓库不含权重/数据（GitHub 单文件 100 MB 限制 + 仓库体积要求）。权重包（119.7 MB）
提供 **4 个下载渠道，任选其一**：

**① GitHub Release（推荐，免登录提取码，国内外均可）**

```powershell
# Release 页面：https://github.com/Abyss-sword/Voice-Guard/releases/tag/v1.0-weights
curl.exe -L -C - -o voiceguard_weights_v1.0.zip `
  "https://github.com/Abyss-sword/Voice-Guard/releases/download/v1.0-weights/voiceguard_weights_v1.0.zip"
```

**② 云盘（百度 / 迅雷 / 夸克，国内下载更快）**

| 云盘 | 链接 | 提取码 |
|---|---|---|
| 百度网盘 | https://pan.baidu.com/s/1sc4xGtKd8gpPJPzhxZzO8Q?pwd=hce6 | `hce6` |
| 迅雷云盘 | https://pan.xunlei.com/s/VP5ae6D82ZBPuGJxz5JyEQqpA1?pwd=ptb5 | `ptb5` |
| 夸克网盘 | https://pan.quark.cn/s/a7419932b2f8 | `yjQE0` |

**文件**：`voiceguard_weights_v1.0.zip`（119.7 MB，解压后 128.8 MB）

**SHA256 校验**：`065d4e89f6b098902fa3395054ec28b5eb953956caaa3d7fdfbee9d2a1874bbb`

```powershell
certutil -hashfile voiceguard_weights_v1.0.zip SHA256   # Windows
sha256sum voiceguard_weights_v1.0.zip                   # Linux/macOS
```

**包内容**（21 个条目）：
- `weights/`（14 个文件）：AdaIN-VC 官方权重 + 配置、ECAPA-TDNN（+hyperparams/label encoder）、MOSNet（safetensors）
- `registry/`（3 个文件）：`registry.npz`（ECAPA 192 维）、`registry_vc.npz`（Es 128 维）、`asv_probe.npz`
- `samples/`（3 个文件）：平台接入样例音频
- `README-weights.txt`：放置说明

**放置方式**：解压后 `weights/` 内容放进仓库 `checkpoints/`，`registry/` 内容放进 `outputs/`，
见包内 `README-weights.txt`。

**重新打包**（权重或注册库更新时）：

```powershell
.\.venv\Scripts\python.exe tools\make_weights_package.py --version v1.1
# 新 zip 上传云盘后可替换旧文件，分享链接通常保持不变
```

**数据（VCTK22，10.69 GB）不入包**：用脚本下载（带校验）
`.venv\Scripts\python.exe tools\download_models.py vctk --extract`；
或向本组索取演示子集（每说话人 1 句，约 4 MB）。

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

这些是对已存 Markdown 行的重新计数，不是本轮运行结果。历史 VG ε 扫描的逐句 CSV
本机保留（`outputs/exp03_eps*/results.csv`），可作历史声明依据；本轮云端重跑的身份维为
3/4、2/4、2/4、2/4，四档联合质量通过均为 0/4，详见当前报告。

## 目录

`vguard/` 核心算法与评估；`scripts/` 注册、保护、转换与实验；`frontend/` Gradio；
`deploy/` ECS/Docker；`tools/` 下载、预检和统计；`tests/` 数值与回归；
`third_party/` 上游参考；`docs/` 方案、历史报告、实验协议与审计。

权重/数据/大批输出不入源码包。真实云端证据已归档；成员、权重云盘链接与正式课程提交待补齐。

## 课程平台接入

本轮更新 `platform/` 适配服务（含启动时任务目录清理 `--keep-jobs`）、1.1.0 方法配置与三份有来源摘要的真实音频。
启动及接口说明见 [课程平台接入说明](docs/10_课程平台接入说明.md) 和
[小组 README](platform/README-group.md)。平台 Fork 已存在（`Abyss-sword/algorithm_research_frontend`，
分支 `group_xx_submission` 已推送方法配置/索引/样例/小组 README，endpoint 指向 `http://127.0.0.1:8020/process`），
本地 6 项测试、配置校验与生产构建通过；可选参考输入导致样例按钮拒绝执行的问题有补丁
`platform/patches/optional-input-sample.patch`。真实组号、成员、云盘链接与正式 PR 仍待补齐。

完整模型分发包位于忽略目录 `outputs/course/resources/VoiceGuard-model-resources-complete.zip`（253,422,166 字节），
含 AdaIN-VC、ECAPA、MOSNet、Whisper base、来源、逐文件摘要与恢复说明；不含数据、注册库及运行环境。
本机已核验压缩包和全部成员摘要，SHA256 为
`bf2254d9413028820bba5ddb7e8170cf02abe1d820d287d0e049847f00d9e208`，记录见
`docs/evidence/2026-11-09/model-resources-complete.receipt.json`。填写与演示流程见 [提交信息模板](deliverables/提交信息模板.md)。
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

## 主要结果（分本地 / 云端 / 质量改进三组）

**① 本机（4060 8GB）抽样实验**（逐句 CSV 在 `outputs/`）：
- ε 扫描（5 句 × 4 档，clean-VC 筛选）：防御率 0.02→75%、0.05→75%、0.08→50%、0.10→50%
  （曲线 `docs/fig_eps_curve.png`）；低 ε 平台受 F_str"顽固句"限制，高 ε 由 F_imp 退化主导。
- 方法对比（ε=0.05，筛选后）：emb 0%（0/4）、fb 25%（1/4）、e2e 0%（0/4）、
  VG w/o P 75%（3/4）、**VG 75%**（3/4）——时域 + 掩蔽显著优于频域基线。
- 加分项（真实信道）：mp3/aac 几乎无影响；8k/G.711 双向分化；F_imp 是最薄弱一环；
  攻击侧 EOT 使防御全面变差（G.711 上 60%→20%），为如实报告的负面结果。

**② 华为云 T4 实测**（`docs/11_华为云实测.md`，证据 `docs/evidence/2026-11-04/`）：
- 部署：systemd + SSH 隧道 HTTP 200，真实 `/on_protect` 调用通过。
- 单句 benchmark（3000/1500 步）：194.67s/句、CUDA 峰值 175.8 MiB；身份维通过
  （p225→p225，转换后 p234），但 MOS 3.33→1.02、WER 0→40%——未复现"强且不可感知"。

**③ 质量改进（refine，探索）**（`docs/12_质量改进实验.md`）：掩蔽投影后继续优化，
平均 MOS 1.397→3.192、平均逐句 WER 62.76%→5.71%；固定参数验证联合门槛 1/2（筛选后），
小样本、非严格不可感知。

## ☁️ 云端待办（需登录华为云 ECS 执行）

| # | 事项 | 命令/说明 | 预计耗时 |
|---|---|---|---|
| C1 | **云端完整主表**（5 方法 × 5 句） | `.venv/bin/python scripts/reproduce.py main --max-hours 3` | 25 句 × 3.2min ≈ 1.3h |
| C2 | **云端 ε 扫描**（4 档 × 5 句） | `.venv/bin/python scripts/reproduce.py eps --max-hours 3` | 20 句 × 3.2min ≈ 1.1h |
| C3 | 多种子方差（每句 2–3 seed） | `run_ecs_campaign.sh` 或 reproduce.py 换 seed | 视规模 |
| C4 | refine 独立说话人验证 | `scripts/refine_masked.py` + `quality_sweep.py` 在未调参说话人上 | ~0.5h |
| C5 | 人工听测（5–10 人 ABX） | 用云端产出的 defended/convert 音频 | 离线 |

> 说明：C1/C2 是最关键的科学证据缺口（现只有本地 4–5 句 + 云端单句 benchmark）；
> 跑完把 `outputs/cloud/` 的结果 CSV/JSON 回传本仓库 `docs/evidence/` 或发我，
> 我负责合并进报告/结果表/PPT。云端脚本 `--max-hours` 只限制脚本运行，**不停止 ECS 计费**，
> 跑完记得按 docs/07 §6 留证并释放资源。

## ⏳ 待完成事项（成员协作清单）

| # | 事项 | 负责人 | 说明 |
|---|---|---|---|
| 0 | 课程平台 PR | 任一成员 | 替换真实组号 → `npm test && npm run build` → 发 PR（标题 `第XX组-音频主动防御`，暂缓） |
| 1 | 华为云部署与云端实验 | 任一成员 | ✅ **已部署并实测**：Tesla T4 ECS（15GB/32GB/100GB），systemd 服务 + SSH 隧道 HTTP 200，真实 `/on_protect` 通过；benchmark 3000/1500 步 194.67s/句、CUDA 峰值 175.8 MiB——证据见 `docs/11_华为云实测.md` 与 `docs/evidence/2026-11-04/`。剩余：云端主表/ε 扫描完整批次（后台任务） |
| 2 | 组员姓名与分工 | 两位成员 | 报告 `docs/04_报告.md` §6 与 PPT 封面（现为「成员 A/B」占位） |
| 3 | 权重/数据分发链接 | 任一成员 | ✅ **已完成**：权重包（119.7 MB）已上传百度/迅雷/夸克三个云盘，链接与 SHA256 见「权重分发」章节 |
| 4 | 验收 PPT 更新 | 任一成员 | 以修复后的管线重跑结果更新 `docs/验收PPT_v2.pptx` 数据页 |
| 5 | 最终校对 | 任一成员 | 报告/README 过一遍（数字与 `docs/06_结果表.md` 一致） |

## 状态

- [x] 复现审计分支合并：重采样/续跑/种子/筛选分母/WER 参考等修复 + ECS 工具 + 预检/回归测试
- [x] 本机实验：ε 扫描（75/75/50/50）、基线对比（筛选后 emb 0/fb 25/e2e 0/vg-nop 75/VG 75）、严格指标（MCD/WER/MOS）、信道分析 + EOT 对照
- [x] 系统：Gradio 前端（双语/昼夜）+ 课程平台适配服务 + Docker/ECS 部署物料
- [x] 权重包（119.7 MB）已上传百度/迅雷/夸克云盘 + GitHub Release v1.0-weights（含 SHA256）
- [x] 华为云 T4 实机部署与单句实测（systemd 服务、API 调用、benchmark 耗时/显存，证据入库）
- [x] 质量改进扩展（掩蔽投影后继续优化 refine_masked）：平均 MOS 1.397→3.192、WER 62.76%→5.71%
- [~] 待用户侧：真实组号、组员姓名、云端完整批次（主表/ε 扫描）、PPT 最终更新
