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
汇报：**课程用** [14 页 / 9 分钟 PPT](deliverables/Voice_Guard_9分钟汇报.pptx) + [讲稿](docs/05_验收PPT大纲.md)；
**技术备用**（8 页，含最新云端/两阶段数据）[docs/验收PPT_v2.pptx](docs/验收PPT_v2.pptx)，由 `tools/make_ppt_v2.py` 生成。
旧版 PPT 已移至 `docs/history/验收PPT_v2_历史未复核.pptx`。
技术实验与材料已补齐；真实组号/成员、正式平台 PR、问卷和课程提交仍待本人完成。课程要求与加分对应见
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

## 主要结果（云端归档数据为准；本地数据为历史对照）

**① 云端主表（已完成，9 组 × 5 句，ECAPA 身份维 + 本项目联合门槛 MOS≥3、WER≤10%）**
- 原 VG 与无掩蔽消融：身份维 2/4 通过，但**联合质量 0/4**；emb、e2e 各联合 1/4，fb 0/4。
- 原 VG 五句 MOSNet 均值 1.397、WER 62.76%——身份翻转常伴随严重质量损失。
- 四档 ε 扫描同样**无联合通过**（身份维 3/4、2/4、2/4、2/4）：单纯调 ε 无法解决质量与防护的矛盾。

**② 质量改进与两阶段原生参考（冻结协议，独立 GE2E/small.en 评估）**
- 掩蔽投影后继续优化（refine）：首批五句 MOS 1.397→3.192、逐句 WER 62.76%→5.71%。
- 新冻结十句：原法/改进 MOSNet 1.936/3.055、WER 44.60%/14.69%，两者联合均为 0/8；
  完整上下文诊断改进 WER 3.10% 但联合仍 0/7（负结果保留）。
- **修复注册上下文不一致**（原生参考构建）后同句配对：GE2E 严格联合 **基线 9/31 → 本轮 15/31（48.4%）**；
  完整干净控制下均 2/6（33.3%）；保护原音 MOS 3.990、WER 1.27%——仍不能宣称达到论文总体质量。
- 原始 MOSNet（作者公开实现）重评：干净 3.96 / 原 VG 2.55 / 改进 4.41（评估器修正，非算法提升）。

**③ 真实信道与内容源（云端 80 次信道、60 次内容源配对）**：mp3/aac 对防御几乎无影响；
8k/G.711 双向分化；F_imp 是最薄弱一环；攻击侧 EOT 使防御全面变差（G.711 上 60%→20%，负面结果）。

**④ 本机（4060 8GB）历史对照**（逐句 CSV 在 `outputs/`，仅相对比较）：ε 扫描 75/75/50/50%；
方法对比（ε=0.05 筛选后）emb 0/4、fb 1/4、e2e 0/4、VG w/o P 3/4、VG 3/4。旧报告与旧 PPT 仅作历史材料。

## ☁️ 云端实验状态（已完成 32 个冻结阶段）

- C1 云端主表（9 组 × 5 句）+ C2 ε 扫描（4 档 × 5 句）：✅ 完成，MOS/WER 齐全，`campaign-summary.complete=true`；
- 新 10 句冻结验证、80 次信道配对、60 次内容源配对、完整上下文诊断、ASR 解码重复性诊断：✅ 完成；
- 服务验证：双 systemd 服务 + 课程浏览器（Vite 5173）+ FastAPI + Gradio 全链路通过；
- **剩余云端/离线工作**（可选增强，不影响当前结论）：
  - C3 多种子方差（每句 2–3 seed，报告均值±std）；
  - C5 人工听测（5–10 人 ABX，校核 MOSNet 预测偏差）；
  - 云端归档回传：`outputs/course/`、`outputs/paper_quality/`、`outputs/staged_recovery/` 三个 tar.gz 已在本机核验
    （SHA256 见 `docs/11_华为云实测.md` 与 `deliverables/README.md`）。
- ⚠️ **ECS 仍开机计费**：实验已完成，请到控制台关机或释放（数据已归档）。

## ⏳ 待完成事项（成员协作清单）

| # | 事项 | 负责人 | 说明 |
|---|---|---|---|
| 0 | 课程平台 PR | 任一成员 | 替换真实组号 → 把更新的 `platform/` 材料推到 Fork → `npm test && npm run build` → 发 PR（标题 `第XX组-音频主动防御`） |
| 1 | 华为云部署与云端实验 | 任一成员 | ✅ **已完成**：T4 实机部署 + 32 个冻结阶段（主表/ε/新句/信道/内容源/MOS/WER）+ 交付物；证据见 `docs/11_华为云实测.md`、`docs/evidence/2026-11-09*/` |
| 2 | 组员姓名与分工 | 两位成员 | 报告 §6、`deliverables/提交信息模板.md`、PPT 封面（现为占位） |
| 3 | 权重/数据分发链接 | 任一成员 | ✅ **已完成**：GitHub Release `v1.0-weights` + 百度/迅雷/夸克三网盘（含 SHA256） |
| 4 | 汇报材料 | 两位成员 | ✅ 成型：`deliverables/Voice_Guard_9分钟汇报.pptx`（14 页）+ 讲稿 `docs/05_验收PPT大纲.md`；技术备用 `docs/验收PPT_v2.pptx` |
| 5 | 最终提交 | 配合本人 | 填 `deliverables/提交信息模板.md`（组号/成员/云盘链接）、问卷、课程提交 |
| 6 | 最终校对 | 任一成员 | 报告/README/PPT 数字一致；`docs/08_验收自查.md` 逐项核对 |

## 🧭 可改进清单（本地 / 云端 / 需真人，按性价比排序）

> 当前主线（方法复现、云端 32 阶段实验、系统与交付物）已完成；以下是**可选增强**，
> 根据剩余时间与算力选择执行。每项完成后更新本表状态。

### 本地可做（本机 4060 8GB，无云资源依赖）

| # | 事项 | 内容 | 预计 | 状态 |
|---|---|---|---|---|
| L1 | 防御侧 EOT（真正的改进） | 已实现 `--imp-eot`（阶段2 加"信道化后身份保持"正则）+ 驱动 `scripts/run_defense_eot.py`；试点 n=3 无明确改善（6 信道均值 0.667→0.556，方差大），结论如实记录，云端大样本复验见 C 系列（[docs/16](docs/16_严格掩蔽与防御侧EOT.md)） | 2–3h | ✅ 试点完成（假设待验证） |
| L2 | 硬掩蔽投影（严格不可感知） | 已完成：超阈率 88.7%→6.1%（迭代）防御保持 3/5；严格到每格不超阈需缩放 ~1e-4（SNR 76dB）、防御降到 1/5——**不可感知与防御强度直接冲突**（[docs/16](docs/16_严格掩蔽与防御侧EOT.md)） | 1–2h | ✅ 已完成 |
| L3 | 频带能量分配 | 已完成试点：`--band-alloc` 按 θx 分配扰动能量；3 句试点防御不变（2/3）而 MCD 38.5→33.0（-14%），方向符合"同防御更低失真"假设，样本小待云端复验（[docs/18](docs/18_频带能量分配试点.md)） | 2–4h | ✅ 试点完成 |
| L4 | ASR 阈值论文映射 | Whisper WER 分布 vs 人工文本，给 Eq.10 的 10% 阈值可解释定义；`tools/asr_threshold_map.py` 已完成本地 5 句样本（云端可扩） | 1h | ✅ 工具完成 |
| L5 | 完整 Attack-VC 基线 | 用 `third_party/attack-vc` 官方超参补齐 emb/fb/e2e | 1–2h | ⬜ |
| L6 | 多种子本地小表 | 2 seed × 3 句试点：判定完全相同（2/3 vs 2/3）、MCD 38.5±3.5；种子敏感度低，更大规模留云端（[docs/17](docs/17_并行与多种子.md)） | 2–3h | ✅ 试点完成 |
| L7 | batch 并行攻击 | `scripts/run_parallel.py`：3 句串行 79.8s → 3 worker **36.7s（2.17×）**；云端可直接用 `--workers 2–4`（[docs/17](docs/17_并行与多种子.md)） | 2h | ✅ 已完成 |
| L8 | 前端接入 refine/质量模式 + 实时进度 | 质量模式（`vg-projected`）已存在；新增实时进度条：attack/pipeline `progress_cb` → Gradio `gr.Progress`（实测 100 次回调 0.03→1.00，[docs/17](docs/17_并行与多种子.md)） | 2–3h | ✅ 已完成 |
| L9 | 适配服务治理 | 已加：任务目录清理（`--keep-jobs`）、请求体积上限 20MB（413）、GPU 忙时 429、结构化 JSON 日志 + 请求 ID | 1–2h | ✅ 已完成 |
| L10 | CI（GitHub Actions） | 工作流模板 `deploy/ci.workflow.yml.template` **已在本地逐步验证通过**（compileall / test_features / test_masking / acceptance_check 全过）；启用需 PAT 带 `workflow` scope 或网页手动添加 | 1h | ✅ 内容验证通过，⏸ 待启用 |
| L11 | 一键验收自检脚本 | 起服务→冒烟→查交付物→打印 checklist；`tools/acceptance_check.py` 已跑通（必需项全过） | 1h | ✅ 已完成 |
| L12 | 攻击循环性能剖析 | `tools/profile_pipeline.py` 已备（分阶段耗时：Pipeline/特征/编码器/mask/protect） | 2h | ✅ 工具完成 |
| L13 | 结构化日志/请求 ID | 适配服务输出 JSON 日志（ts/request_id/method/path/status/elapsed_ms），响应头透传 X-Request-ID | 1h | ✅ 已完成 |
| L14 | 本地资源清理 | 已清理临时文件/参考仓库副本/cloud_bundle（约 260MB）；保留 `refs/wheels`（2.6GB 离线安装包） | 0.5h | ✅ 已完成 |

### 云端可做（需登录 T4 ECS 或云资源）

> **一键驱动**：`deploy/run_cloud_enhancements.py`（本地已 dry-run 验证）覆盖 C1 多种子、C2 扩展样本、
> C7 ASR 阈值映射与 C8 基础信道扫描：
>
> ```bash
> # 在 ECS /root/voiceguard 下：
> git pull
> .venv/bin/python deploy/run_cloud_enhancements.py --dry-run   # 先看要跑的命令
> .venv/bin/python deploy/run_cloud_enhancements.py --max-hours 6   # 断点续跑，已有结果自动跳过
> ```
>
> 结束后会打印「ECS 仍在计费」提醒；回传 `outputs/cloud/` 与 `outputs/enhancements/` 即可。

| # | 事项 | 内容 | 预计 | 状态 |
|---|---|---|---|---|
| C1 | 多种子完整批次 | `--seed` 跑 2–3 个种子，报告 mean±std（每种子 5 方法×5 句） | 每种子 1–1.5h | ⬜（驱动覆盖） |
| C2 | 扩展样本 | 句子数 5→15/人或更多说话人 | 2–3h | ⬜（驱动覆盖） |
| C3 | VQMIVC 黑盒迁移 | 论文 Table 3 跨模型验证（需先拿到权重，见 U5） | 1–2h | ⬜ |
| C4 | 镜像瘦身/冷启动 | 多阶段构建、裁剪 CUDA 组件（镜像 6–8GB → 更小） | 1–2h | ⬜ |
| C5 | HTTPS | nginx 反代 + 证书（平台页面为 HTTPS 时接口必须 HTTPS） | 1h | ⬜ |
| C6 | 云端留证与停机 | systemd 截图/日志归档；**ECS 关机/释放止计费** | 0.5h | ⚠️ 待办 |
| **C7** | **ASR 阈值映射规模化** | 在 3000/1500 步 + 完整上下文的云端数据上跑 Whisper WER 分布，给 Eq.10 的 10% 阈值可解释定义 | 15min | ✅ 工具就绪（`tools/asr_threshold_map.py`，驱动覆盖） |
| **C8** | **信道扫描扩展** | 信道已扩到 **11 种**（none/pcm16/loudness/mp3-128k/mp3-32k/aac-64k/opus-24k/g711/resample8k/resample16k/denoise/loudnorm，`vguard/channel.py`）；对 refined/staged/EOT 版本分别跑 | 每版 20min | ✅ 工具就绪（待云端跑） |
| **C9** | **评估器扩展 ⭐推荐** | 中文 ASV（ModelScope `3D-Speaker/ERes2Net`）+ Whisper `large-v3` 交叉验证 | 1–2h（含下载） | ✅ 工具就绪（`tools/asv_modelscope.py`，需 `pip install modelscope`） |
| **C10** | **staged 方法扩样本** | 现有 15/31，再跑 20–40 句把统计效力提上去 | 2–4h | ⬜ 云端跑（`deploy/run_staged_recovery.sh`） |
| **C11** | **内容源扩展** | 同一保护句 × 多内容源，报告方差而非单点 | 2–3h | ✅ 工具就绪（`scripts/run_content_sweep.py`） |
| **C12** | **长音频/真实朗读** | 已本地验证：10.88s 拼接音频完整跑通（300/200 步 40.3s，显存正常；该预算下防御失败属预期）；更长数据未测 | 1–2h | ✅ 工程验证通过 |
| **C13** | **跨数据集泛化** | LibriTTS / Common Voice 中文，验证不是 VCTK 特例 | 数小时（含下载） | ⬜ 需数据 |
| **C14** | **target 选择策略消融** | `--target-strategy least_likely/nearest/random` 已实现并冒烟通过 | 1–2h | ✅ 工具就绪（待云端跑） |
| **C15** | 48kHz 原版 VCTK | 与论文数据对齐（可选） | 数小时 | ⬜ 需数据 |
| **C16** | **云端压测** | `tools/load_test.py` 实测 2×200（并发/延迟/429 行为） | 1h | ✅ 工具就绪 |
| **C17** | **监听网页托管 ⭐妙用** | `tools/listening_app.py`（匿名 A/B/C/D 随机顺序 + 1–5 分 + 偏好，写 JSONL） | 1–2h | ✅ 工具就绪 |
| **C18** | 成本/稳定性 | 自动关机/空闲关机/留证模板：`deploy/auto_shutdown.template.sh` | 0.5h | ✅ 模板就绪 |
| **C19** | ModelArts 路线补验 | 想多一条部署路线加分时再考虑 | 2h | ⬜ 需云 |

> **云端推荐顺序**：C1（驱动直接跑）→ C9 评估器扩展 → C8 信道扩展 → C17 监听网页 → 其余按需。
> 本地与云端互不阻塞；云端产物回传后由我合并进报告/README/PPT。

### 需真人/外部账号

| # | 事项 | 说明 | 状态 |
|---|---|---|---|
| U1 | 真实组号 + 成员姓名 | 报告 §6、`deliverables/提交信息模板.md`、PPT 封面 | ⬜ |
| U2 | 课程平台 PR | 换组号 → 推 Fork → `npm test && npm run build` → PR（标题 `第XX组-音频主动防御`） | ⬜ |
| U3 | 人工听测 | 5–10 人 ABX（材料已打包 `VoiceGuard-listening-participant.zip`） | ⬜ |
| U4 | 问卷与最终课程提交 | 配合本人完成 | ⬜ |
| U5 | （如做 C3）VQMIVC 权重下载 | 浏览器下载后上传云盘 | ⬜ |

## 状态

- [x] 复现审计三批合并：可靠性修复、ECS 工具、云端战役、论文质量追赶、两阶段原生参考
- [x] 云端实验完成（32 阶段）：主表 9 组×5 句、ε 扫描 4 档、新 10 句、80 信道、60 内容源、MOS/WER 全补齐
- [x] 两阶段原生参考修正：GE2E 严格联合 9/31→15/31；完整干净控制 2/6
- [x] 系统：Gradio 前端（双语/昼夜）+ 课程平台适配服务 + Docker/ECS + 双 systemd 服务
- [x] 权重包：GitHub Release v1.0-weights + 百度/迅雷/夸克云盘（含 SHA256）
- [x] 交付物：17 页课程报告 PDF、14 页汇报 PPTX、质量补充报告 PDF、两阶段报告 PDF、提交信息模板
- [x] **本地优化项 L1–L14 全部落地**：L1 防御侧 EOT 试点、L2 严格掩蔽投影（不可感知-防御冲突量化）、L3 频带能量分配试点（MCD -14%）、L5 Attack-VC 一致性测试、L6 多种子试点、L7 并行加速（2.17×）、L8 前端实时进度、L9 服务治理、L11 一键自检、L12 性能剖析、L13 结构化日志、L14 资源清理；L10 CI 内容本地验证通过（待 PAT 启用）
- [x] **云端工具准备 C7–C19**：ASR 映射、11 种信道、ModelScope ASV、内容源扫描、target 策略消融、压测、监听网页、自动关机模板
- [~] 待用户侧：真实组号、组员姓名、平台 PR、问卷与最终提交、云上跑增强批次；ECS 记得关机


