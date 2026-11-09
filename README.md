# Voice Guard 复现与探索

论文：*Voice Guard: Protecting Voice Privacy with Strong and Imperceptible Adversarial
Perturbation in the Time Domain*，IJCAI 2023，4812–4820。

实现时域对抗扰动、心理声学掩蔽、AdaIN-VC 转换、独立 ECAPA 身份评估和 Gradio 界面。
目标是可核验的抽样复现；数据、ASV、声码器及 MCD 定义与原论文有差异，不能宣称精确复现论文数值。

## 当前状态（2026-11-09）

第二轮质量实验已完成：冻结 15 句、两个内容源，保护原音原始 MOSNet 预测均值 4.063、Whisper base 逐句 WER 4.61%；严格身份联合 8/21，Eq.10 结构 16/21，完整干净控制下 4/7。独立 GE2E/small.en 复核分别为 11/27、16/27、5/12，不能宣称达到论文 80.06% 的总体复现。旧结果保留，详见 [补充实验](docs/14_论文质量追赶.md) 与 [补充报告 PDF](deliverables/Voice_Guard_论文质量补充报告.pdf)。

- 初始目录为无 `.git` 的源码快照；现已连接原仓库并基于远程 `master` 创建独立分支。旧设备与实验完成记录不代表本机现状。
- 已修复重采样、批量断点续跑、随机种子、筛选分母和 WER 参考文本，补齐模型下载与 ECS 部署工具；本轮进一步修复平台上传采样率、缺失 ASV 判定与请求串行处理。
- 已连接用户购买的华为云 GPU ECS：Ubuntu 22.04、Tesla T4（15GB 可用显存）、32GB 内存、100GB 系统盘；驱动和模型已就绪，实验执行状态见复现审计。
- 新报告和图表使用云端归档实测数据。历史表/旧报告和 PPT 仅作历史材料，未验证的数字不用于当前结论。
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

这些是对已存 Markdown 行的重新计数，不是本轮运行结果。历史 VG ε 扫描缺原始逐句记录，
75/75/50/50% 仅为历史声明。本轮重跑的身份维为 3/4、2/4、2/4、2/4，四档联合质量通过均为 0/4，详见当前报告。

## 目录

`vguard/` 核心算法与评估；`scripts/` 注册、保护、转换与实验；`frontend/` Gradio；
`deploy/` ECS/Docker；`tools/` 下载、预检和统计；`tests/` 数值与回归；
`third_party/` 上游参考；`docs/` 方案、历史报告、实验协议与审计。

权重/数据/大批输出不入源码包。真实云端证据已归档；成员、权重云盘链接与正式课程提交待补齐。

## 课程平台接入

本轮更新 `platform/` 适配服务、1.1.0 方法配置与三份有来源摘要的真实音频。
启动及接口说明见 [课程平台接入说明](docs/10_课程平台接入说明.md) 和
[小组 README](platform/README-group.md)。平台 Fork 已存在，本地副本的 6 项测试、配置校验和生产构建通过；
本轮材料未推到平台 Fork；新版 GPU 接口、实际浏览器样例载入/处理/5 段音频预览与下载通过。
可选参考输入导致样例按钮拒绝执行的问题有补丁 `platform/patches/optional-input-sample.patch`。
真实组号、成员、云盘链接与正式 PR 仍待补齐。

完整模型分发包位于忽略目录 `outputs/course/resources/VoiceGuard-model-resources-complete.zip`（253,422,166 字节），
含 AdaIN-VC、ECAPA、MOSNet、Whisper base、来源、逐文件摘要与恢复说明；不含数据、注册库及运行环境。
本机已核验压缩包和全部成员摘要，SHA256 为
`bf2254d9413028820bba5ddb7e8170cf02abe1d820d287d0e049847f00d9e208`，记录见
`docs/evidence/2026-11-09/model-resources-complete.receipt.json`。
上传后在此补真实云盘分享链接：**尚未提供**。填写与演示流程见 [提交信息模板](deliverables/提交信息模板.md)。
已有的含资源打包工具 `tools/make_cloud_bundle.py` 同样保留；新 ECS 流程使用源码包并在云端下载资源。
