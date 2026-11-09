# 2026-11-09 实验与系统证据

当前结论以 `report-data.json` 和课程报告为准。历史材料中的近似数字、未核验完成状态不用于本轮结果。

## 实验设计

`frozen-protocol.json` 在新增句子的优化与质量评测前保存，含十份新句的 WAV/人工文本摘要、选择规则、固定参数、信道与内容源。
它是本项目事前保存的协议，不是外部平台注册的预注册。新句来自已见说话人，不是 speaker-heldout。

原实验续跑为 9 组各 5 句：5 方法与 4 档 ε；全部 MOS/WER 使用保存音频及 VCTK 人工文本。
新句的两方法各 10 句，每方法另有 10 条 scale=0 控制，共 40 条质量评测。
配对信道共 80 次，内容源共 60 次，每次包括同条件干净和保护目标的重新转换。
这些共享说话人/内容源的重复不是独立受试者，不能按总行数夸大统计样本量。

`posthoc-context-protocol.json` 在看到新句的结果后追加，检查只恢复裁剪区间外原音的作用。
它不重新优化、不调参、不改变门槛，单列为事后诊断，不替代原冻结验证。

## 阅读顺序

| 文件/目录 | 用途 |
|---|---|
| `report-data.json` | 报告使用的实测逐句结果、清单摘要和完整状态 |
| `cloud/cloud/` | 原主表/ε 的 manifest、逐句结果、MOS/WER、完整 campaign 摘要 |
| `cloud/course/2026-11-09/` | 冻结新句、配对信道、内容源及追加上下文诊断；含每阶段日志与输入/代码哈希 |
| `cloud/quality/` | 首批改进、仅投影、缩放、PCM16、完整上下文和独立重跑的原记录 |
| `extension-analysis.md` / `extension-diagnostics.json` | 完成后从逐句记录计算的量化解释和 MP3 固定条件诊断 |
| `nominal-repeat-diagnostic.json` | 主表 VG 与 ε=.05 的同参数独立运行差异；原因未被独立定位 |
| `cloud/course/2026-11-09/asr_repeatability_posthoc/` | 同五份 WAV、两种子及两种解码条件，共 20 次事后 ASR 控制，保留默认回退幻觉 |
| `final-archive.receipt.json` | 云端全量备份、本机摘要和逐项实验 WAV 核验 |
| `platform-api-projected.json` / `platform-api-response.json` | 22.05 kHz 双声道、自动身份、快速改进的真实 GPU 请求及下载 |
| `platform-browser.json` / `platform-*.png` | 真实课程浏览器载入样例、提交、5 音频、6 产物和实际下载 |
| `gradio-api.json` | 原 VG 快速模式、完整上下文的真实 Gradio 请求 |
| `platform-contract.json` / `local-checks/` | schema、参数、样例、单元/数值测试、构建与真实编解码检查 |
| `model-resources-complete.receipt.json` | 含 Whisper 的完整模型包，29 个成员已逐个核验；未上传云盘 |
| `listening-*.json` | 15 音频试听包及浏览器下载验证；未收集人工评分，表单测试值是合成输入 |
| `slides-validation.json` | 当前交付 PPT 的结构检查记录；视觉检查另由逐页预览完成 |
| `deliverables-validation.json` | 最终报告 17 页、PPT 14 页及逐页视觉复核和文件摘要 |

追加完整上下文诊断共 20 条（10 句各有无扰动），联合 0/7。续跑质量缓存新增 2 项回归测试通过，并对归档内 18 份完整 MOS/WER 文件验证可保留，见 `local-checks/quality-resume.json`。保留缓存用于防止续跑默认随机解码覆盖历史观测，不改变已有实验版本。

## 关键负结果

原 VG 首批身份维为 2/4，联合门槛为 0/4。固定谱投影改进在首批五句中联合通过 2/4，但在十个新句中原法和改进均为 0/8。
新句改进的 MOSNet 从 1.936 提升为 3.055，宏平均 WER 从 44.60% 降为 14.69%，仍不代表联合防护有效。
零扰动新句本身 MOSNet 为 3.088、宏平均 WER 为 9.50%；全部内容错误不能归因于扰动。

服务验证采用快速模式，证明 I/O 和 GPU 流程可运行。在线例子的干净 VC 未通过筛选，页面正确显示未通过，未当作成功防御。
正式实验保留完整 3000/1500 和 c 搜索语义；API 快速模式不是完整实验的性能替代。

## 完整音频与源码

大批 WAV、数据和模型不进入普通 Git 提交。含音频的全量证据包在本机 `outputs/course/`，摘要见收据。
包内 `executed-source/` 保存云端导出时的源码字节；各阶段 manifest 的代码摘要用于识别真正执行版本。
算法分支的最终交付提交可能晚于实验，不把它反向写成实验开始时的版本。

模型包 `outputs/course/resources/VoiceGuard-model-resources-complete.zip` 需由资源所有者上传云盘并填写真实 URL。
包含 AdaIN-VC、ECAPA、MOSNet、Whisper base；不含数据、注册库或运行环境，恢复步骤与来源随包保存。
试听包只供后续试听，不构成人工 MOS、正式盲测或真人参与证据。

重新生成报告和图表见根 README。修改输入、参数或被记录的评估代码时使用新的实验输出目录，不能覆盖已有验证结果。
