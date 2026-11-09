# 课程交付材料

- `Voice_Guard_课程报告.pdf`：17 页正式技术报告，正文源为 `docs/04_报告.md`。
- `Voice_Guard_9分钟汇报.pptx`：14 页、讲稿计划 540 秒，可编辑文字和表格。讲稿见 `docs/05_验收PPT大纲.md`。
- `Voice_Guard_论文质量补充报告.pdf`：第二轮冻结验证与独立评估，应与原课程报告一起阅读；源为 `docs/14_论文质量追赶.md`。原报告与 PPT 保持上一轮结果。
- `Voice_Guard_原生参考与两阶段报告.pdf`：7 页最新附录，解释注册上下文问题，保留全部开发失败，报告二十句同句配对验证、独立声纹与完整干净控制。源为 `docs/15_两阶段与原生参考修正.md`。
- `提交信息模板.md`：填写真实组号、成员及分工、权重云盘链接和提交信息。

代码与实验完成情况按 `docs/08_验收自查.md` 核对。第一轮新增十句的原法与固定改进均为联合 0/8，该历史负结果保留。最新同句配对的 GE2E 严格联合为基线 9/31、本轮 15/31；完整干净控制下均 2/6，仍不能宣称达到论文总体质量。没有采集真人 MOS 或互评。

以下大文件已生成并在本机核验，不进入普通 Git：

- `outputs/course/final-complete-evidence.tar.gz`：完整实验及实际执行源码备份，700 个实验 WAV 摘要已逐项核对。
- `outputs/course/resources/VoiceGuard-model-resources-complete.zip`：AdaIN-VC、ECAPA、MOSNet、Whisper base 模型资源及恢复说明。
- `outputs/course/resources/VoiceGuard-listening-participant.zip`：15 段匿名试听材料，无人工评分结果。

摘要和核验收据见 `docs/evidence/2026-11-09/`。资源所有者仍须上传权重包并填写真实分享链接，按真实组号提交课程平台 PR，由本人完成问卷及最终课程提交。当前文件不代表课程已提交或验收通过。

第二轮另有 `outputs/paper_quality/final-recovery-evidence.tar.gz`（132 个 WAV）及 `VoiceGuard-quality-evaluator-resources.zip`（约 515 MB 新增评估器资源），配合旧完整模型包使用。冻结协议、独立评估、分母与哈希收据位于 `docs/evidence/2026-11-09-quality/`，不覆盖上一轮负结果。

第三轮新增 `outputs/staged_recovery/final-staged-evidence.tar.gz`（33,613,073 字节，302 个 WAV），658 个成员已在本机核验，其中 270 条评分音频路径与记录匹配。包含原生参考银行、修正构建器的新注册库、原始转录和实际执行源码；旧两个归档不变。归档与最终报告核验收据见 `docs/evidence/2026-11-09-staged/final/`。ECS 未自动关机。
