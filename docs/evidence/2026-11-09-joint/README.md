# 输出波形反馈与联合质量证据

这是第三轮之后的新实验，使用独立输出目录；原有协议、WAV、参考库和负结果不覆盖。

- `protocol.json`：开发前冻结的算法、代码/输入/模型摘要、20 个新句、4 个固定内容源和选择规则。
- `selection.json`：仅在三句开发后选定一个全局挑战者；验证开始前写入。
- `development-results.json`、`validation-results.json`：原始评分、完整转录、两套身份判定和全部门槛，失败不删除。
- `historical-clean-audit.json`：旧 40 对干净转换的失败原因，未改变旧门槛或结果。
- `clean-chain-diagnostic.json`：四个预定内容源的原始录音、仅重建和自转换质量诊断，不参与选源或调参。
- `smoke.json`：真实开发语音的前向重建一致性和输出声纹梯度检查。
- `analysis.json`：完整配对审核、每源/留出文本结果、按人物整组重采样的不确定性。
- `archive-verification.json`、`final-verification.json`：本机证据归档、执行字节、评分音频和交付文件核验。

主结果使用没有进入梯度的 GE2E、原始 MOSNet 和 Whisper small.en。ECAPA 参与优化，仅作代理结果。统计单位为转换对，但置信区间按人物整组重采样；相同原音的质量按唯一 WAV 去重。

最终方法和结论见 `docs/16_输出波形与联合质量优化.md`。后续防护保持候选在 `2026-11-09-joint-anchored` 单独记录，不能以其结果覆盖本轮负结果。

大文件归档在忽略目录 `outputs/joint_recovery/joint-evidence.tar.gz`，原始模型与额外质量评估资源需沿用旧资源包；归档含新输入、注册原型、输出音频与执行源码。协议中的代码哈希指实际云端执行字节，交付核验另记录 LF/CRLF 差异。

离线审核：`python tools/analyze_joint_recovery.py --evidence docs/evidence/2026-11-09-joint --run <解压根目录>/outputs/joint_recovery/2026-11-09`。
