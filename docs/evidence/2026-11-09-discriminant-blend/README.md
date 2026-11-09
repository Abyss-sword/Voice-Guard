# 固定全局混合续研

父研究的防护提高但预测 MOS 降幅超过预设门槛，且没有使用预留验证集。本研究在开发评分前另行冻结 25%、50%、75% 三个统一混合系数：`baseline + lambda * (lda20 - baseline)`。

开发复用父研究的精确 PCM16 基线、判别优化结果和干净控制；独立分数按音频哈希缓存。没有按句选择系数，也没有重复计算优化。最终 PCM16 再检查训练代理，失败回退同句基线。所有系数均使用 SNR20、原掩蔽预算，不使用父研究的放宽预算分支。

192 条开发记录按预定规则选中 `lda_blend50`：完整联合 8/16 到 9/16，严格身份联合 17/37 到 22/37，原音质量均 10/12、WER 均 4.26%，预测 MOS 4.164 到 4.126。全部开发门槛通过，才允许在二十个预留句上评测同句基线和这个唯一候选。新验证结果以完整状态、全部原始记录及审核统计为准。

核心文件为 `protocol.json`、`selection.json`、`extension-decision.json`、完成后发布的 `development-results.json` / `validation-results.json`、`analysis.json` 与归档核验收据。父研究证据必须同时保留。

命令：`python scripts/run_discriminant_blend.py --split development --max-hours .666`；全局选择 `--split select`；门槛 `python tools/decide_blend_validation.py`；通过后 `--split validation --max-hours 1.25`。审核：`python tools/audit_blend_results.py`。实验不自动关机。
