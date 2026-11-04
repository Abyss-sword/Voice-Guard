# 质量实验文本证据

来自华为云 `/root/voiceguard/outputs/quality` 和 `outputs/cloud/main_vg`，2026-11-04 实际运行。
说明与协议见 [质量改进实验](../../../12_质量改进实验.md)。

`comparison.json` / `.md` 由 `python tools/build_quality_report.py` 从 PCM16 的两份 `summary.json`
及原始 VG 的 MOS/WER CSV 重建，包含来源 SHA256、全部 5 句、开发/固定参数验证分组。
`frozen_refinement_profile.json` 是开发后、验证前保存的参数；不是注册在外部平台的预注册协议。

目录保留缩放、仅投影、投影后优化、FLOAT/PCM16 评估、完整上下文诊断的全部文本结果。
`baseline_main_vg/` 是本轮主实验原始 VG，而非先前单句 benchmark。
`refine_dev/`、`refine_validation/` 的 `optimization.json` 含优化轨迹与波形摘要；
评分、转录和身份结果在对应 `*_eval` 目录，筛选后的统计在 `summary.json/results.csv`。
历史 manifest 的代码摘要对应运行当时版本；本轮后续增加了导出验证、余弦分数记录及续跑检查。
数值算法 `vguard/attack.py` 与 `vguard/masking.py` 的摘要在固定配置中保留。
`executed-source/` 保存这两个模块当时执行的原始字节（`.py.txt`）。Git 对源码换行的规范化会改变
原始 SHA256；`source-hashes.json` 映射执行时摘要和提交中摘要，并已核对仅 CRLF/LF 不同。
证据目录关闭 Git 换行转换，保留原始 CSV 和源码快照的可核验摘要。
`refine_replay_p225*` 与 `replay-check.json` 是最终生成脚本的一次独立重跑：联合门槛仍通过，
但原始与重跑波形摘要不同，不声称 CUDA 逐位确定性，也未替换主比较表中的首次结果。

原始音频与日志包备份：

- 本机 `outputs/deploy/quality-evidence-20261104.tar.gz`
- 本机补充独立重跑 `outputs/deploy/quality-replay-20261104.tar.gz`
- 本机解包 `outputs/deploy/quality-evidence/outputs/quality/`
- 云端 `/root/voiceguard/outputs/quality/`

音频、模型、数据集、SSH 凭据不提交 Git。这里的绝对模型路径是运行记录，不包含凭据。
`campaign-snapshot.*` 为 14:32 UTC 的旧进度快照，`complete:false`，其 MOS/WER 空项表示当时未汇总，
不覆盖本目录之后单独完成的原始 VG WER。它不代表主实验最终结果。

结论：两个小样本通过工程联合门槛，论文级总体质量仍未证实；完整音频诊断的失败和排除项均保留。
