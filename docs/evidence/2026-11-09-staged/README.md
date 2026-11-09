# 原生参考与两阶段实验的证据索引

本轮已完成开发、全局选择及二十句新验证。该目录属于独立第三轮，不覆盖 2026-11-09 和 2026-11-09-quality 的既有观测。最新报告为 docs/15_两阶段与原生参考修正.md，PDF 为 deliverables/Voice_Guard_原生参考与两阶段报告.pdf。

## 原因诊断

- transfer-diagnosis-old-enrollment.json 与 native-enrollment-diagnosis.json：相同旧 WAV，在两种 Es 注册上下文下的内部判据。干净原音正确数从 6/15 到 15/15，干净转换从 13/30 到 30/30；不算防护算法提升。
- strong-stage-development-audit.json：首次选中配置的开发强阶段和最终音频对照。两阶段严格联合均 4/6，不支持把所有差距归因于隐藏阶段。

## 两次开发与新验证

| 证据 | 内容 |
|---|---|
| protocol.json、development-state/results.json、selection.json | 初轮三候选与同句基线，共 24 条；选出 native_emb_snr20 |
| final/protocol.json、development-state/results.json、selection.json | 在同三句上追加软掩蔽比较，共 18 条；原生嵌入 4/6，软掩蔽 0/6；新二十句尚未查看时冻结选择 |
| final/validation-state/results.json | 二十新句、十位注册人物、两个内容源、两方法，共 80 条，全部 complete |
| final/analysis.json | 完整配对分母、按唯一保护 WAV 去重的质量、内容源及人物分组、说话人整组重采样区间 |
| final/native-validation-audit.json | 验证结束后，仅对既有 WAV 追加原生 Es 分类；不更改主分数或选择 |
| final/production-registry-check.json | 修正的常规构建器另建二十人各三十句库，与原生参考银行原型完全一致；不覆盖旧库 |
| final/final-tests.log | 39 项单测全部通过，包含原生读取/静音裁剪/立体声重采样回归 |
| final/archive.receipt.json、archive-verification.json | 新归档及逐成员本机核验 |
| final/report-inputs.json、final-verification.json | 当前报告输入、7 页 PDF 与页面预览 SHA、实际执行源码和 Git 文本核对 |

独立 GE2E/small.en：同句基线严格联合 9/31，本轮 15/31；身份控制 Eq.10 分别 14/31、19/31；完整干净控制 Eq.10 均 2/6。保护原音 MOS 均值 3.990、WER 1.27%、原身份保留 19/20。改进的描述性配对 95% 区间为 [-5.88, 44.83] 个百分点，不能据此建立稳定改进或论文等价性。

次评估 ECAPA/base 的身份控制 22/27（81.5%）不能代替总体复现：ECAPA 参与目标选取，完整干净控制仅 1/2。主 GE2E 完整干净控制只有 6/40 对合格，明确显示当前 VC/ASR 链路与论文设置的差距；不把本来就失败的干净转换算作保护收益。

## 实际执行字节与恢复

初轮协议 SHA256：16a228f795b33796dae973dca2d4a3ddbc332d02d24ba6ff8196157690628eec。

最终协议 SHA256：35a1373e6fa15baffa822794dd9b136424a7dec1111a17d7f145d9777e4ea3ba。

执行时九个沿用的 Python 文件是 CRLF，而 Git 文本是 LF，内容核对后确认仅换行不同。协议记录执行字节，新包的 executed-source 保存原始字节；最终交付核验单独记录执行 SHA 与 Git 文本 SHA。云端最终同步保留已冻结的运行文件，不以交付提交冒充更早的执行版本。

新包 outputs/staged_recovery/final-staged-evidence.tar.gz：33,613,073 字节；SHA256 874865a9d35672143a8cbbac105bb5724189698d805fd4d0efaf8f5ea2134c80。全部 658 个成员已核验；302 个 WAV 中，270 条文件路径的哈希与评分一致；35 个 strong.wav 中有 3 个参与独立强阶段诊断，其余为中间文件。重复字节或内容源不作为独立人物样本。

模型仍需旧完整模型包及第二轮评估器资源包。本轮目标银行和新建库随新实验归档；不进入普通 Git。先保存证据包及其 receipt，再在隔离目录执行：

```bash
python tools/archive_staged_recovery.py --verify-to outputs/staged_recovery/remote
python tools/analyze_staged_recovery.py --evidence docs/evidence/2026-11-09-staged/final --run outputs/staged_recovery/remote/outputs/staged_recovery/2026-11-09/final
```

有完整资源的独立工作副本可保留已提交的协议和 selection，直接运行 bash deploy/run_native_soft_comparison.sh validation。实际运行文件若被 Git 换行过滤改变，先从 executed-source 恢复协议指定的原始字节，再运行；不要改协议 SHA 来绕过输入检查。prepare/select 工具拒绝覆盖已冻结文件。再次研究使用新的协议、目录及未知验证句。

真人 MOS 尚未采集，第二种 VC 与原作者完整检查点尚未补齐；未宣称论文整体数值复现。ECS 未自动关机，任务完成不代表停止计费。
