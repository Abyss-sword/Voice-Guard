# 防护保持约束扩展证据

第一轮输出波形反馈在开发集上改善、部分新验证出现退步后，另冻结本扩展。第一轮所有结果保留；本扩展不得用于事后重算第一轮分数。

- `protocol.json`：固定相对基线的输出身份目标、ECAPA/原生编码器代理、原音软约束、代码/资源摘要和另一组十句新验证。
- `selection.json`、`development-results.json`：三句开发结果、一次全局选择。
- `extension-decision.json`：新验证前强制检查开发门槛，完整联合至少 5/6 且原音质量 3/3；未通过则停止于开发。
- `validation-results.json`、`analysis.json`：仅通过开发门槛才生成；不得把文件缺失解释为验证通过。
- `smoke.json`：真实开发样本的多代理输出梯度检查，模型权重冻结。
- `archive-verification.json`、`final-verification.json`：独立归档与交付核验。

决策依据的第一轮部分结果快照在归档内 `outputs/joint_anchored/first-joint-partial-observation.json`，其摘要与记录数在协议中。扩展的验证句与第一轮、注册集和此前已优化句子分离，仍是已注册人物，不能称为完全未见人物。

ECAPA 和原生 AdaIN 声纹参与梯度，只有 GE2E 是留出的声纹模型。软目标和截断的相位/裁剪梯度不能保证独立联合成功。所有评估均在实际 PCM16 原音和官方链路转换的 WAV 上完成。

说明与最终结论见 `docs/17_防护保持约束与联合质量验证.md`；综合 PDF 为 `deliverables/Voice_Guard_联合质量优化报告.pdf`。独立归档在忽略目录 `outputs/joint_anchored/anchored-evidence.tar.gz`。

通过开发门槛后的离线审核：`python tools/analyze_joint_recovery.py --evidence docs/evidence/2026-11-09-joint-anchored --run <解压根目录>/outputs/joint_anchored/2026-11-09`。门槛失败时仅审核开发与停止决定，不运行该验证分析命令。
