# 原生注册判别间隔研究

二十位注册人物各三十句拟合收缩 LDA；十二个历史句用于开发，不是独立测试集。参考和解码后梅尔身份间隔参与梯度，完整上下文 CTC 约束内容；独立 GE2E、MOSNet 和 Whisper 不参与优化。

`protocol.json` 在新版运行前冻结源码、输入、权重摘要、两个候选和二十个预留新句。`initial-protocol.json` 是投影参数纠错前的初版；初版源码、部分输出和中断状态保存在完整归档中。二者不混合计数。

本轮 144 条开发记录：原预算判别方案完整联合 10/16，配对基线 8/16；严格身份联合 24/37，基线 17/37。原音质量均 10/12、WER 均 4.26%，预测 MOS 4.164 到 4.101。MOS 降幅超过事前 0.05 门槛，因此 `extension-decision.json` 禁止本方案直接进入新句验证。二十个预留句交给另行冻结、通过门槛的全局混合续研。

- `selection.json`：唯一全局选择，不按句或评分器挑音频。
- `development-results.json`：完成后发布的全部原始分数和转录。
- `analysis.json`：实际 PCM16 哈希、配对分母、代理检查、SNR 和按人物重采样的描述性区间。
- `masking-constant-diagnosis.json`：论文附录与作者代码两处常数的单独敏感性诊断，未修改历史掩蔽实现。
- `archive-verification.json`：本地解压后逐项核验的收据；模型权重另用已有资源包。

完整证据：`outputs/discriminant/discriminant-evidence.tar.gz`。归档保留 600 个注册输入及精确执行源码；大文件在忽略目录，Git 中保留摘要和可审查记录。脚本不会自动关机。

运行开发：`python scripts/run_discriminant.py --split development --max-hours 1`；完成后 `--split select`，再执行 `python tools/decide_discriminant_validation.py`。退出码 3 代表未通过开发门槛，不能接着运行该方案的新句验证。
