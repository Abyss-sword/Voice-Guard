"""Voice Guard: 时域对抗扰动 + 心理声学掩蔽的语音隐私主动防御复现库。

核心模块：
- features : 可微 mel 提取器 H（梯度从频域传回时域波形）
- masking  : 心理声学掩蔽阈值 θx（Appendix A）+ 扰动 PSD 的 Transform
- attack   : Eq.3 / Eq.7 / Eq.8 + Algorithm 1（两阶段 + α 二分）
- vc       : AdaIN-VC / VQMIVC 加载与转换
- speaker  : 说话人判别（GE2E 或 ECAPA 兜底）
- metrics  : MCD / MOS / WER / ASV / 防御判定
"""

__version__ = "0.1.0"
