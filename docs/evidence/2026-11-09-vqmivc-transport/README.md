# VQMIVC 输入路径对照

这是直接迁移结果产生后的诊断，不能冒充事前未知测试。已知正常参考在原始采样率路径下的预测 MOS 为 3.494，经 AdaIN 采样率与 PCM16 导出后为 2.554；同一句、同一种子。因路径同时涉及重采样和编码，不能据此归因于单个因素。

随后在任何搬运对照评分前固定同样六个开发参考、两个内容、三种方法和成对种子。16 kHz 保护参考构造为 `R(native original) + R(defended24 - original24)`，随后所有参考都导出 PCM16。该操作改变了输入通道，不能替代直接波形迁移，也不保证原 24 kHz 掩蔽约束在 16 kHz 保持。

36 条记录完成，但完整干净控制分母仍为 0，严格身份联合与直接迁移相同：1/6、1/6、3/6。这个对照未解决基础转换质量问题，也没有建立独立迁移收益。相关原始音频、2×2 已知干净参考诊断和重采样特征检查保存在 `outputs/vqmivc/transport-evidence.tar.gz`；它们是对根因的线索，不是因果结论。

在 VQMIVC 可选启动环境中运行 `tools/prepare_vqmivc_transport.py`、`scripts/run_vqmivc_transport.py --max-hours .5`；审核使用 `tools/audit_vqmivc_transport.py`。协议、输入波形、代码和资源摘要均需匹配，不覆盖既有协议或归档。
