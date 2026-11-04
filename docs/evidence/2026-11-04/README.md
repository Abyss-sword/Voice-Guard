# ECS 单句验证的文本证据

这些 JSON/CSV 来自 2026-11-04 华为云 ECS 的实际输出。
`smoke/` 是 60/30 步，`benchmark/` 是 3000/1500 步；二者均只含 p225_040 一句。
原始音频备份在本机 `outputs/deploy/initial-evidence/`，服务器 `/root/voiceguard/outputs/cloud/`。

`preflight.json` 记录 CUDA、模型摘要、数据数量与两份注册库形状；注册清单记录前 30 句的摘要。
`VCTK22_subset_manifest.json` 记录原 ZIP 成员 CRC32、文件 SHA256 和人工文本来源。
它由断点续跑补取文本时生成，`downloaded_bytes` 只计最后一次调用的传输量，
不是全部音频的累计下载流量，也不是完整 ZIP 摘要校验。

`frontend-interaction.json` 是真实 Gradio API 调用的摘录，移除了临时文件路径。
这里只保留文本元数据，不提交数据集音频、模型、SSH 信息或私钥。

首句身份维通过，但质量预测下降；这组证据不能证明论文总体防御率或不可感知性。
正式批量任务状态见 [华为云实测](../../11_华为云实测.md)。

后续质量改进的完整控制、固定参数验证、失败探索和 PCM16 检查已单独保存于
[quality/](quality/README.md)，不覆盖这里的首次单句记录。
