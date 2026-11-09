# 内容监督与环境鲁棒优化证据

这是输出反馈和防护保持研究之后的独立扩展。原有代码、协议、音频和负结果不覆盖。

- `protocol.json`：12 个已观察句用于开发；两个候选与同句原生嵌入基线；20 个新句仅在开发门槛通过后验证。
- `smoke.json`：真实 wav2vec2-CTC 输入梯度、模型冻结和七项新增数值/协议测试。
- `research-review.json`：用户提供 CSDN 文章的来源审核、目标区别与未核实声明。本项目不声称复现 Kalypta 官方算法或产品指标。
- `selection.json`、`extension-decision.json`：一次全局开发选择和新验证的强制门槛。开发复用历史句，不作为未知测试集成绩。
- `development-results.json`：全部开发分数和转录。`validation-results.json` 只有开发门槛通过才允许生成，缺失不等于通过。
- `analysis.json`：按原音去重的质量、完整干净控制和身份控制分母、代理接受/回退统计及独立 GE2E 配对结果。
- `channel-protocol.json`：四个已观察句、一个固定内容源，原始、另一个房间响应、MP3 64kbps 的事先固定诊断。各方法使用匹配的干净控制，不参与全局方法选择。
- `channel-protocol-initial.json`：任何信道结果产生之前的初始版本；仅汇总工具修复后重新冻结，保留追踪记录。
- `channel-analysis.json`：开发句上的有限信道诊断，不能当成新句泛化或真实房间测试。
- `archive-verification.json`、`final-verification.json`：完成后核验实际 WAV、输入、执行字节、报告和历史归档。

AdaIN 原生编码器和 wav2vec2-CTC 参与梯度；ECAPA 参与原音身份的代理接受检查。GE2E、MOSNet、Whisper base/small.en 均不参与优化或音频接受决策。代理约束不保证最终独立评估达标。

参考预处理是两组第一阶镜像声源反射、固定麦克风响应和重采样；不属于完整 Treble 仿真、实测 RIR 或平台录制。使用合成信道后的分数仅说明这些固定条件下的结果。

完整归档位于忽略目录 `outputs/content_robust/content-evidence.tar.gz`，需沿用已有模型及质量评估资源包，并补充协议列出的 wav2vec2-CTC 权重。报告和统计以实际完成状态为准，实验脚本不会自动关闭 ECS。

CTC 权重可使用 `python tools/fetch_content_ctc.py` 从官方 torchaudio 地址下载；脚本对照冻结协议校验完整 SHA256，不覆盖摘要不同的已有文件。离线复核使用 `tools/audit_content_results.py --evidence docs/evidence/2026-11-09-content-robust --run <解压根目录>/outputs/content_robust/2026-11-09`。

本轮开发 144 条、新验证 160 条、信道诊断 36 条均完成。开发完整联合由 7/16 到 8/16；20 新句两法均 10/22，严格身份联合由 25/54 到 24/54，原音质量均 17/20。没有建立独立联合改善，保留原生基线。完整说明见 `docs/18_内容监督与环境鲁棒优化.md` 与 `deliverables/Voice_Guard_内容监督与环境鲁棒报告.pdf`。
