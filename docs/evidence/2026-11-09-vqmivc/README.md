# 官方 VQMIVC 有限迁移诊断

固定官方仓库 `Wendison/VQMIVC` 的提交 `851b4f5ca5bb60c11fea6a618affeb4979b17cf3`，使用其 v0.0 模型和 ParallelWaveGAN 声码器。保留 41 个推理文件、原许可证、原始字节摘要，未改写第三方网络。适配器只处理现代依赖接口、内存内声码器调用和成对固定随机噪声。

六个已观察参考句、两个固定内容、基线及两个判别候选，共 36 条记录；`transfer-plan.json` 早于任何 VQMIVC 防护评分。该模型不参与 AdaIN 的梯度、开发方法选择或检查点接受。相同种子的已知干净转换两次字节一致，模型全部冻结；这项烟雾检查不是防护证据。

完整干净控制的有效分母为 0：12 个转换中 6 个身份不通过，其余 6 个质量不通过。不能报告完整联合率，更不能把干净模型失效算作防护。严格身份联合依次为基线 1/6、原预算判别 1/6、放宽预算判别 3/6；它们是历史开发句、不同预算下的有限诊断，不是新句迁移或论文规模结果。

在 Linux 已有主环境中安装隔离的可选依赖：

```bash
.venv/bin/python -m pip install --no-deps --target outputs/vqmivc/vendor pyworld==0.3.5 resampy==0.4.3
.venv/bin/python tools/fetch_vqmivc_resources.py
PYTHONPATH=outputs/vqmivc/vendor OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python tools/smoke_vqmivc.py --score
```

模型、编译工具链和原主环境依赖需先按项目部署文档准备。下载器不覆盖已有收据；大模型逐块校验长度并计算摘要。摘要是本地下载实测值，未建立独立上游摘要。启动时设置 `PYTHONPATH`，避免旧版 pyworld 的分发包发现机制受导入顺序影响；首次失败状态与日志保留。

先运行 `tools/prepare_vqmivc_transfer.py` 冻结实际资产，再以同一启动环境运行 `scripts/run_vqmivc_transfer.py --max-hours .5`。审核使用 `tools/audit_vqmivc_transfer.py`。

证据归档为 `outputs/vqmivc/transfer-evidence.tar.gz`；独立权重包为 `outputs/vqmivc/vqmivc-official-resources.tar.gz`。后续输入路径对照在相邻 `2026-11-09-vqmivc-transport` 中，保留直接迁移的原结果。
