# 第XX组：音频主动防御 Voice Guard

> 本文件按课程平台 `docs/小组README模板.md` 填写，用于提交到本组 Fork 的
> `lym-lab/algorithm_research_frontend` 根目录（提交前把「XX」替换为实际组号）。
> 算法源码仓库：https://github.com/Abyss-sword/Voice-Guard

## 1. 小组与项目

| 信息 | 本组填写 |
| --- | --- |
| 组号 | 第 XX 组 |
| 选题名称 | 音频主动防御 Voice Guard |
| 成员 | 成员 A（算法与实验）、成员 B（工程与部署） |
| 方法 ID | `group_xx_voice_guard` |
| 本组仓库与提交分支 | 算法源码：github.com/Abyss-sword/Voice-Guard；Fork 与提交分支见 PR |

本组复现 IJCAI 2023《Voice Guard》的**时域对抗扰动 + 心理声学掩蔽**方法，面向**语音/音频**素材：
在发布前给说话人音频注入人耳不可感知的扰动，使零样本语音转换（AdaIN-VC）无法再伪造其声纹，
实现语音隐私的主动防御。

## 2. 论文与代码来源

- 论文：Li et al., *Voice Guard: Protecting Voice Privacy with Strong and Imperceptible
  Adversarial Perturbation in the Time Domain*, IJCAI 2023, pp.4812–4820
  （https://www.ijcai.org/proceedings/2023/535）
- baseline/开源项目：官方 AdaIN-VC（jjery2243542/adaptive_voice_conversion，权重为官方
  NTU 直链版本）；官方 Voice Guard 仓库仅含部分代码（Transform / asr_model 未发布，本组补全）；
  攻击对照的频域基线来自 attack-vc（cyhuang-tw/attack-vc）。
- 本组算法代码：https://github.com/Abyss-sword/Voice-Guard（本适配服务位于
  `platform/service/adapter.py`，方法配置 `platform/methods/group_xx_voice_guard.json`）。
- 第三方许可证：AdaIN-VC（MIT）、attack-vc（MIT）；复现代码按各自 LICENSE 保留引用。

## 3. 本组实现内容

| 功能 | 完成状态 | 实现位置与验证方式 |
| --- | --- | --- |
| 可微 mel 提取器 H（PyTorch 重写） | 已完成 | `vguard/features.py`；与官方 numpy 预处理逐点误差 ≤1e-4（tests/test_features.py） |
| 心理声学掩蔽阈 θx + 可微 Transform | 已完成 | `vguard/masking.py`；同 96dB 尺度归一化（tests/test_masking.py） |
| 两阶段攻击 + c 二分（Eq.3/7/8, Alg.1） | 已完成 | `vguard/attack.py`；端到端批量实验 outputs/exp* |
| 评估链（ECAPA/Es 探头/MCD/WER/MOS） | 已完成 | `vguard/asv.py`、`vguard/speaker.py`、`vguard/metrics/` |
| 平台 HTTP 适配服务 | 已完成 | `platform/service/adapter.py`；契约见平台 `docs/算法接入规范.md` |
| 实验验证（ε 扫描/方法对比/消融/信道分析/EOT） | 已完成 | `docs/04_报告.md`、`docs/06_结果表.md`；时域 VG 75% vs 频域基线 25%（ε=0.05） |

## 4. 输入、输出与参数

| 输入字段 | 素材类型与格式 | 必填 | 限制与说明 |
| --- | --- | --- | --- |
| `defend_audio` | audio（wav/mp3/flac/ogg） | 是 | 要保护的语音，建议 1–8 秒、单声道、≤30MB |
| `adv_audio` | audio（wav/mp3/flac/ogg） | 否 | 扰动方向音频（least-likely 说话人）；留空由服务自动选取 |

| 参数字段 | 类型 | 默认值与范围 | 作用 |
| --- | --- | --- | --- |
| `eps` | number | 0.05，范围 0.02–0.3，步长 0.01 | 扰动强度；0.05 为实验中防御率与感知质量的最佳折中 |
| `fast_mode` | boolean | true | 快速模式（300/200 步，GPU 约 30–90 秒）；关闭用完整 1500/750 步 |
| `target_speaker` | text | 空（自动识别） | 被保护说话人编号（如 p225）；留空用 ECAPA 自动识别上传音频 |

| 输出 | 文件类型 | 含义 |
| --- | --- | --- |
| 保护后音频 | WAV | 注入时域扰动后的音频（F_imp：仍被识别为本人） |
| 扰动本身 | WAV | δ=2ε·tanh(w) 的波形 |
| 对照转换 / 防御后转换 | WAV | 干净与被保护音频分别经 AdaIN-VC 转换的结果（F_str：被保护后不再被识别为本人） |
| 处理报告 | JSON | 参数、判定、ECAPA 相似度等 |

## 5. 环境与安装

- 已验证：Ubuntu 22.04 / Windows 11，Python 3.12，PyTorch 2.5.1（CUDA 12.4，或 CPU）。
- 硬件：GPU（≥8GB 显存）或 CPU（仅建议快速模式演示）；显存峰值 ~4.4GB。
- 依赖文件：源码仓库 `requirements.txt`；关键版本链固定为
  `gradio==4.44.1, gradio-client==1.3.0, fastapi==0.112.4, starlette==0.37.2, jinja2==3.1.2`。
- 权重：AdaIN-VC 官方权重（NTU 直链）到 `checkpoints/adain-vc-official/`；
  注册库 `outputs/registry*.npz`（本仓库生成脚本可复现）。

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## 6. 启动与运行

1. 启动算法服务（适配服务，监听 8020）：

   ```bash
   python platform/service/adapter.py \
     --model-dir checkpoints/adain-vc-official \
     --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz \
     --vctk data/VCTK22 --device cuda --host 0.0.0.0 --port 8020
   ```

2. 方法配置 `platform/methods/group_xx_voice_guard.json` 的 `adapter.endpoint`
   指向上述服务地址（本地测试用 `http://127.0.0.1:8020/process`）。

3. 前端（本组 Fork 的 algorithm_research_frontend）：

   ```bash
   npm install
   npm run dev
   ```

   在「方法集合」中搜索 Voice Guard，上传音频、填 ε、点「开始处理」。

4. 第一次运行：使用 `public/samples/group_xx_voice_guard/input.wav`，参数
   `eps=0.05, fast_mode=true, target_speaker=p225`；结果出现在工作区并可下载。

算法配置文件位置：`public/methods/group_xx_voice_guard.json`（在 Fork 中的实际链接）。

离线/其他入口：本组自研 Gradio 界面见源码仓库 README「本地启动交互界面」一节。

## 7. 样例与实际结果

| 样例 | 输入链接 | 输出链接 | 参数 | 观察重点 |
| --- | --- | --- | --- | --- |
| VCTK p225 单句保护 | [input.wav](public/samples/group_xx_voice_guard/input.wav) | [output_defended.wav](public/samples/group_xx_voice_guard/output_defended.wav) | eps=0.05, fast_mode=true, target_speaker=p225 | 保护后音频听感接近原声（掩蔽约束） |
| 同输入经 VC 转换 | 同上 | [output_converted.wav](public/samples/group_xx_voice_guard/output_converted.wav) | 同上 | 转换后不再被识别为 p225（F_str 通过） |

更多实验：ε 扫描（75/75/50/50）、方法对比（时域 75% vs 频域 25%）、信道分析与 EOT 对照，
见源码仓库 `docs/04_报告.md`。

## 8. 验证与已知限制

- 已执行：`tests/` 全过；平台契约用 requests 端到端测试（6 个产物全部 200 可下载，
  defense_ok=True）；`npm test` / `npm run build` 在 Fork 中执行。
- 接口状态：按算法服务实际部署情况填写（本地/云主机运行时段）。
- 已知限制：VC 端为官方 AdaIN-VC（24kHz），不同 VC 模型效果有差异；ε 过大时被保护音频
  自身身份受损；CPU 处理慢（建议快速模式）；取消请求后服务端任务可能继续（由服务清理策略决定）。
- 耗时：快速模式 GPU 约 30–90 秒/句；完整 1500/750 步约 10–20 分钟/句（2s 音频）。

## 9. 附加文档

- [项目报告（实验与结果）](https://github.com/Abyss-sword/Voice-Guard/blob/master/docs/04_报告.md)
- [实验计划与协议](https://github.com/Abyss-sword/Voice-Guard/blob/master/docs/03_实验计划.md)
- [环境与进度记录](https://github.com/Abyss-sword/Voice-Guard/blob/master/docs/02_环境与进度.md)
- [结果表](https://github.com/Abyss-sword/Voice-Guard/blob/master/docs/06_结果表.md)
- [适配服务代码](https://github.com/Abyss-sword/Voice-Guard/blob/master/platform/service/adapter.py)
