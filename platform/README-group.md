# 第XX组：音频主动防御 Voice Guard

本文件供复制到课程平台 Fork 根 README；正式提交前填写真实组号与成员。

| 项目 | 内容 |
|---|---|
| 选题 | A036 Voice Guard，按非开源单篇项目验收 |
| 成员、学号后四位、组长 | 待本人填写 |
| 个人贡献 | 待本人确认，每人不超过 50 字 |
| 源码分支 | https://github.com/Abyss-sword/Voice-Guard/tree/codex/huawei-ecs-reproduction |
| 方法 ID | `group_xx_voice_guard`（组号待替换） |

## 方法与来源

Voice Guard 在语音波形中优化扰动，尝试阻碍 AdaIN-VC 复制目标声纹，同时约束保护语音质量。
实现可微 mel、心理声学掩蔽及两阶段/c 搜索，并提供额外谱投影继续优化的实验性方法。
论文：https://www.ijcai.org/proceedings/2023/535 。官方部分代码 WHUB401/Voice-Guard，
VC 来源 jjery2243542/adaptive_voice_conversion；保留上游许可证和引用。

评估使用 20 人闭集 ECAPA、第三方 MOSNet 与 Whisper WER，均不同于论文完整协议。
目前首批 5 句原始 VG 平均 MOSNet 1.397、宏平均 WER 62.76%；改进后为 3.192、5.71%。
改进法开发集筛选后联合通过 1/2，固定参数验证集为 1/2，均为极小样本。
没有证据保证人耳不可感知、跨模型防御或论文总体性能。
11 月 9 日冻结扩展 32 阶段全部完成：新 10 句原法/改进联合均为 0/8，MOSNet 为 1.936/3.055，WER 为 44.60%/14.69%。
另完成 80 次信道、60 次内容源配对与上下文/ASR 事后控制。局部质量改善没有形成新句联合防护，完整结果和限制见算法仓库报告。

## 输入输出与参数

输入 `defend_audio` 必填，`adv_audio` 可选；建议 1–8 秒单人语音。
自动识别会按实际采样率重采样/下混，再作闭集匹配，库外身份不可靠。

| 参数 | 默认 | 含义 |
|---|---|---|
| eps | 0.05，范围 0.02–0.3 | 已测起点，更大不保证防御或质量更好 |
| fast_mode | true | 300/200 步；关闭使用服务设定的完整预算 |
| target_speaker | 空 | 可指定注册说话人，空值自动匹配 |
| method | vg | vg 原方法，vg-projected 额外 1000 步优化 |
| preserve_context | false | 保留裁剪区间以外原样本，改变评估协议 |

输出原音、保护音频、扰动、干净转换、保护后转换，以及 JSON 报告，共 6 个可下载产物。
身份维通过需干净转换控制成立；API 未计算离线 MOS/WER，不能直接宣布联合质量通过。
同步 GPU 任务可能持续数分钟，具体耗时以报告为准，快速模式结果不代表正式实验。

## 启动

算法环境、模型和数据按源码仓库 README 配置。在 ECS 项目目录：

```bash
bash deploy/install_platform_service.sh
curl http://127.0.0.1:8020/health
```

本机经 SSH 隧道把 8020 转发到云端回环地址；配置 endpoint 为
`http://127.0.0.1:8020/process`。课程前端运行 `npm install`、`npm test`、
`npm run build`、`npm run dev`。HTTPS 公共部署还需 HTTPS 可访问后端。
完整模式的新服务配置 3000/1500；算法服务 CLI 的缺省是 1500/750，应显式设置或用 systemd 脚本。

## 样例及证据

`public/samples/group_xx_voice_guard/` 包含 p225_040 的真实 PCM16 样例及 `sample-provenance.json`。
保护音频 MOSNet=3.275、WER=0，身份 p225，转换后 p246；离线内容源为 p226_002。
它是离线实验产物，交互 API 自动选择的内容源不同，不能保证返回同一 WAV 或判定。

源码报告 `docs/04_报告.md`，证据 `docs/evidence/2026-11-04/quality/`。
2026-11-09 新配置的 schema、默认值、样例及前端测试/构建通过；新版云端 API 和真实浏览器运行/下载通过。
平台可选参考样例补丁位于算法仓库 `platform/patches/optional-input-sample.patch`，需应用到本 Fork。
权重云盘链接、真实成员及组号、正式 PR 待补齐。问卷及邮件由真实成员处理。
