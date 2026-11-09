# 严格掩蔽投影与防御侧 EOT（L1/L2 本地探索）

> 2026-11-09 本地实验（RTX 4060 8GB）。两项都是"论文之外的工程探索"，
> 用于量化论文"强且不可感知"的边界，不作为复现结论。

## L2：严格掩蔽投影——"不可感知"的真实代价

**动机**：论文的掩蔽损失是软 hinge（`mean(ReLU(PSD_δ − θx))`），实测保护音频在
ε=0.05 时约 88.7% 的时频格仍高于掩蔽阈（均值超 29 dB）。本实验把扰动**硬投影**到
θx 之下，量化严格不可感知要求的幅度代价与防御代价。

**方法**（`vguard/masking.py`）：
1. `masking_exceedance()`：逐格计算 PSD_δ 与 θx 的差值，统计超阈率；
2. `project_masked_delta_hard()`：逐格幅度钳位 + iSTFT 重建，迭代 60 次；
3. `project_to_mask_exact()`：按最紧违规格一次性缩放扰动，使**每格**都不超阈。

**结果**（5 句，`outputs/exp03_eps005`，ε=0.05，ECAPA 身份维）：

| 版本 | 超阈率 | 扰动缩放 | SNR | F_imp | F_str | 防御（F_imp∧F_str） |
|---|---:|---:|---:|---:|---:|---:|
| 原始 VG（软 hinge） | **88.7%** | 1.0 | 3.2–4.4 dB | 5/5 | — | （原实验 3/4） |
| 硬投影 60 次迭代 | **6.1%** | 1.0 | 3.2–4.4 dB | 5/5 | 3/5 | **3/5** |
| 严格缩放（每格不超阈） | **≈0** | **1.0e-4 ~ 3.1e-4** | **70–82 dB** | 5/5 | 1/5 | **1/5** |

**结论（如实）**：
- 迭代硬投影能把超阈率从 88.7% 压到 6.1%，防御率基本保持（3/5）——这是可行的工程折中；
- 但要做到"每个时频格都不超过掩蔽阈"，扰动必须缩放到原幅度的 **~1e-4**（SNR 76 dB 量级），
  此时转换后身份几乎不再被改变（防御 1/5）——**严格不可感知与防御强度在本设置下直接冲突**；
- 这解释了论文为何用软 hinge 且只报 MOS：软约束在感知上"够用"，但不是严格不可感知。

产物：`outputs/exp03_eps005/hard_masking.json` + 每句 `defended_hard.wav` / `defended_exact.wav`。

复现命令：

```powershell
.\.venv\Scripts\python.exe scripts\hard_masking_eval.py --exp-dir outputs\exp03_eps005 `
  --max-iters 60 --evaluate --device cuda
```

## L1：防御侧 EOT（信道后仍保持 F_imp 的正则）

**动机**：W6 加分项发现"攻击侧 EOT 使防御全面变差"（负面结果）。本项探索相反方向：
在阶段2 损失中加入"**信道化后的保护音频仍应被判为原说话人**"的正则项（F_imp 信道稳健），
期望改善真实信道（G.711/8kHz 等）下的防御存活率。

**方法**：`vguard/attack.py` 的 `_stage2` 增加 `imp_channel_fn / imp_weight`：

```
loss = L_vc + c·L_θ + w·MSE(Es(C(x+δ)), Es(x))      # C = 可微信道（重采样/低通/量化/增益）
```

`scripts/run_experiment.py --imp-eot --imp-eot-weight 0.5` 开启；
`scripts/run_defense_eot.py` 自动跑"基线 vs 防御侧 EOT"两组 + 6 信道扫描 + 对比表。

**状态**：实验运行中（3 句 × 400/200 步试点；800/400 全量因 c 二分搜索耗时过长而缩为试点）。
结果表待补充。

命令：

```powershell
.\.venv\Scripts\python.exe scripts\run_defense_eot.py --speakers p225,p226,p227 `
  --iters1 400 --iters2 200 --imp-weight 0.5 --out-root outputs\l1_defense_eot_v2 --device cuda
```
