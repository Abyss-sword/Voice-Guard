"""Build the final course-facing report from audited study summaries."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]

def pct(x):
    return f"{100*x:.1f}%" if x is not None else "-"

def main():
    d=json.loads((ROOT/'docs/evidence/2026-11-09-discriminant/analysis.json').read_text())
    b=json.loads((ROOT/'docs/evidence/2026-11-09-discriminant-blend/analysis.json').read_text())
    v=b['splits']['validation']['profiles']; dev=b['splits']['development']['profiles']
    q=json.loads((ROOT/'docs/evidence/2026-11-09-vqmivc/analysis.json').read_text())
    lines=['# Voice Guard 复现、质量改进与云端验证报告','',
        '## 结论摘要','',
        '本项目在华为云 Tesla T4 上完成了原始 Voice Guard 管线、原生注册判别间隔改进、固定全局混合续研以及官方 VQMIVC 的有限迁移诊断。最终采用的统一 50% 混合方案在预留 20 句验证中，完整干净控制下的联合防护由配对基线 14/23（60.9%）提高到 15/23（65.2%）；严格身份联合由 44/69（63.8%）提高到 46/69（66.7%）。保护原音质量通过数均为 18/20，small.en WER 均为 0.42%，预测 MOS 从 3.926 降至 3.906。该结果支持本项目的受控改进，但仍低于论文表 2 的整体 80.06% 参考，不能宣称论文规模等价复现。','',
        '## 复现对象与资源','',
        '论文为 Voice Guard: Protecting Voice Privacy with Strong and Imperceptible Adversarial Perturbation in the Time Domain（IJCAI 2023）。项目固定 AdaIN-VC、原生 speaker encoder、心理声学掩蔽、两阶段时域优化和独立 GE2E/MOSNet/Whisper 评分。华为云实例为 Ubuntu 22.04、Tesla T4、约 32GB 内存；所有实验使用固定协议、输入摘要、随机种子和 PCM16 导出检查。','',
        '论文表 2 的 AdaIN 白盒参考为防护 80.06%、保护原音预测 MOS 4.07、WER 7.84%、CER 3.05%、ASV 99.17%。论文 MOS 是自动预测指标；真人听测只能作为额外证据。本项目的注册库、权重、VC 预处理和评分实现未证实与作者全部相同。','',
        '## 改进方法','',
        '首先在二十位注册人物各三十句注册语音上拟合带 0.2 球形收缩的 LDA，把原生 speaker embedding 转成本人相对最高其他人的判别间隔。防护从原生 PCM16 基线开始，在保护参考和两个固定内容的解码后梅尔再编码路径上优化平滑最差间隔；wav2vec2 CTC 只作为内容梯度约束，ECAPA 只作原音代理检查。GE2E、MOSNet、Whisper 不参与梯度和候选接受。','',
        '原预算判别候选开发集为完整联合 10/16、严格身份联合 24/37，而基线为 8/16、17/37；但预测 MOS 4.101 相比 4.164 下降超过事前 0.05，因此禁止直接进入新句验证。之后固定候选与基线作全局波形混合，唯一选择为 50%。','',
        '| 研究 | 完整联合 | 严格身份联合 | 原音质量 | 预测 MOS | WER |','|---|---:|---:|---:|---:|---:|',
        f"| 判别间隔原预算开发 | {pct(d['splits']['development']['profiles']['lda_margin_snr20']['all']['paper_full_screen']['rate'])} | {pct(d['splits']['development']['profiles']['lda_margin_snr20']['all']['strict_joint']['rate'])} | {d['splits']['development']['profiles']['lda_margin_snr20']['all']['protected_quality_count']}/12 | {d['splits']['development']['profiles']['lda_margin_snr20']['all']['protected_mos_mean']:.3f} | {d['splits']['development']['profiles']['lda_margin_snr20']['all']['protected_wer_macro']:.2%} |",
        f"| 全局混合开发基线 | {pct(dev['native_emb_snr20']['all']['paper_full_screen']['rate'])} | {pct(dev['native_emb_snr20']['all']['strict_joint']['rate'])} | {dev['native_emb_snr20']['all']['protected_quality_count']}/12 | {dev['native_emb_snr20']['all']['protected_mos_mean']:.3f} | {dev['native_emb_snr20']['all']['protected_wer_macro']:.2%} |",
        f"| 全局混合 50% 开发 | {pct(dev['lda_blend50']['all']['paper_full_screen']['rate'])} | {pct(dev['lda_blend50']['all']['strict_joint']['rate'])} | {dev['lda_blend50']['all']['protected_quality_count']}/12 | {dev['lda_blend50']['all']['protected_mos_mean']:.3f} | {dev['lda_blend50']['all']['protected_wer_macro']:.2%} |",
        f"| 全局混合基线新句 | {pct(v['native_emb_snr20']['all']['paper_full_screen']['rate'])} | {pct(v['native_emb_snr20']['all']['strict_joint']['rate'])} | {v['native_emb_snr20']['all']['protected_quality_count']}/20 | {v['native_emb_snr20']['all']['protected_mos_mean']:.3f} | {v['native_emb_snr20']['all']['protected_wer_macro']:.2%} |",
        f"| 全局混合 50% 新句 | {pct(v['lda_blend50']['all']['paper_full_screen']['rate'])} | {pct(v['lda_blend50']['all']['strict_joint']['rate'])} | {v['lda_blend50']['all']['protected_quality_count']}/20 | {v['lda_blend50']['all']['protected_mos_mean']:.3f} | {v['lda_blend50']['all']['protected_wer_macro']:.2%} |",'',
        '新句结果按固定人物和内容配对，完整联合分母分别为干净转换通过身份与质量的句-内容对，严格身份联合分母为干净身份通过的句-内容对。按人物聚类的描述性 95% 区间和逐句转移表保存在 analysis.json；这 10 位注册人物不是总体随机样本。','',
        '## 第二 VC 迁移诊断','',
        f"固定官方 VQMIVC 提交 851b4f5ca5bb60c11fea6a618affeb4979b17cf3，6 个开发参考句、2 个内容、3 种方法共 36 条记录。干净控制中身份失败 6 条、质量失败 6 条，有效完整联合分母为 0；因此不能报告 VQMIVC 的完整联合防护率。严格身份联合为基线 1/6、原预算判别 1/6、放宽预算判别 3/6。模型只作黑盒迁移诊断，没有参与优化或方法选择。",'',
        '已知干净样例还显示，直接使用原始 22050Hz 参考时 MOS 3.494；使用 AdaIN 导出的 24kHz PCM16 参考时 MOS 2.554。另一项固定搬运诊断把原始参考重采样到 16kHz，再搬运 24kHz 防护差分；干净质量分母仍为 0。两者说明输入通道和基础转换质量需要优先修复，但不能单独证明某个环节的因果责任。','',
        '## 掩蔽常数核对','',
        '论文 PDF 附录 Eq.17 使用 -0.6025、Eq.13 没有 -12dB；作者公开代码使用 -6.025 和额外 -12dB。历史实现继续匹配公开代码，没有静默替换。单独诊断显示改用论文字面常数会令功率上限提高约 5.42-12dB、幅度上限放大约 1.87-3.98 倍；这只是敏感性结果，不能冒充论文完整实验。','',
        '## 证据与复核','',
        '主判别研究归档为 outputs/discriminant/discriminant-evidence.tar.gz；全局混合研究保留 development/validation 原始结果、状态、选择、门槛和审核统计；VQMIVC 代码、资源、转换音频和路径诊断分别归档。历史课程、联合质量和内容鲁棒归档保持原 SHA256。所有新句和第二 VC 研究均保留失败结果，脚本不会自动关机。','',
        '建议复核顺序：先核对 docs/evidence/2026-11-09-discriminant-blend/analysis.json 的 validation 配对统计，再核对 VQMIVC 的 clean_control_failures 和 transport 诊断，最后使用对应 archive receipt 逐项解压校验。','',
        '## 局限与后续','',
        '当前仍没有论文原始二十人各百句、两种 VC、双向迁移的完整同权重实验，也没有真人 MOS 研究。下一步应先解决 VQMIVC 干净转换质量和参考采样率通道，再扩大说话人和内容规模；不应通过放宽质量门槛、删除失败句或事后挑选候选来追赶单一防护百分比。']
    path=ROOT/'docs/20_最终复现总结.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(path)

if __name__=='__main__':main()
