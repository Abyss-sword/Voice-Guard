"""Append reproducible final recovery tables to the second-round report."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/2026-11-09-quality'


def rate(value):
    count = f"{value['successes']}/{value['denominator']}"
    return count + (f" ({value['rate']*100:.1f}%)" if value['rate'] is not None else ' (N/A)')


def main():
    data = json.loads((EVIDENCE / 'recovery-analysis.json').read_text(encoding='utf-8'))
    selection = json.loads((EVIDENCE / 'recovery-selection.json').read_text(encoding='utf-8'))
    primary = data['evaluations']['ECAPA_Whisper_base']['all']['pooled_correlated_pairs']
    secondary = data['evaluations']['GE2E_Whisper_small_en']['all']['pooled_correlated_pairs']
    lines = ['<!-- FINAL_RECOVERY_RESULTS -->', '## 冻结验证最终结果', '',
        '开发三句、三配置、两个步数共 18 条已完成。按预先声明规则选取全局反馈目标、20 dB SNR 预算、1500 步。选择后才开始十五句验证，未因验证结果更换配置。', '',
        f"所选配置开发严格联合 {selection['selected_development_summary']['strict_joint']}/3；开发保护质量 3/3。开发结果用于选择，不能替代独立验证。", '',
        '验证完整完成：15 个保护原音、10 位说话人、两个内容源，共 30 对干净/保护转换。音质与逐句 WER 对保护 WAV 去重，防护率逐内容源列出；两个内容源是相关重复。', '',
        '| 保护原音指标 | 本轮结果 | 论文表 2 |', '|---|---:|---:|',
        f"| 原始 MOSNet 预测均值 | {primary['protected_mos_original_mean']:.3f} | 4.07 |",
        f"| Whisper base 逐句 WER 均值 | {primary['protected_wer_macro']*100:.2f}% | 7.84% |",
        f"| small.en 逐句 WER 均值 | {secondary['protected_wer_macro']*100:.2f}% | 非同评估器 |",
        f"| ECAPA 身份保留 | {primary['protected_identity_retained']}/15 | 非同评估器 |",
        f"| GE2E 身份保留 | {secondary['protected_identity_retained']}/15 | 非同检查点 |",
        f"| 实际完整波形平均 SNR | {primary['mean_snr_db']:.2f} dB | 不直接对比 |", '',
        f"干净原音原始 MOSNet 均值 {primary['clean_mos_original_mean']:.3f}，Whisper base 逐句 WER {primary['clean_wer_macro']*100:.2f}%。这些是自动模型预测，尚无真人 MOS 问卷结果。", '',
        '### 防护判据与干净控制', '',
        '严格联合要求保护原音身份、MOS、内容都保留，同时转换身份改变。Eq.10 结构允许转换身份改变、MOS<3、WER>10% 任一成立。完整干净控制额外要求零扰动转换本身通过身份、MOS 和 WER；分母为零时写 N/A。', '',
        '| 评估组合 / 内容源 | 严格联合 | Eq.10 结构 | 完整干净控制 |', '|---|---:|---:|---:|']
    for label, evaluation in data['evaluations'].items():
        pooled = evaluation['all']['pooled_correlated_pairs']
        lines.append(f"| {label} / 合并相关转换对 | {rate(pooled['strict_joint'])} | {rate(pooled['paper_structure_identity_screen'])} | {rate(pooled['paper_structure_full_screen'])} |")
        for content, row in evaluation['all']['by_content'].items():
            lines.append(f"| {label} / {content} | {rate(row['strict_joint'])} | {rate(row['paper_structure_identity_screen'])} | {rate(row['paper_structure_full_screen'])} |")
    lines += ['', '分母仅纳入相应干净控制合格的转换；分子始终同时要求保护原音身份及质量合格。不能把既有的转换内容错误视为扰动的因果贡献。论文 80.06% 是不同实验规模、语音转换和评估设置下的值，本表不构成数值等价复现。', '',
        '### 两组说话人的结果', '',
        '| 评估组合 / 组别 | 句数 | 严格联合（相关转换对） | Eq.10（相关转换对） |', '|---|---:|---:|---:|']
    for label, evaluation in data['evaluations'].items():
        for cohort, title in [('development_speakers_new_sentences', 'p225-p229 新句'), ('speakers_not_used_in_attack_development', 'p230-p234 未用于攻击开发')]:
            row = evaluation[cohort]['pooled_correlated_pairs']
            lines.append(f"| {label} / {title} | {row['utterances']} | {rate(row['strict_joint'])} | {rate(row['paper_structure_identity_screen'])} |")
    lines += ['', '后五位说话人仍在注册库中，也可能出现在转换器预训练数据中，因此这里只能声称未用于本轮攻击开发，不能声称完全未见说话人泛化。', '',
        '### 失败样例与约束残差', '',
        '唯一未通过保护原音质量门槛的句子是 p227_053。参考为 "Four other charges were dropped."；base 对干净和保护音频均输出 "Foot of the charges were dropped."，WER 均为 60%。small.en 对干净音频输出正确的 Four，对保护音频输出数字 4，因此按冻结文本规则计 WER 20%。这说明机器转录门槛含识别器及文本形式因素，不能完全替代真人可懂度；本轮未据此改变归一化或重新计分。原始观测全部保留。', '',
        f"PCM16 导出后，平均时频单元掩蔽超额比例为 {primary['mask_exceed_fraction_mean']*100:.3f}%，平均正超额为 {primary['mask_excess_mean_db']:.5f} dB。投影有残差，不能宣称每个时频点严格满足掩蔽阈值。全局 SNR 与峰值预算另行约束。", '',
        '## 结论与未达到的目标', '',
        f"本轮保护 MOS 预测 {primary['protected_mos_original_mean']:.3f}，但严格防护与完整干净控制下的证据仍不足以支持论文级总体复现。保护原音高音质和内容保留，是必要条件，不是联合防护成功的充分条件。", '',
        '干净 AdaIN-VC 输出本身常有内容识别错误，已分别保留 base 和 small.en 原始转录。不得加入参考文本提示、挑选逐句更有利的评估器或更换验证配置来提高分数。另一次研究若要修正转换器或引入神经声码器，应使用新协议、新开发集及新验证集，并与本轮结果分开。', '',
        '本轮没有同十五句的原 VG 对照，因此不能用旧五句/十句的不同数据均值宣称新算法因果改善。论文规模、多转换器泛化、真人听测、原作者未公开实现和精确评估检查点也未补齐。所有失败均留存。', '',
        '## 复跑、资源与证据', '',
        '新增推理依赖见 deploy/quality-recovery-requirements.txt；在原部署环境中安装。tools/download_models.py 支持 mosnet-original（固定提交、校验 SHA）。tools/download_quality_asr.py 下载固定 small.en 权重；GE2E 来自固定 Resemblyzer 0.1.4，注册库使用二十人各前三十句，资源与六百输入哈希见 ge2e-resources.json。旧模型压缩包不包含这些新增资源；本轮另提供 VoiceGuard-quality-evaluator-resources.zip（约 515 MB），含恢复说明、原始 MOSNet、small.en、GE2E 权重及注册库，仍需与旧完整模型包配合。尚无真实云盘分享链接。', '',
        'tools/verify_original_mosnet.py 支持 PyTorch、隔离 TensorFlow 和 compare 三种模式；正常推理不依赖 TensorFlow。冻结协议校验源码、资产和选择文件。原开发/验证 runner、补评 runner 和协议不应修改后冒充原实验。', '',
        '离线分析：python tools/analyze_quality_recovery.py --root <解压目录>/outputs/paper_quality/2026-11-09-recovery。分析工具拒绝未完成、缺失、重复或错误配置的验证结果。原始表格、转录、全部 WAV、执行源码快照和注册库保存在独立 final-recovery-evidence.tar.gz；旧 final-complete-evidence.tar.gz 保留。', '',
        '本轮补充报告以本文件与 recovery-analysis.json 为依据。原课程报告 PDF 和九分钟 PPT 是上一轮交付，不含此轮新增结果，应与本补充报告一起阅读。ECS 未自动关机，空闲时仍可能计费。', '',
        '证据索引：recovery-protocol.json、recovery-selection.json、supplemental-protocol.json、mosnet-parity.json、original-mosnet-audit.json、archived-paper-criteria.json、recovery-analysis.json、recovery-archive-verification.json。全部位于 docs/evidence/2026-11-09-quality/。', '']
    path = ROOT / 'docs/14_论文质量追赶.md'
    prefix = path.read_text(encoding='utf-8').split('<!-- FINAL_RECOVERY_RESULTS -->')[0]
    prefix = prefix.replace('实验仍在运行，最终结果以完整状态文件与后续表格为准。', '开发、冻结验证及独立补评均已完整结束，结果见下文。')
    path.write_text(prefix.rstrip() + '\n\n' + '\n'.join(lines), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
