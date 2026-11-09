"""Build final paired-result tables and a speaker-cluster uncertainty figure."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def ratio(value):
    counts = f"{value['successes']}/{value['denominator']}"
    return counts + (f" ({value['rate']*100:.1f}%)" if value['rate'] is not None else ' (N/A)')


def main():
    evidence = ROOT / 'docs/evidence/2026-11-09-staged/final'
    read = lambda name: json.loads((evidence / name).read_text(encoding='utf-8'))
    analysis, selection, protocol = read('analysis.json'), read('selection.json'), read('protocol.json')
    native_audit = read('native-validation-audit.json')
    archive_verification = read('archive-verification.json')
    production = read('production-registry-check.json')
    chosen = analysis['selected_profile']
    profiles = ['previous_feedback_snr20', chosen]
    primary = analysis['evaluations']['ge2e_small_en']
    secondary = analysis['evaluations']['ecapa_base']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure_dir = ROOT / 'docs/figures/2026-11-09-staged'
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.7), constrained_layout=True)
    colors = ['#8B97A6', '#186B9C']
    for ax, endpoint, title in zip(axes, ['strict_joint', 'paper_full_screen'], ['Identity + retained quality', 'Eq.10: full clean controls']):
        for index, profile in enumerate(profiles):
            point = primary[profile]['all'][endpoint]
            if point['rate'] is None:
                ax.text(index, .1, 'N/A', ha='center'); continue
            ax.bar(index, point['rate'], width=.48, color=colors[index])
            ci_key = 'baseline_95' if index == 0 else 'staged_95'
            low, high = primary['speaker_cluster_bootstrap'][endpoint][ci_key]
            ax.plot([index, index], [low, high], color='#19212B', linewidth=1.3)
            ax.plot([index-.08, index+.08], [low, low], color='#19212B', linewidth=1.3)
            ax.plot([index-.08, index+.08], [high, high], color='#19212B', linewidth=1.3)
            ax.text(index, high+.045, f"{point['successes']}/{point['denominator']}", ha='center', fontsize=10)
        if endpoint == 'paper_full_screen':
            ax.axhline(.8006, color='#A35124', linestyle='--', linewidth=1)
        ax.set_xticks([0, 1], ['Previous', 'Staged'], fontsize=10)
        ax.set_yticks([0, .2, .4, .6, .8, 1], ['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=10)
        ax.set_xlim(-.6, 1.6); ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=11); ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
        for spine in ('top', 'right', 'left'): ax.spines[spine].set_visible(False)
    fig.suptitle('Independent GE2E: 20 new utterances, 10 enrolled speakers', fontsize=12)
    fig.savefig(figure_dir / 'paired-protection.png', dpi=180)
    fig.savefig(figure_dir / 'paired-protection.svg')
    plt.close(fig)
    lines = ['<!-- FINAL_STAGED_RESULTS -->', '## 开发选择与完整验证结果', '',
        f"初轮开发 24 条和追加软掩蔽比较 18 条均完整结束后，按预先规则选出唯一配置 {chosen}。追加开发只使用原三句，二十句新验证未查看；初轮协议、选择和结果保留。最终选择文件保存于验证开始前，validation_examined=false。该配置的开发严格联合为 {selection['selected']['strict_joint']}/6；开发三句、两个内容源是相关重复。", '',
        '独立新验证完整完成：二十句、十位已注册说话人、两个内容源、两个方法，共八十个方法/句子/内容源记录。保护音质和 WER 按唯一保护原音去重；防护率以转换对为单位，同时按内容源报告。', '',
        '| 指标 | 上轮方法（同句重跑） | 本轮选定方法 | 论文表 2 参考 |', '|---|---:|---:|---:|']
    dev_table = ['', '| 追加开发候选（同三句） | GE2E 严格联合 | 保护质量通过 | MOS 均值 | small.en WER |',
        '|---|---:|---:|---:|---:|']
    for candidate in selection['all_candidates']:
        dev_table.append(f"| {candidate['profile']['name']} | {candidate['strict_joint']}/6 | {candidate['protected_quality']}/3 | {candidate['mean_mos']:.3f} | {candidate['mean_wer']:.2%} |")
    dev_table += ['', '候选比较始终使用同三句开发语音。软掩蔽的所有失败尝试、强阶段回退和音频保留；没有把未通过质量的样例从分母中移除。', '']
    lines[4:4] = dev_table
    before, after = primary[profiles[0]]['all'], primary[chosen]['all']
    for title, key, formatting, paper in [
        ('原始 MOSNet 预测均值', 'protected_mos_mean', '.3f', '4.07'),
        ('small.en 逐句 WER', 'protected_wer_macro', '.2%', '7.84%（非同 ASR）'),
        ('实际完整波形 SNR', 'mean_snr_db', '.2f', '不直接对比')]:
        lines.append(f"| {title} | {format(before[key], formatting)} | {format(after[key], formatting)} | {paper} |")
    lines += [f"| GE2E 原音身份保留 | {before['protected_identity_retained']}/20 | {after['protected_identity_retained']}/20 | 99.17%（检查点不同） |",
              f"| GE2E 原音质量门槛通过 | {before['protected_quality_count']}/20 | {after['protected_quality_count']}/20 | 不直接对比 |", '',
              '### 防护结果与配对分母', '',
              '| 评估组合 / 方法 | 严格身份联合 | Eq.10 身份控制 | Eq.10 完整干净控制 |', '|---|---:|---:|---:|']
    for label, evaluation in [('GE2E / small.en（主）', primary), ('ECAPA / base（次）', secondary)]:
        for profile in profiles:
            row = evaluation[profile]['all']
            lines.append(f"| {label} / {profile} | {ratio(row['strict_joint'])} | {ratio(row['paper_identity_screen'])} | {ratio(row['paper_full_screen'])} |")
    secondary_selected = secondary[chosen]['all']
    lines += ['', f"次评估 ECAPA/base 的身份控制口径为 {ratio(secondary_selected['paper_identity_screen'])}，但其完整干净控制口径为 {ratio(secondary_selected['paper_full_screen'])}，且 ECAPA 参与了目标人物选择。较高的身份控制点估计不能用来宣布原文 80.06% 的总体质量复现；应同时查看独立 GE2E、完整控制覆盖率和下述不确定性。", '']
    lines += ['', '完整干净控制要求零扰动转换本身满足身份、MOS 和内容门槛。两个方法共用同一组干净控制；失败的保护原音不会从分母中移除。严格联合只计转换身份变化，Eq.10 允许转换身份、自然度或内容任一失败。N/A 与 0% 不混用。', '',
        '![同句配对防护比较](figures/2026-11-09-staged/paired-protection.png)', '',
        '图 1：独立 GE2E 评估。误差线为按说话人整组重采样的描述性 95% 区间，重复内容源不作为独立说话人。仅右图标出论文 Eq.10 防护 80.06% 参考虚线；设置和检查点不同，不代表可直接进行等价性检验。左图身份联合是额外的更严格终点。', '',
        '| 主评估 / 内容源 | 上轮严格联合 | 本轮严格联合 | 上轮完整控制 | 本轮完整控制 |', '|---|---:|---:|---:|---:|']
    for content in protocol['evaluation_contents']:
        b, a = primary[profiles[0]]['by_content'][content], primary[chosen]['by_content'][content]
        lines.append(f"| {content} | {ratio(b['strict_joint'])} | {ratio(a['strict_joint'])} | {ratio(b['paper_full_screen'])} | {ratio(a['paper_full_screen'])} |")
    lines += ['', '### 配对变化与泛化边界', '',
        '| GE2E 主终点 | 两法都失败 | 仅上轮通过 | 仅本轮通过 | 两法都通过 |', '|---|---:|---:|---:|---:|']
    for endpoint in ('strict_joint', 'paper_full_screen'):
        r = primary['paired_transitions'][endpoint]
        lines.append(f"| {endpoint} | {r['both_fail']} | {r['baseline_only']} | {r['staged_only']} | {r['both_pass']} |")
    low, high = primary['speaker_cluster_bootstrap']['strict_joint']['paired_difference_95']
    lines += ['', f"GE2E 严格联合的配对点差为 {(after['strict_joint']['rate']-before['strict_joint']['rate'])*100:.2f} 个百分点；按十位说话人整组重采样的描述性 95% 区间为 [{low*100:.2f}, {high*100:.2f}] 个百分点。区间跨过零，本样本不能证明稳定改进。", '']
    lines += ['', '这些是四十个相关转换对的变化，包含未通过干净筛选的共同失败，不能作为四十个独立人物的统计检验。十位说话人是有目的抽取的注册人物；按说话人重采样的区间只能辅助描述本样本的不确定性。', '',
        '| GE2E 主评估 / 说话人组别 | 上轮严格联合 | 本轮严格联合 | 本轮完整控制 |', '|---|---:|---:|---:|']
    for group, title in [('p225_p229_includes_three_development_speakers', 'p225-p229（含三位开发人物）新句'), ('p230_p234_not_used_in_attack_development', 'p230-p234 未参与攻击开发')]:
        b, a = primary[profiles[0]]['by_group'][group], primary[chosen]['by_group'][group]
        lines.append(f"| {title} | {ratio(b['strict_joint'])} | {ratio(a['strict_joint'])} | {ratio(a['paper_full_screen'])} |")
    lines += ['', '后五位人物仍在注册库内，也可能出现在 VC 预训练中，不能称为完全未见人物泛化。ECAPA 参与目标人物选择，其结果单独作为次评估呈现，不能取代 GE2E 的独立检验。', '',
        '### 文本归一化敏感性与约束', '',
        f"在同一组 protected WAV 上，主 small.en WER 按冻结 Basic 规则为 {after['protected_wer_macro']*100:.2f}%；标准 EnglishTextNormalizer 结果为 {after['protected_english_wer_macro_sensitivity_only']*100:.2f}%。后者只用于诊断数字/单词等文本形式，不用于替换主防护门槛、挑选样例或配置。", '',
        f"本轮平均时频掩蔽超额比例 {after['mask_exceed_fraction_mean']*100:.3f}%，平均正超额 {after['mask_excess_mean_db']:.5f} dB。无论采用软惩罚还是谱投影，都不能据此声称每个时频点严格不可闻。最终预测 MOS 是自动模型结果，真人 MOS 问卷仍未采集。", '',
        '## 对原文质量目标的判断', '',
        f"本轮独立 GE2E 严格联合为 {ratio(after['strict_joint'])}，完整干净控制下的 Eq.10 防护为 {ratio(after['paper_full_screen'])}；同句基线分别为 {ratio(before['strict_joint'])}、{ratio(before['paper_full_screen'])}。音质与内容预测见上表。是否改进应以这些配对观测判断，不以修正内部注册库后的分类准确率替代防护率。", '',
        f"二十个保护原音中，质量失败的非互斥原因：身份改变 {after['protected_quality_failure_reasons_nonexclusive']['identity']} 句、MOS<3 {after['protected_quality_failure_reasons_nonexclusive']['mos']} 句、WER>10% {after['protected_quality_failure_reasons_nonexclusive']['wer']} 句。四十个保护转换中仍识别为原人物的有 {after['converted_identity_retained']} 个。这些分解说明质量损失与身份扰动迁移各自的限制，不能把任一软件测试通过当作论文质量达标。", '',
        f"ASR 门槛失败进一步分解：干净原音与保护原音都超过 10% 的有 {after['asr_quality_comparison_to_clean_original']['both_exceed_10pct']} 句；仅保护原音超过门槛的有 {after['asr_quality_comparison_to_clean_original']['only_protected_exceeds_10pct']} 句；仅干净原音超过门槛的有 {after['asr_quality_comparison_to_clean_original']['only_clean_exceeds_10pct']} 句。这一区分保留原始转录，有助于区分当前 ASR/文本规则的共同误差与新增扰动引起的内容损失，不据此修改主门槛。", '',
        '本实验修正了可验证的目标参考处理不一致，并检验了更接近原文的两阶段优化思路。原文规模、第二种 VC 模型、原作者完整检查点与真人听测仍未补齐；即便某个小样本点估计接近或超过 80%，也不能据此宣称整篇论文精确复现。失败、零分母、原始转录和所有方法结果均保留。', '',
        '## 复跑与归档', '',
        '旧完整模型包与第二轮评估器资源包提供模型权重。本轮新增 native-target-bank.npz 保存原生目标参考；数据输入、模型、代码和解码设置的 SHA 均在 protocol.json 中。新方法的目标选取使用 ECAPA 注册信息，GE2E 权重未参与梯度优化。', '',
        '本轮实际执行顺序：初轮开发与首次选择；tools/prepare_native_soft_comparison.py 冻结追加协议；追加开发与最终选择；deploy/run_native_soft_comparison.sh validation。准备工具与选择工具均拒绝覆盖已提交的冻结文件，不应在现有证据目录再次执行准备或选择。', '',
        '复核已有结果，先在隔离目录核验并解压新证据包。重新推理时应在独立工作副本恢复模型、原始 VCTK 输入和 native-target-bank.npz，保留已提交的最终协议与 selection.json，以 bash deploy/run_native_soft_comparison.sh validation 运行固定配置；已有完整状态不会覆盖。任何研究性改参数都应新建协议和输出目录，并使用新的验证句；固定随机种子不保证不同 CUDA 内核环境的逐字节一致性。', '',
        '离线审核：python tools/analyze_staged_recovery.py --evidence docs/evidence/2026-11-09-staged/final --run <解压目录>/outputs/staged_recovery/2026-11-09/final。工具会拒绝不完整记录、重复样本、错误配置或不同干净对照。', '',
        '独立证据包为 outputs/staged_recovery/final-staged-evidence.tar.gz，包含实际导出 WAV、评分/转录、原生参考、日志与执行源码快照。strong.wav 是强扰动阶段的中间文件，不等于最终防护成功；主表只用最终 defended.wav。旧归档保持原样。ECS 未自动关机，空闲仍可能计费。', '']
    audit_lines = ['### 原生编码器与独立声纹的差异', '',
        '验证全部结束后，仅对已经评分的 WAV 追加原生 Es 分类诊断，没有改音频、选择或主分数。原生参考来自实际裁剪后的三十句注册均值。', '',
        '| 方法 | 原生 Es：保护原音仍原人物 | 原生 Es：保护转换仍原人物 | GE2E：保护转换仍原人物 | 仅原生判变身份 |',
        '|---|---:|---:|---:|---:|']
    for profile in profiles:
        a = native_audit['summaries'][profile]
        audit_lines.append(f"| {profile} | {a['native_protected_original_identity_retained']}/{a['utterances']} | {a['native_protected_converted_identity_retained']}/{a['conversion_pairs']} | {a['ge2e_protected_converted_identity_retained']}/{a['conversion_pairs']} | {a['native_only_converted_identity_changed']} |")
    audit_lines += ['', '最后一列统计内部原生编码器认为转换身份已变、独立 GE2E 仍认作原人物的转换对。内部分类的改变不能直接替代外部保护效果；反向分歧与完整相似度也保存在诊断 JSON 中。这是验证后的机制描述，不参与调参。', '']
    lines[lines.index('## 对原文质量目标的判断'):lines.index('## 对原文质量目标的判断')] = audit_lines
    lines += [f"证据包本机核验：SHA256 {archive_verification['sha256']}；共 {archive_verification['all_files_verified']} 个归档成员逐项核验，{archive_verification['measured_wavs_verified_against_scores']} 条 WAV 文件路径与评分哈希一致，合计 {archive_verification['wav_files']} 个 WAV，含强阶段中间音频。相同字节在不同路径中可能重复，这些计数不作为独立统计样本数。", '']
    lines += [f"修正后的常规注册构建器另外输出新库，二十人各三十句的全部原型与冻结原生目标银行核对通过，最大嵌入绝对误差 {production['max_abs_embedding_error']:.8g}。新库未替换冻结实验的旧注册文件；该工程修复不改变前述分数。三十九项单测全部通过，日志位于最终证据目录。", '']
    doc = ROOT / 'docs/15_两阶段与原生参考修正.md'
    prefix = doc.read_text(encoding='utf-8').split('<!-- FINAL_STAGED_RESULTS -->')[0]
    prefix = prefix.replace('当前开发仍在运行；最终结论以完整状态文件和后续结果为准。', '开发、全局选择及配对新验证均已完整结束，最终结果见下文。')
    doc.write_text(prefix.rstrip() + '\n\n' + '\n'.join(lines), encoding='utf-8')
    write_json(evidence / 'report-inputs.json', {'analysis_sha256': file_sha256(evidence / 'analysis.json'),
        'report_builder_sha256': file_sha256(__file__),
        'selection_sha256': file_sha256(evidence / 'selection.json'), 'native_audit_sha256': file_sha256(evidence / 'native-validation-audit.json'),
        'figure_png_sha256': file_sha256(figure_dir / 'paired-protection.png'),
        'markdown_sha256': file_sha256(doc)})
    print(doc)


if __name__ == '__main__': main()
