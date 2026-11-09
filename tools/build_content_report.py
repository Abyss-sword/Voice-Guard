"""Generate the content-supervised study report from verified local evidence."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_joint_report import ratio
from vguard.run_state import file_sha256, write_json

LABELS = {'native_emb_snr20': '原生嵌入基线', 'ctc_tf_snr20': 'CTC＋时频分配',
    'ctc_tf_eot_snr20': 'CTC＋时频分配＋环境变化'}


def main():
    evidence = ROOT / 'docs/evidence/2026-11-09-content-robust'
    read = lambda name: json.loads((evidence / name).read_text(encoding='utf-8'))
    protocol, analysis, decision = [read(name) for name in
        ('protocol.json', 'analysis.json', 'extension-decision.json')]
    channels, archive = read('channel-analysis.json'), read('archive-verification.json')
    selection, test_record = read('selection.json'), read('test-verification.json')
    names = [protocol['baseline']['name']] + [p['name'] for p in protocol['profiles']]
    dev = analysis['splits']['development']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = ROOT / 'docs/figures/2026-11-09-content-robust'
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.9), constrained_layout=True)
    for ax, endpoint, title in zip(axes, ('paper_full_screen', 'strict_joint'),
            ('Complete clean controls', 'Identity controls')):
        for i, name in enumerate(names):
            point = dev['profiles'][name]['all'][endpoint]
            if point['rate'] is None:
                ax.text(i, .05, 'N/A', ha='center'); continue
            ax.bar(i, point['rate'], color=['#8997a4', '#167599', '#438869'][i], width=.55)
            ax.text(i, point['rate'] + .035, f"{point['successes']}/{point['denominator']}", ha='center', fontsize=9)
        ax.set_xticks([0, 1, 2], ['Baseline', 'CTC + TF', 'CTC + TF + EOT'])
        ax.set_ylim(0, 1.12); ax.set_yticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%'])
        ax.set_title(title, fontsize=11); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        for spine in ('top', 'right', 'left'): ax.spines[spine].set_visible(False)
    fig.suptitle('Observed development only: 12 utterances, 10 enrolled speakers', fontsize=11)
    fig.savefig(directory / 'development.png', dpi=180)
    fig.savefig(directory / 'development.svg'); plt.close(fig)
    lines = ['# Voice Guard 内容监督与环境鲁棒优化', '',
        '本报告研究如何在保留原音内容的条件下改进语音防克隆。它是独立实验扩展，保留既有基线、协议与全部负结果。用户提供的 Kalypta/Treble 博客仅作设计灵感，不作为已核实的算法或产品性能证据。', '',
        '## 主要结果', '',
        '十二个此前已观察的句子被明确划作开发集，十位注册人物、四个固定内容源、三个方法，共 144 条相关转换记录。开发结果不能当作未知测试集成绩。', '',
        '| 方法 | 完整联合 | 严格身份联合 | 原音质量 | 预测 MOS | WER |',
        '|---|---:|---:|---:|---:|---:|']
    for name in names:
        value = dev['profiles'][name]['all']
        lines.append(f"| {LABELS[name]} | {ratio(value['paper_full_screen'])} | {ratio(value['strict_joint'])} | {value['protected_quality_count']}/12 | {value['protected_mos_mean']:.3f} | {value['protected_wer_macro']:.2%} |")
    if decision['validation_allowed']:
        values = analysis['splits']['validation']['profiles']
        b, a = values[analysis['baseline']]['all'], values[analysis['challenger']]['all']
        lines += ['', f"开发门槛通过，随后运行二十个新句。完整联合：配对基线 {ratio(b['paper_full_screen'])}，所选候选 {ratio(a['paper_full_screen'])}。严格身份联合分别为 {ratio(b['strict_joint'])} 和 {ratio(a['strict_joint'])}。", '',
            '新验证只比较事前选定的一个候选，不能根据新分数重新选择另一种方法或导出检查点。']
    else:
        lines += ['', '开发门槛未通过，按协议停止于开发阶段，二十个预留新句没有运行评分。本轮没有新的独立句子泛化成绩，继续保留原生嵌入基线。']
    lines += ['', '![开发结果](figures/2026-11-09-content-robust/development.png)', '',
        '图 1：已观察开发句上的完整干净控制和身份控制结果，三个方法共用匹配干净控制。四个内容源属于相关重复，不是四倍独立人物。', '',
        '## 来源审核与任务区别', '',
        '用户给出的 CSDN 文章于 2026 年 11 月 5 日发布，描述 Kalypta 令机器难以转录、真人仍可听懂。本项目的课程质量门槛要求原音转录正确，因此不采用防 ASR 损失，而使用 CTC 内容保留监督。', '',
        '文章未提供可核实的 Kalypta 原始论文、官方代码、权重和完整配置；1.2M 参数、低于 12ms、识别率降至 11.3% 均不作为本项目事实。ISO 226 等响曲线也不能证明任意低于 1dB 的扰动不可听。Treble 官方介绍波动有限元和几何声学，不支持把文中麦克斯韦方程叙述当成声学仿真依据。来源审核摘要保存在 research-review.json。', '',
        '## 方法与可执行约束', '',
        '每句先运行相同的原生嵌入基线：3000 步强扰动、1500 步隐藏，最终 SNR 下限 20dB。两个候选从同一 PCM16 基线起步，再各优化 160 步。防护目标仍是 AdaIN 原生参考编码器的目标人物吸引和本人排斥，不反传最终输出的 Griffin-Lim 相位梯度。', '',
        '内容监督使用冻结的 torchaudio WAV2VEC2_ASR_BASE_960H，以真实参考文本计算 CTC 损失，在完整参考上下文中反传到保护裁剪。训练字符归一化仅服务该检查点，不改变冻结 Basic 归一化或最终 WER<=10% 门槛。GE2E、MOSNet 和两种 Whisper 都不参与梯度或音频接受决策。', '',
        '分别归一化身份梯度和 CTC 梯度，以两者短时频谱的相对敏感度生成 [.1,1] 权重，再做时频平滑，将更新分配给身份梯度较强、内容梯度较弱的位置。之后移除一阶上会增加 CTC 的冲突分量。该门控只是局部启发式，既不等同于可解释的声纹频带，也不保证下一步真实转录不变。', '',
        '加入环境变化的候选循环使用原始参考、两组固定房间一阶镜像反射、三抽头麦克风响应以及 8kHz 降采样/恢复。干净原音和保护原音通过同一变换后计算原生身份目标。两组房间只含一次反射，未模拟弥散混响、完整设备或实测平台链路，不称为完整 Treble 或 pyroomacoustics 仿真。', '',
        '每八步检查一次训练代理：CTC 不高于基线+0.01；原音 ECAPA 相似度不低于基线-0.005；相对频谱距离不高于基线+0.01；原始条件身份目标不高于基线+0.005；环境平均目标不高于上个接受点+0.0001。失败则退回上个接受波形并减小步长。SNR、掩蔽、峰值与余量继续投影。', '',
        '实际 PCM16 导出后、独立评分之前，再检查相同代理上限；越限则复制本句基线 WAV。该回退是事先定义的训练算法，不是根据最终 WER/MOS/GE2E 挑选音频；即便通过代理检查，独立质量仍可能退步。', '',
        '## 代理接受与实际更新', '',
        '| 候选 | 接受块 | 拒绝块 | 梯度冲突次数 | PCM 回退 | 与基线逐字节相同 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, value in dev['proxy_diagnostics'].items():
        lines.append(f"| {LABELS[name]} | {value['accepted_blocks']} | {value['rejected_blocks']} | {value['content_gradient_conflicts']} | {value['post_export_fallbacks']} | {value['identical_to_baseline']}/12 |")
    lines += ['', '更新块和冲突次数是优化机制记录，不是独立防护成功数。CTC 是训练代理，不应将其下降写成最终 Whisper WER 已改善。', '']
    failures = [(name, row) for name, result in dev['paired_comparisons'].items()
        for row in result['new_original_quality_failures']]
    if failures:
        lines += ['基线合格、候选新增原音失败的样例：', '',
            '| 方法 / 原音 | 基线 WER | 候选 WER | 候选 MOS |', '|---|---:|---:|---:|']
        for name, row in failures:
            lines.append(f"| {LABELS[name]} / {row['utterance']} | {row['baseline_wer']:.2%} | {row['candidate_wer']:.2%} | {row['candidate_mos']:.3f} |")
        example = failures[0][1]
        lines += ['', f"例如 {example['utterance']} 原文为 “{example['reference']}”，基线转录 “{example['baseline_hypothesis']}”，候选转录 “{example['candidate_hypothesis']}”。完整转录、质量与身份判定保留在逐句证据，没有删除失败或调整归一化。", '']
    else:
        lines += ['本次开发中没有出现基线原音合格、候选原音新增质量失败的句子；这不保证其他新句也能保持。', '']
    lines += [
        '## 开发选择与停止规则', '',
        f"一次全局开发选择为 {LABELS[selection['selected']['profile']['name']]}。新验证要求完整联合严格提高、严格联合和唯一原音质量数不下降、保留每一个基线原音质量通过句，宏平均 WER 不高于基线+0.5 个百分点、预测 MOS 不低于基线-0.05。", '',
        '| 事前门槛 | 实际判定 |', '|---|---|']
    labels = {'full_joint_improved': '完整联合严格提高', 'strict_joint_not_worse': '严格身份联合不下降',
        'original_quality_count_not_worse': '原音质量通过数不下降',
        'every_baseline_passing_original_preserved': '每个基线合格原音仍合格',
        'wer_not_worse_beyond_tolerance': 'WER 在预定容差内',
        'mos_not_worse_beyond_tolerance': '预测 MOS 在预定容差内'}
    for key, value in decision['checks'].items(): lines.append(f"| {labels[key]} | {'通过' if value else '未通过'} |")
    lines += ['', '开发集有意纳入旧失败与更多人物，仍可能对既有观测过拟合。扩大开发范围有助于暴露退步，不能替代新验证。没有通过门槛时不靠换句、放宽阈值或重复选择继续消耗预留测试集。', '']
    if decision['validation_allowed']:
        validation = analysis['splits']['validation']
        before = validation['profiles'][analysis['baseline']]['all']
        after = validation['profiles'][analysis['challenger']]['all']
        comparison = validation['paired_comparisons'][analysis['challenger']]
        lines += ['## 二十个新句的独立验证', '',
            '十位已注册人物各两句、四个固定内容源、配对基线与唯一已选候选，共 160 条转换记录。句子不在开发、此前优化/验证或注册库句子中；人物可能出现在 VC 预训练中，不能称为完全未见人物。', '',
            '| 主指标 | 配对基线 | 所选候选 |', '|---|---:|---:|']
        for title, key in [('完整干净控制联合', 'paper_full_screen'),
                ('严格身份联合', 'strict_joint'), ('身份控制 Eq.10', 'paper_identity_screen')]:
            lines.append(f"| {title} | {ratio(before[key])} | {ratio(after[key])} |")
        lines += [f"| 原音质量通过 | {before['protected_quality_count']}/20 | {after['protected_quality_count']}/20 |",
            f"| 原音身份保留 | {before['protected_identity_retained']}/20 | {after['protected_identity_retained']}/20 |",
            f"| 原音预测 MOS | {before['protected_mos_mean']:.3f} | {after['protected_mos_mean']:.3f} |",
            f"| 原音 WER | {before['protected_wer_macro']:.2%} | {after['protected_wer_macro']:.2%} |",
            f"| 平均 SNR | {before['mean_snr_db']:.2f} | {after['mean_snr_db']:.2f} |", '']
        if before['paper_full_screen']['rate'] is not None:
            delta = after['paper_full_screen']['rate'] - before['paper_full_screen']['rate']
            interval = comparison['intervals']['paper_full_screen']['paired_difference_95']
            conclusion = f"完整联合点差为 {delta*100:+.2f} 个百分点。"
            if interval:
                lo, hi = interval
                conclusion += f"按人物整组重采样的描述性 95% 区间为 [{lo*100:.2f}, {hi*100:.2f}] 个百分点。"
                if lo == hi == 0:
                    conclusion += '完整联合逐对判定相同，重采样点差恒为零；这不代表总体差异已精确确定为零。'
                elif lo <= 0 <= hi:
                    conclusion += '区间包含零，尚不能证明稳定提升。'
            conclusion += '开发与新验证结果分别保留；单次小样本不构成论文总体质量复现。'
            if delta <= 0: conclusion += '新验证未建立完整联合改善，继续保留原生嵌入基线。'
            lines += [conclusion, '']
        lines += ['| 固定内容源 | 基线完整联合 | 候选完整联合 | 基线严格身份 | 候选严格身份 |',
            '|---|---:|---:|---:|---:|']
        for content in protocol['evaluation_contents']:
            b = validation['profiles'][analysis['baseline']]['by_content'][content]
            a = validation['profiles'][analysis['challenger']]['by_content'][content]
            lines.append(f"| {content} | {ratio(b['paper_full_screen'])} | {ratio(a['paper_full_screen'])} | {ratio(b['strict_joint'])} | {ratio(a['strict_joint'])} |")
        failures = comparison['new_original_quality_failures']
        lines += ['', f"基线原音合格而候选新增质量失败的句子数为 {len(failures)}。原音身份、MOS 和 WER 的失败原因分别记录，不根据验证音频重新挑选检查点或回退。", '']
        for row in failures:
            lines += [f"{row['utterance']}：原文 “{row['reference']}”；基线转录 “{row['baseline_hypothesis']}”，候选转录 “{row['candidate_hypothesis']}”，WER 从 {row['baseline_wer']:.2%} 到 {row['candidate_wer']:.2%}，预测 MOS {row['candidate_mos']:.3f}。", '']
        lines += [f"PCM16 实测掩蔽阈超额频点比例的逐句均值为基线 {before['mask_exceed_fraction_mean']:.2%}、候选 {after['mask_exceed_fraction_mean']:.2%}，正超额的全频点均值分别为 {before['mask_excess_mean_db']:.4f} 和 {after['mask_excess_mean_db']:.4f}dB。投影被实际执行，但重建一致性与 PCM16 量化后仍有残差，不能写成所有频点严格零超额。", '']
        validation_rows = json.loads((evidence / 'validation-results.json').read_text(encoding='utf-8'))['rows']
        before_rows = {(r['utterance'], r['content']): r for r in validation_rows if r['profile'] == analysis['baseline']}
        after_rows = {(r['utterance'], r['content']): r for r in validation_rows if r['profile'] == analysis['challenger']}
        for key, row in before_rows.items():
            changed = after_rows[key]
            if row['gates_ge2e']['strict_joint'] and not changed['gates_ge2e']['strict_joint']:
                lines += [f"严格身份联合的退步对为 {key[0]} / {key[1]}：基线转换被 GE2E 判为 {row['protected_vc']['asv_ge2e']}，候选转换被判为 {changed['protected_vc']['asv_ge2e']}。完整联合还允许转换音质或内容被破坏，因此严格身份退步不一定改变完整联合判定。", '']
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), constrained_layout=True)
        for ax, endpoint, title in zip(axes, ('paper_full_screen', 'strict_joint'),
                ('Complete clean controls', 'Identity controls')):
            for i, name in enumerate((analysis['baseline'], analysis['challenger'])):
                point = validation['profiles'][name]['all'][endpoint]
                if point['rate'] is None: continue
                ax.bar(i, point['rate'], width=.5, color=['#8997a4', '#167599'][i])
                interval = comparison['intervals'][endpoint]['baseline_95' if i == 0 else 'staged_95']
                if interval:
                    lo, hi = interval
                    ax.plot([i, i], [lo, hi], color='#202a32', linewidth=1.2)
                else: hi = point['rate']
                ax.text(i, min(hi + .04, 1.06), f"{point['successes']}/{point['denominator']}", ha='center', fontsize=10)
            ax.set_xticks([0, 1], ['Baseline', 'CTC + TF'])
            ax.set_ylim(0, 1.12); ax.set_yticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%'])
            ax.set_title(title, fontsize=11); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
            for spine in ('top', 'right', 'left'): ax.spines[spine].set_visible(False)
        fig.suptitle('Frozen validation: 20 fresh sentences, 10 enrolled speakers', fontsize=11)
        fig.savefig(directory / 'validation.png', dpi=180)
        fig.savefig(directory / 'validation.svg'); plt.close(fig)
        lines += ['![新句验证](figures/2026-11-09-content-robust/validation.png)', '',
            '图 2：二十个新句的配对验证，误差线按人物整组重采样。完整联合逐对相同，严格身份联合有一个基线通过对退步。小样本区间不建立总体代表性。', '']
        lines += ['四个内容源沿用上一研究的固定设置，优化参考目标不使用这些转换的最终评分。所有内容源和零分母均报告，没有按干净质量事后换源。该数据集与之前两项研究不同，不能跨轮比较百分比当作算法增益。', '']
    lines += [
        '## 固定信道诊断', '',
        '在任何信道结果产生前固定四个已观察句、一个内容源，所有三个方法都评测原始、另一个房间响应及 MP3 64kbps 条件，共 36 条记录。新房间几何不在训练的两组响应中，但保护原音句子已用于开发；因此只能称有限信道诊断，不能称新句泛化。', '',
        '房间核使用一致的 L1 归一化以避免放大削波；各方法与其干净参考使用相同变换。每个条件重新转换并独立评分干净参考，干净控制失败不会记作防护成功。', '',
        '| 条件 | 方法 | 完整联合 | 严格身份联合 | 原音质量 |', '|---|---|---:|---:|---:|']
    for channel, values in channels['by_channel'].items():
        for name in names:
            value = values[name]
            lines.append(f"| {channel} | {LABELS[name]} | {ratio(value['paper_full_screen'])} | {ratio(value['strict_joint'])} | {value['protected_quality_count']}/4 |")
    lines += ['', '零分母保留为 N/A。上述信道分数不参与方法选择，不能从某个较高百分比推断总体鲁棒性。', '',
        '本次有限信道诊断中三个方法的成功计数相同，未观察到环境变化候选的额外优势。房间处理使三个方法的原音质量通过数都从 3/4 降到 2/4，说明该预处理本身也会损伤质量，不能将其算作候选防护增益。', '',
        '## 测试、证据与复跑', '',
        f"新增七项数值与门槛检查后，全部 {test_record['count']} 项单元测试在云端通过。真实语音的 CTC 输入梯度非零且有限，所有模型权重冻结。软件测试不替代独立质量测量。", '',
        f"不可变归档在本机核验 {archive['all_files_verified']} 个成员、{archive['wav_files']} 个 WAV，评分路径逐项匹配 {archive['measured_wavs_verified_against_scores']} 个；SHA256 {archive['sha256']}。包含原始输入、注册原型、执行源码、输出波形、转录、选择、门槛及全部负结果。历史归档保留。", '',
        '权重分发需要已有完整模型包和质量评估包，另补充 VoiceGuard-content-ctc-resources.zip，其准确摘要在 ctc-resource-receipt.json。包内 checkpoints/ctc/wav2vec2_fairseq_base_ls960_asr_ls960.pth 约 360MiB，通过官方 torchaudio 下载；本机逐项核验。也可运行 tools/fetch_content_ctc.py 下载并对照冻结协议校验；不将大权重或私钥提交 Git。', '',
        '运行入口 scripts/run_content_robust.py，使用新工作副本和空输出目录；完整状态不会重复覆盖。开发选择和门槛工具为 tools/decide_content_validation.py。离线审核 tools/audit_content_results.py 使用 --evidence 和 --run 指向证据与解压后的运行目录。信道诊断入口 tools/probe_content_channels.py。修改方案必须新建协议，不能覆盖本次执行字节。', '',
        '本轮没有真人 MOS、第二种 VC、实时生成器或真实房间录制；不能宣称达到论文整体质量，也不能宣称 Kalypta 产品复现。报告记录实际观测和局限，课程加分及验收由教师决定。实验结束不会自动关闭 ECS。', '',
        '来源：用户提供的 CSDN 文章 https://blog.csdn.net/weixin_33139275/article/details/170178909；Treble 官方 https://www.treble.tech/；相关开源声学工具 https://github.com/LCAV/pyroomacoustics。']
    markdown = ROOT / 'docs/18_内容监督与环境鲁棒优化.md'
    markdown.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    write_json(evidence / 'report-inputs.json', {'markdown_sha256': file_sha256(markdown),
        'analysis_sha256': file_sha256(evidence / 'analysis.json'),
        'channel_analysis_sha256': file_sha256(evidence / 'channel-analysis.json'),
        'archive_verification_sha256': file_sha256(evidence / 'archive-verification.json')})
    print(markdown)


if __name__ == '__main__': main()
