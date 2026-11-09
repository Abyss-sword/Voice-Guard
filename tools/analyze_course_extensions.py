"""Describe completed frozen experiments without inventing positive conclusions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.evaluation import summarize_pairs
from vguard.run_state import write_json


def tab(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers),
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course-dir',type=Path,default=ROOT/'outputs/course/remote/outputs/course/2026-11-09')
    args=parser.parse_args()
    new={}
    for name in ('baseline','refined'):
        summary=json.loads((args.course_dir/f'quality_new_{name}/summary.json').read_text(encoding='utf-8'))
        if len(summary['rows'])!=20:raise ValueError('New-utterance evaluation is incomplete')
        new[name]={row['utterance']:row for row in summary['rows'] if row['scale']==1}
    if set(new['baseline'])!=set(new['refined']) or len(new['baseline'])!=10:
        raise ValueError('Need ten matched new utterances')
    control_disagreements=[key for key in new['baseline'] if
        new['baseline'][key]['clean_vc_identity_ok']!=new['refined'][key]['clean_vc_identity_ok']]
    stats={name:summarize_pairs(list(rows.values())) for name,rows in new.items()}
    deltas={metric:sum(new['refined'][u][metric]-new['baseline'][u][metric]
                       for u in new['baseline'])/10 for metric in ('mos_defended','wer_defended')}
    counts={name:{field:sum(bool(row[field]) for row in rows.values())
                  for field in ('quality_ok','identity_ok','clean_vc_identity_ok','screened_quality_identity_ok')}
            for name,rows in new.items()}
    names={'channels_baseline':40,'channels_refined_dev':16,'channels_refined_validation':24,
           'contents_baseline':30,'contents_refined_dev':12,'contents_refined_validation':18}
    audits={}
    for name,count in names.items():
        obj=json.loads((args.course_dir/name/'summary.json').read_text(encoding='utf-8'))
        if not obj.get('complete') or len(obj['rows'])!=count:raise ValueError(f'Incomplete audit: {name}')
        audits[name]=obj
    lines=['### 6.2 扩展结果的解释','',
        f"新 10 句的配对平均变化为 MOSNet {deltas['mos_defended']:+.3f}、宏平均 WER {100*deltas['wer_defended']:+.2f} 个百分点。所有 10 对均纳入，不根据身份或质量事后删句。两方法的干净转换筛选差异为 {len(control_disagreements)}/10，联合结果分别使用各自实测分母，不择优替换。",'',
        tab(['新句方法','仅保护质量与原身份通过','仅身份维通过','联合通过'],
            [[name,f"{counts[name]['quality_ok']}/10",f"{counts[name]['identity_ok']}/10",
              f"{stats[name]['joint_successes']}/{stats[name]['screened']}" if stats[name]['screened'] else 'N/A']
             for name in ('baseline','refined')]),'',
        '质量门槛通过与转换端身份变化是不同条件。即使质量均值改善，也不能把没有联合通过的句子解释成有效保护。新句验证不参与调参，未达到的目标作为负结果保留。','',
        '信道审计以本次原样条件作为保留率起点。转换随机种子按“目标句:内容源”固定，与旧质量评测按“目标句:scale”派生的种子不同；因此本审计的原样判定独立记录，不强行套用历史 2/4。原样保留源文件解码后的样本，PCM16 则显式量化；原 VG 源文件已为 PCM16，改进源文件为 FLOAT，量化的作用应分别解释。','']
    retention=[]
    mechanism=[]
    for name,obj in audits.items():
        if not name.startswith('channels_'):continue
        controls={(r['utterance'],r['content']):r for r in obj['rows'] if r['channel']=='none'}
        for row in obj['rows']:
            if row['channel']!='mp3-32k':continue
            original=controls[(row['utterance'],row['content'])]
            target=row['utterance'].split('_')[0]
            def margin(record):
                scores=record['converted_defended_scores']
                return max(value for person,value in scores.items() if person!=target)-scores[target]
            mechanism.append({'method':name,'utterance':row['utterance'],
                'none_joint':original['screened_quality_identity_ok'],
                'mp3_joint':row['screened_quality_identity_ok'],
                'none_wrong_minus_target':margin(original),'mp3_wrong_minus_target':margin(row),
                'none_difference_highband_fraction':original['difference_high_band_fraction_ge4khz'],
                'mp3_difference_highband_fraction':row['difference_high_band_fraction_ge4khz'],
                'mp3_clean_screen':row['clean_vc_identity_ok'],'mp3_quality_ok':row['quality_ok'],
                'mp3_identity_ok':row['identity_ok']})
    for channel in audits['channels_baseline']['protocol']['channels']:
        row=[channel];lost=0
        for name in ('channels_baseline','channels_refined_dev','channels_refined_validation'):
            record=audits[name]['channels'][channel];n=record['initial_joint_successes']
            row.append(f"{record['retained_initial_joint_successes']}/{n}" if n else 'N/A')
            lost+=record['initial_successes_losing_clean_screen']
        retention.append([*row,lost])
    lines += [tab(['信道','原 VG 成功保留','改进开发成功保留','改进验证成功保留','丢失干净控制合计'],retention),'',
              '本次改进法 p225 在 MP3 128 kbps 下保护质量仍通过、干净转换仍为 p225，但保护后转换也恢复为 p225，属于可定位的身份维防护丢失。同一句在 MP3 32 kbps 却保留联合通过，因此不能宣称码率越低防护必然越差。p228 在 MP3 32 kbps 下的干净转换已经变为 p241，应排除而不是当作扰动被消除。','',
              'G.711 和 8 kHz 重采样使两个原样成功样本的干净转换控制都失败；两者成功保留为 0/2，但不能单凭这个数推断防御扰动失效。Opus 下 p225 是身份维防护丢失，p228 则是干净转换失败，需要区别解释。','',
              '内容源也影响局部有效性：改进开发组在 p226_002 和 p227_002 下各联合通过 1/2，换成 p225_002 后为 0/2；验证组在三种内容源下均为 1/2。以上为固定五句的相关重复，不能作为对所有内容的稳健性保证。','',
              '### 6.3 编码后机制诊断','',
              '下面以 MP3 32 kbps 为固定诊断条件，报告转换身份的“最大非目标余弦减目标余弦”。正值表示非目标排名领先，负值表示仍是目标身份；不是认证系统阈值。高频差分比取共同前缀差分的 4 kHz 以上能量比例，编码延迟、非线性和边界变化也会影响差分，不能当成独立扰动被滤除的直接证明。','',
              tab(['实验组/目标','原样→MP3 身份余量','原样→MP3 高频差分比','MP3 干净/质量/身份'],
                  [[r['method'].replace('channels_','')+'/'+r['utterance'],
                    f"{r['none_wrong_minus_target']:.3f} → {r['mp3_wrong_minus_target']:.3f}",
                    f"{r['none_difference_highband_fraction']:.3f} → {r['mp3_difference_highband_fraction']:.3f}",
                    '/'.join('通过' if r[k] else '失败' for k in ('mp3_clean_screen','mp3_quality_ok','mp3_identity_ok'))]
                   for r in mechanism]),'',
              '这些配对观测能定位失败发生在质量、干净控制还是保护后身份环节；没有做频带消融或控制时延的独立因果实验，因此不把高频变化断言为唯一原因，也没有将未经实现的 EOT 当作改进结果。','']
    output=ROOT/'docs/evidence/2026-11-09'
    (output/'extension-analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(output/'extension-diagnostics.json',{'new_statistics':stats,'new_counts':counts,
               'clean_control_disagreements':control_disagreements,
               'new_paired_mean_changes':deltas,'mp3_fixed_diagnostics':mechanism})
    print(json.dumps({'new_statistics':stats,'new_paired_changes':deltas},indent=2))


if __name__=='__main__':main()
