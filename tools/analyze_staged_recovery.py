"""Audit complete paired staged results, with no validation configuration choice."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.metrics.staged import gates
from vguard.run_state import file_sha256, write_json


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def rate(count, denominator):
    return {'successes': count, 'denominator': denominator, 'rate': count/denominator if denominator else None}


def summary(rows, asv='ge2e', asr='small_en'):
    gate_key = 'gates_' + asv
    unique = {}
    for r in rows:
        value = {k: r[k] for k in ('original', 'defended', 'snr_db', 'mask_exceed_fraction', 'mask_excess_mean_db')}
        if r['utterance'] in unique and unique[r['utterance']] != value:
            raise ValueError('Repeated content changes protected observations')
        unique[r['utterance']] = value
    values = list(unique.values())
    identity = sum(r[gate_key]['clean_identity_ok'] for r in rows)
    full = sum(r[gate_key]['clean_full_quality_ok'] for r in rows)
    return {'utterances': len(values), 'conversion_pairs': len(rows),
        'protected_mos_mean': statistics.mean(v['defended']['mos_original'] for v in values),
        'clean_mos_mean': statistics.mean(v['original']['mos_original'] for v in values),
        'protected_wer_macro': statistics.mean(v['defended']['wer_' + asr] for v in values),
        'clean_wer_macro': statistics.mean(v['original']['wer_' + asr] for v in values),
        'protected_english_wer_macro_sensitivity_only': statistics.mean(v['defended']['wer_english_' + asr] for v in values),
        'protected_identity_retained': sum(v['defended']['asv_' + asv] == u.split('_')[0] for u, v in unique.items()),
        'protected_quality_count': len({r['utterance'] for r in rows if r[gate_key]['protected_quality_ok']}),
        'protected_quality_failure_reasons_nonexclusive': {
            'identity': sum(v['defended']['asv_' + asv] != u.split('_')[0] for u, v in unique.items()),
            'mos': sum(v['defended']['mos_original'] < 3 for v in values),
            'wer': sum(v['defended']['wer_' + asr] > .10 for v in values)},
        'asr_quality_comparison_to_clean_original': {
            'both_exceed_10pct': sum(v['original']['wer_' + asr] > .10 and v['defended']['wer_' + asr] > .10 for v in values),
            'only_protected_exceeds_10pct': sum(v['original']['wer_' + asr] <= .10 and v['defended']['wer_' + asr] > .10 for v in values),
            'only_clean_exceeds_10pct': sum(v['original']['wer_' + asr] > .10 and v['defended']['wer_' + asr] <= .10 for v in values)},
        'converted_identity_retained': sum(r['protected_vc']['asv_' + asv] == r['utterance'].split('_')[0] for r in rows),
        'clean_converted_mos_mean': statistics.mean(r['clean_vc']['mos_original'] for r in rows),
        'protected_converted_mos_mean': statistics.mean(r['protected_vc']['mos_original'] for r in rows),
        'strict_joint': rate(sum(r[gate_key]['strict_joint'] for r in rows), identity),
        'paper_identity_screen': rate(sum(r[gate_key]['paper_identity_screen'] for r in rows), identity),
        'paper_full_screen': rate(sum(r[gate_key]['paper_full_screen'] for r in rows), full),
        'mean_snr_db': statistics.mean(v['snr_db'] for v in values),
        'mask_exceed_fraction_mean': statistics.mean(v['mask_exceed_fraction'] for v in values),
        'mask_excess_mean_db': statistics.mean(v['mask_excess_mean_db'] for v in values)}


def cluster_intervals(before, after, gate_key, samples=5000):
    """Describe uncertainty with shared speaker resampling, retaining paired controls."""
    speakers = sorted({r['utterance'].split('_')[0] for r in before})
    generator = np.random.default_rng(2026110903)
    picks = generator.integers(0, len(speakers), size=(samples, len(speakers)))
    output = {'resamples': samples, 'cluster_count': len(speakers),
        'note': 'Descriptive percentile intervals, treating speakers as clusters. Purposefully selected enrolled speakers do not establish population representativeness.'}
    for endpoint, denominator in [('strict_joint', 'clean_identity_ok'), ('paper_full_screen', 'clean_full_quality_ok')]:
        counts = []
        for speaker in speakers:
            b = [r for r in before if r['utterance'].split('_')[0] == speaker]
            a = [r for r in after if r['utterance'].split('_')[0] == speaker]
            db, da = sum(r[gate_key][denominator] for r in b), sum(r[gate_key][denominator] for r in a)
            if db != da: raise ValueError('Paired clean denominators differ')
            counts.append([sum(r[gate_key][endpoint] for r in b), sum(r[gate_key][endpoint] for r in a), db])
        aggregate = np.asarray(counts)[picks].sum(axis=1)
        valid = aggregate[:, 2] > 0
        if not valid.any():
            output[endpoint] = {'baseline_95': None, 'staged_95': None, 'paired_difference_95': None,
                                'zero_denominator_resamples': samples}
            continue
        baseline = aggregate[valid, 0] / aggregate[valid, 2]
        staged = aggregate[valid, 1] / aggregate[valid, 2]
        output[endpoint] = {'baseline_95': np.quantile(baseline, [.025, .975]).tolist(),
            'staged_95': np.quantile(staged, [.025, .975]).tolist(),
            'paired_difference_95': np.quantile(staged-baseline, [.025, .975]).tolist(),
            'zero_denominator_resamples': int((~valid).sum())}
    return output


def analyze(protocol_path, selection_path, run):
    protocol, selection = read(protocol_path), read(selection_path)
    state = read(run / 'validation-state.json')
    rows = read(run / 'validation-results.json')['rows']
    chosen = selection['selected']['profile']['name']
    profiles = ['previous_feedback_snr20', chosen]
    expected = {(u, c, p) for u in protocol['validation'] for c in protocol['evaluation_contents'] for p in profiles}
    if state['state'] != 'complete' or state.get('rows') != len(expected): raise ValueError('Validation incomplete')
    if len(rows) != len(expected) or {(r['utterance'], r['content'], r['profile']) for r in rows} != expected:
        raise ValueError('Missing or duplicate paired validation')
    if state['protocol_sha256'] != file_sha256(protocol_path) or state['selection_sha256'] != file_sha256(selection_path):
        raise ValueError('Frozen protocol/selection differs')
    by_pair = {}
    for r in rows:
        if r['protocol_sha256'] != file_sha256(protocol_path): raise ValueError('Row protocol differs')
        for asv, asr in [('ge2e', 'small_en'), ('ecapa', 'base')]:
            computed = gates(r['utterance'], r['original'], r['clean_vc'], r['defended'], r['protected_vc'], asv=asv, asr=asr)
            if r['gates_' + asv] != computed: raise ValueError('Stored gates inconsistent')
        key = r['utterance'], r['content']
        clean = {k: r[k] for k in ('original', 'clean_vc')}
        if key in by_pair and by_pair[key] != clean: raise ValueError('Baseline and staged use different clean controls')
        by_pair[key] = clean
    output = {'protocol_sha256': file_sha256(protocol_path), 'selection_sha256': file_sha256(selection_path),
        'validation_results_sha256': file_sha256(run / 'validation-results.json'),
        'counts_verified': True, 'selected_profile': chosen, 'utterances': len(protocol['validation']),
        'speakers': len({u.split('_')[0] for u in protocol['validation']}), 'evaluations': {},
        'limitations': ['Two content sources per utterance are correlated repetitions.',
            'ECAPA selected the new method target identity; only GE2E is an independent ASV model for this change.',
            'English-normalized WER is a diagnostic; frozen primary Basic-normalized gates never change.',
            'Native enrollment, MSE and staged constraints change together; comparison measures their combined effect.',
            'Automatic MOS and this small single-VC sample do not establish paper-scale equivalence.']}
    for asv, asr in [('ge2e', 'small_en'), ('ecapa', 'base')]:
        output['evaluations'][asv + '_' + asr] = {}
        for p in profiles:
            values = [r for r in rows if r['profile'] == p]
            output['evaluations'][asv + '_' + asr][p] = {'all': summary(values, asv, asr),
                'by_content': {c: summary([r for r in values if r['content'] == c], asv, asr) for c in protocol['evaluation_contents']},
                'by_group': {label: summary([r for r in values if (int(r['utterance'][1:4]) <= 229) == seen], asv, asr)
                    for label, seen in [('p225_p229_includes_three_development_speakers', True), ('p230_p234_not_used_in_attack_development', False)]}}
        before = {(r['utterance'], r['content']): r for r in rows if r['profile'] == profiles[0]}
        after = {(r['utterance'], r['content']): r for r in rows if r['profile'] == chosen}
        output['evaluations'][asv + '_' + asr]['paired_transitions'] = {
            endpoint: {label: sum(a['gates_' + asv][endpoint] == b and after[key]['gates_' + asv][endpoint] == n
                                 for key, a in before.items())
                       for label, b, n in [('both_fail', False, False), ('baseline_only', True, False), ('staged_only', False, True), ('both_pass', True, True)]}
            for endpoint in ('strict_joint', 'paper_identity_screen', 'paper_full_screen')}
        output['evaluations'][asv + '_' + asr]['speaker_cluster_bootstrap'] = cluster_intervals(list(before.values()), list(after.values()), 'gates_' + asv)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('docs/evidence/2026-11-09-staged/final'))
    parser.add_argument('--run', type=Path, default=Path('outputs/staged_recovery/2026-11-09/final'))
    args = parser.parse_args()
    output = analyze(args.evidence / 'protocol.json', args.evidence / 'selection.json', args.run)
    write_json(args.evidence / 'analysis.json', output)
    print(json.dumps({k: {p: v['all'] for p, v in value.items() if p in ('previous_feedback_snr20', output['selected_profile'])} for k, value in output['evaluations'].items()}, indent=2))


if __name__ == '__main__': main()
