"""Describe proxy/held-out disagreement after validation; never select methods."""
import argparse
import json
from pathlib import Path
import sys
import statistics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def margin(score, speaker, model):
    similarities = score['similarities_' + model]
    return similarities[speaker] - max(value for name, value in similarities.items() if name != speaker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('docs/evidence/2026-11-09-joint'))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    protocol, selection = read(args.evidence / 'protocol.json'), read(args.evidence / 'selection.json')
    run = args.run or Path(protocol['output'])
    if read(run / 'validation-state.json')['state'] != 'complete': raise ValueError('Complete validation required')
    path = run / 'validation-results.json'
    rows = read(path)['rows']
    baseline, challenger = protocol['baseline']['name'], selection['selected']['profile']['name']
    groups = {name: [r for r in rows if r['profile'] == name] for name in (baseline, challenger)}
    expected = len(protocol['validation']) * len(protocol['evaluation_contents'])
    if any(len(v) != expected for v in groups.values()): raise ValueError('Missing records')
    summaries = {}
    for name, values in groups.items():
        speakers = {r['utterance']: r for r in values}
        summaries[name] = {
            'pairs': len(values),
            'converted_proxy_changed_heldout_retained': sum(r['protected_vc']['asv_ecapa'] != r['utterance'].split('_')[0]
                and r['protected_vc']['asv_ge2e'] == r['utterance'].split('_')[0] for r in values),
            'converted_heldout_changed_proxy_retained': sum(r['protected_vc']['asv_ge2e'] != r['utterance'].split('_')[0]
                and r['protected_vc']['asv_ecapa'] == r['utterance'].split('_')[0] for r in values),
            'mean_output_ecapa_own_margin': statistics.mean(margin(r['protected_vc'], r['utterance'].split('_')[0], 'ecapa') for r in values),
            'mean_output_ge2e_own_margin': statistics.mean(margin(r['protected_vc'], r['utterance'].split('_')[0], 'ge2e') for r in values),
            'mean_original_ecapa_own_margin': statistics.mean(margin(r['defended'], r['utterance'].split('_')[0], 'ecapa') for r in speakers.values()),
            'mean_original_ge2e_own_margin': statistics.mean(margin(r['defended'], r['utterance'].split('_')[0], 'ge2e') for r in speakers.values())}
    new = {(r['utterance'], r['content']): r for r in groups[challenger]}
    changes = []
    for b in groups[baseline]:
        a = new[b['utterance'], b['content']]
        if b['gates_ge2e']['paper_full_screen'] == a['gates_ge2e']['paper_full_screen']: continue
        speaker = b['utterance'].split('_')[0]
        changes.append({'utterance': b['utterance'], 'content': b['content'],
            'baseline_joint': b['gates_ge2e']['paper_full_screen'],
            'challenger_joint': a['gates_ge2e']['paper_full_screen'],
            'challenger_original_quality_ok': a['gates_ge2e']['protected_quality_ok'],
            'baseline_output_ge2e_identity': b['protected_vc']['asv_ge2e'],
            'challenger_output_ge2e_identity': a['protected_vc']['asv_ge2e'],
            'baseline_output_ge2e_margin': margin(b['protected_vc'], speaker, 'ge2e'),
            'challenger_output_ge2e_margin': margin(a['protected_vc'], speaker, 'ge2e'),
            'baseline_output_wer': b['protected_vc']['wer_small_en'],
            'challenger_output_wer': a['protected_vc']['wer_small_en'],
            'baseline_output_mos': b['protected_vc']['mos_original'],
            'challenger_output_mos': a['protected_vc']['mos_original']})
    destination = args.evidence / 'transfer-audit.json'
    if destination.exists(): raise ValueError('Preserve post-validation audit')
    result = {'validation_results_sha256': file_sha256(path), 'audit_code_sha256': file_sha256(__file__),
        'post_validation_only': True, 'method_selection_changed': False,
        'summaries': summaries, 'changed_full_joint_pairs': changes,
        'note': 'Proxy and GE2E embeddings have different spaces and margins. Each margin is within its own registry; do not compare raw margin magnitude across models. Descriptive mechanism check, not a causal ablation.'}
    write_json(destination, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
