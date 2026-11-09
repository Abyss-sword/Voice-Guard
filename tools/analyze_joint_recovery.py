"""Audit paired joint-defense results without choosing on validation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_staged_recovery import summary, cluster_intervals
from vguard.metrics.staged import gates
from vguard.run_state import file_sha256, write_json


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def analyze(evidence, run):
    protocol_path, selection_path = evidence / 'protocol.json', evidence / 'selection.json'
    protocol, selection = read(protocol_path), read(selection_path)
    digest = file_sha256(protocol_path)
    if selection['protocol_sha256'] != digest or selection['validation_examined']:
        raise ValueError('Invalid development selection')
    state = read(run / 'validation-state.json')
    rows = read(run / 'validation-results.json')['rows']
    baseline, challenger = protocol['baseline']['name'], selection['selected']['profile']['name']
    profiles = [baseline, challenger]
    expected = {(u, c, p) for u in protocol['validation'] for c in protocol['evaluation_contents'] for p in profiles}
    if state['state'] != 'complete' or state['rows'] != len(expected): raise ValueError('Incomplete validation')
    if len(rows) != len(expected) or {(r['utterance'], r['content'], r['profile']) for r in rows} != expected:
        raise ValueError('Missing/duplicate validation records')
    if state['protocol_sha256'] != digest or state['selection_sha256'] != file_sha256(selection_path):
        raise ValueError('State protocol/selection mismatch')
    controls = {}
    for r in rows:
        if r['protocol_sha256'] != digest: raise ValueError('Row protocol mismatch')
        key = r['utterance'], r['content']
        clean = {k: r[k] for k in ('original', 'clean_vc')}
        if key in controls and controls[key] != clean: raise ValueError('Unpaired clean controls')
        controls[key] = clean
        job = run / 'validation' / r['profile'] / r['utterance']
        if r['optimization_sha256'] != file_sha256(job / 'optimization.json'):
            raise ValueError('Optimization metadata changed')
        for asv, asr in [('ge2e', 'small_en'), ('ecapa', 'base')]:
            if r['gates_' + asv] != gates(r['utterance'], r['original'], r['clean_vc'],
                    r['defended'], r['protected_vc'], asv=asv, asr=asr):
                raise ValueError('Stored gate mismatch')
    output = {'counts_verified': True, 'protocol_sha256': digest,
        'selection_sha256': file_sha256(selection_path),
        'validation_results_sha256': file_sha256(run / 'validation-results.json'),
        'baseline': baseline, 'challenger': challenger,
        'recommended_after_development': selection['recommended_after_development'],
        'utterances': len(protocol['validation']), 'speakers': len({u.split('_')[0] for u in protocol['validation']}),
        'evaluations': {}, 'limitations': protocol['limitations'] + [
            'ECAPA is optimized directly; ECAPA results are not independent evidence.',
            'New fixed content sources change the denominator; compare only this paired baseline.']}
    for asv, asr in [('ge2e', 'small_en'), ('ecapa', 'base')]:
        values = {}
        for name in profiles:
            selected = [r for r in rows if r['profile'] == name]
            original_anchors = [r for r in selected if r['content'] in ('p226_002', 'p227_002')]
            held_out_text = [r for r in selected if r['content'] == 'p226_004']
            values[name] = {'all': summary(selected, asv, asr),
                'original_content_anchors': summary(original_anchors, asv, asr),
                'held_out_text': summary(held_out_text, asv, asr),
                'by_content': {c: summary([r for r in selected if r['content'] == c], asv, asr)
                    for c in protocol['evaluation_contents']}}
        before = [r for r in rows if r['profile'] == baseline]
        after = [r for r in rows if r['profile'] == challenger]
        values['paired_intervals'] = cluster_intervals(before, after, 'gates_' + asv)
        pairs = {(r['utterance'], r['content']): r for r in after}
        values['paired_transitions'] = {endpoint: dict(Counter(
            ('both_pass' if a and b else 'baseline_only' if a else 'challenger_only' if b else 'both_fail')
            for r in before for a, b in [(r['gates_' + asv][endpoint], pairs[r['utterance'], r['content']]['gates_' + asv][endpoint])]))
            for endpoint in ('strict_joint', 'paper_identity_screen', 'paper_full_screen')}
        output['evaluations'][asv + '_' + asr] = values
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('docs/evidence/2026-11-09-joint'))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    run = args.run or Path(read(args.evidence / 'protocol.json')['output'])
    output = analyze(args.evidence, run)
    write_json(args.evidence / 'analysis.json', output)
    print(json.dumps({k: {p: v['all'] for p, v in value.items() if p in (output['baseline'], output['challenger'])}
        for k, value in output['evaluations'].items()}, indent=2))


if __name__ == '__main__': main()
