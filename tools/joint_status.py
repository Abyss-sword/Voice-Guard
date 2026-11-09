"""Compact read-only progress for the joint experiment."""
import argparse
import json
from pathlib import Path
import time


def watch_validation(root, expected_utterances):
    for _ in range(112):
        state = json.loads((root / 'validation-state.json').read_text())
        path = root / 'validation-results.json'
        rows = json.loads(path.read_text())['rows'] if path.exists() else []
        counts = {}
        for r in rows:
            counts.setdefault(r['utterance'], {}).setdefault(r['profile'], 0)
            counts[r['utterance']][r['profile']] += 1
        complete = {u for u, v in counts.items() if len(v) == 2 and all(n == 4 for n in v.values())}
        paired = [r for r in rows if r['utterance'] in complete]
        profiles = {}
        for name in sorted({r['profile'] for r in paired}):
            values = [r for r in paired if r['profile'] == name]
            profiles[name] = {'full_joint': sum(r['gates_ge2e']['paper_full_screen'] for r in values),
                'full_clean': sum(r['gates_ge2e']['clean_full_quality_ok'] for r in values),
                'strict_joint': sum(r['gates_ge2e']['strict_joint'] for r in values),
                'identity_clean': sum(r['gates_ge2e']['clean_identity_ok'] for r in values)}
        print(json.dumps({'state': state['state'], 'records': len(rows), 'expected_records': expected_utterances * 8,
            'paired_utterances': len(complete), 'expected_utterances': expected_utterances, 'paired_only': profiles}), flush=True)
        if state['state'] == 'complete': return
        if state['state'] != 'running': raise RuntimeError(state)
        time.sleep(45)
    raise TimeoutError('Bounded read-only monitor finished')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch-validation', action='store_true')
    parser.add_argument('--study', choices=['joint', 'anchored'], default='joint')
    args = parser.parse_args()
    root = Path('outputs/joint_recovery/2026-11-09' if args.study == 'joint' else 'outputs/joint_anchored/2026-11-09')
    if args.watch_validation:
        watch_validation(root, 20 if args.study == 'joint' else 10); return
    for split in ('development', 'validation'):
        state = root / (split + '-state.json')
        if not state.exists(): continue
        result = {'split': split, 'state': json.loads(state.read_text())}
        path = root / (split + '-results.json')
        if path.exists():
            rows = json.loads(path.read_text())['rows']
            result['completed_rows'] = len(rows)
            result['profiles'] = {}
            for name in sorted({r['profile'] for r in rows}):
                values = [r for r in rows if r['profile'] == name]
                unique = {r['utterance']: r for r in values}
                result['profiles'][name] = {'rows': len(values), 'utterances': len(unique),
                    'full_joint': sum(r['gates_ge2e']['paper_full_screen'] for r in values),
                    'full_clean_denominator': sum(r['gates_ge2e']['clean_full_quality_ok'] for r in values),
                    'strict_joint': sum(r['gates_ge2e']['strict_joint'] for r in values),
                    'identity_clean_denominator': sum(r['gates_ge2e']['clean_identity_ok'] for r in values),
                    'protected_quality': sum(r['gates_ge2e']['protected_quality_ok'] for r in unique.values())}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
