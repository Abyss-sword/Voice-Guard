"""Enforce the preregistered original-quality and joint-development gate."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def eligible(selected, baseline, rows):
    def passed(name):
        return {r['utterance'] for r in rows if r['profile'] == name
            and r['gates_ge2e']['protected_quality_ok']}
    checks = {'full_joint_improved': selected['full_joint'] > baseline['full_joint'],
        'strict_joint_not_worse': selected['strict_joint'] >= baseline['strict_joint'],
        'original_quality_count_not_worse': selected['protected_quality'] >= baseline['protected_quality'],
        'every_baseline_passing_original_preserved': passed(baseline['profile']['name']) <= passed(selected['profile']['name']),
        'wer_not_worse_beyond_tolerance': selected['mean_wer'] <= baseline['mean_wer'] + .005,
        'mos_not_worse_beyond_tolerance': selected['mean_mos'] >= baseline['mean_mos'] - .05}
    return all(checks.values()), checks


def main():
    evidence = Path('docs/evidence/2026-11-09-discriminant')
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    protocol, selection = read(evidence / 'protocol.json'), read(evidence / 'selection.json')
    path = evidence / 'extension-decision.json'
    if path.exists(): raise ValueError('Development decision already frozen')
    if selection['protocol_sha256'] != file_sha256(evidence / 'protocol.json'):
        raise ValueError('Selection protocol mismatch')
    run = Path(protocol['output'])
    if (run / 'validation-state.json').exists(): raise ValueError('Validation already examined')
    if file_sha256(run / 'development-results.json') != selection['development_results_sha256']:
        raise ValueError('Development changed after selection')
    allowed, checks = eligible(selection['selected'], selection['baseline'],
        read(run / 'development-results.json')['rows'])
    write_json(path, {'decided_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': file_sha256(evidence / 'protocol.json'),
        'selection_sha256': file_sha256(evidence / 'selection.json'),
        'validation_allowed': allowed, 'checks': checks,
        'gate': protocol['validation_gate'], 'new_validation_examined': False})
    print(json.dumps(read(path), indent=2))
    if not allowed: raise SystemExit(3)


if __name__ == '__main__': main()
