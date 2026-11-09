"""Explain historical clean-control failures without changing their gates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def audit(path):
    rows = json.loads(path.read_text(encoding='utf-8'))['rows']
    unique = {}
    for row in rows:
        key = row['utterance'], row['content']
        observed = {k: row[k] for k in ('original', 'clean_vc')}
        if key in unique and unique[key] != observed:
            raise ValueError('Different clean controls for the same pair')
        unique[key] = observed
    details = []
    counts = Counter()
    for (utterance, content), observations in unique.items():
        clean = observations['clean_vc']
        failures = []
        if clean['asv_ge2e'] != utterance.split('_')[0]: failures.append('identity')
        if clean['mos_original'] < 3: failures.append('mos')
        if clean['wer_small_en'] > .1: failures.append('wer')
        counts.update(failures)
        details.append({'utterance': utterance, 'content': content, 'failures': failures,
            'mos': clean['mos_original'], 'wer': clean['wer_small_en'],
            'reference': clean['reference'], 'hypothesis': clean['hypothesis_small_en'],
            'english_normalized_wer_diagnostic': clean['wer_english_small_en']})
    return {'source_sha256': file_sha256(path), 'clean_pairs': len(unique),
        'full_quality_pass': sum(not r['failures'] for r in details),
        'failure_reasons_nonexclusive': dict(counts),
        'distinct_reference_texts': sorted({r['reference'] for r in details}),
        'note': 'Historical frozen gates retained. WER>10% on an 11-word sentence permits at most one word error. New experiment broadens fixed content sources before evaluation; no historical scores are replaced.',
        'details': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve existing audit')
    result = audit(args.input)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
