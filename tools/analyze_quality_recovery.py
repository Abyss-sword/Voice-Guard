"""Audit completed recovery results without models or selecting validation settings."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def ratio(successes, denominator):
    return {'successes': successes, 'denominator': denominator,
            'rate': successes / denominator if denominator else None}


def average(values):
    return statistics.mean(values) if values else None


def summarize(rows, supplement=False):
    unique = {}
    for row in rows:
        key = row['utterance']
        protected = row['scores']['defended'] if supplement else row['defended']
        clean = row['scores']['original'] if supplement else row['control']['original']
        value = (protected, clean, row['mos_original'] if supplement else protected['mos_original'])
        if key in unique and unique[key] != value:
            raise ValueError('Repeated content has inconsistent protected scores: ' + key)
        unique[key] = value
    wer_key = 'wer_small_en' if supplement else 'wer'
    asv_key = 'asv_ge2e' if supplement else 'asv'
    identity = sum(r['clean_identity_ok'] for r in rows)
    full = sum(r['clean_full_quality_ok'] if supplement else
               r['clean_identity_ok'] and r['clean_quality_ok'] for r in rows)
    return {
        'utterances': len(unique), 'conversion_pairs': len(rows),
        'clean_identity_controls': identity, 'full_clean_controls': full,
        'protected_mos_original_mean': average(v[2] for v in unique.values()),
        'clean_mos_original_mean': None if supplement else average(v[1]['mos_original'] for v in unique.values()),
        'protected_wer_macro': average(v[0][wer_key] for v in unique.values()),
        'clean_wer_macro': average(v[1][wer_key] for v in unique.values()),
        'protected_identity_retained': sum(v[0][asv_key] == u.split('_')[0] for u, v in unique.items()),
        'protected_quality_pass_utterances': len({r['utterance'] for r in rows if r['protected_quality_ok']}),
        'strict_joint': ratio(sum(r['strict_joint'] for r in rows), identity),
        'paper_structure_identity_screen': ratio(sum(r['paper_structure_identity_screen'] for r in rows), identity),
        'paper_structure_full_screen': ratio(sum(r['paper_structure_full_screen'] for r in rows), full),
        'mean_snr_db': None if supplement else average(r['snr_db'] for r in
            {r['utterance']: r for r in rows}.values()),
        'mask_exceed_fraction_mean': None if supplement else average(r['mask_exceed_fraction'] for r in
            {r['utterance']: r for r in rows}.values()),
        'mask_excess_mean_db': None if supplement else average(r['mask_excess_mean_db'] for r in
            {r['utterance']: r for r in rows}.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/paper_quality/2026-11-09-recovery'))
    parser.add_argument('--evidence', type=Path, default=Path('docs/evidence/2026-11-09-quality'))
    args = parser.parse_args()
    def read(path): return json.loads(path.read_text(encoding='utf-8'))
    protocol = read(args.evidence / 'recovery-protocol.json')
    selection = read(args.evidence / 'recovery-selection.json')
    state = read(args.root / 'validation-state.json')
    primary_path = args.root / 'validation-results.json'
    secondary_path = args.root / 'validation-supplement.json'
    primary = read(primary_path)['rows']
    secondary_obj = read(secondary_path)
    secondary = secondary_obj['rows']
    expected = {(u, c) for u in protocol['validation'] for c in protocol['evaluation_contents']['validation']}
    if state['state'] != 'complete' or state['rows'] != len(expected):
        raise ValueError('Validation is not complete')
    for label, rows in [('primary', primary), ('secondary', secondary)]:
        if len(rows) != len(expected) or {(r['utterance'], r['content']) for r in rows} != expected:
            raise ValueError(label + ' pairs missing or duplicated')
        if any(r['profile'] != selection['profile']['name'] or r['checkpoint'] != selection['checkpoint'] for r in rows):
            raise ValueError('Validation deviates from global selection')
    if secondary_obj['primary_results_sha256'] != file_sha256(primary_path):
        raise ValueError('Supplement evaluated different primary results')
    if state['protocol_sha256'] != file_sha256(args.evidence / 'recovery-protocol.json'):
        raise ValueError('Protocol mismatch')
    if state['selection_sha256'] != file_sha256(args.evidence / 'recovery-selection.json'):
        raise ValueError('Selection mismatch')
    for r in primary:
        speaker = r['utterance'].split('_')[0]
        clean = r['control']['converted']
        protected = r['defended']
        converted = r['converted']
        identity = clean['asv'] == speaker
        control_quality = clean['mos_original'] >= 3 and clean['wer'] <= .10
        quality = protected['asv'] == speaker and protected['mos_original'] >= 3 and protected['wer'] <= .10
        changed = converted['asv'] != speaker
        disruption = changed or converted['mos_original'] < 3 or converted['wer'] > .10
        checks = {'clean_identity_ok': identity, 'clean_quality_ok': control_quality,
                  'protected_quality_ok': quality, 'strict_joint': identity and quality and changed,
                  'paper_structure_identity_screen': identity and quality and disruption,
                  'paper_structure_full_screen': identity and control_quality and quality and disruption}
        if any(r[key] != value for key, value in checks.items()):
            raise ValueError('Stored primary gate inconsistent')
    primary_by_pair = {(r['utterance'], r['content']): r for r in primary}
    for r in secondary:
        p = primary_by_pair[(r['utterance'], r['content'])]
        s = r['scores']
        speaker = r['utterance'].split('_')[0]
        for label, score in [('original', p['control']['original']), ('clean_vc', p['control']['converted']),
                             ('defended', p['defended']), ('protected_vc', p['converted'])]:
            if s[label]['sha256'] != score['sha256']:
                raise ValueError('Independent evaluation used different WAVs')
        identity = s['clean_vc']['asv_ge2e'] == speaker
        control_quality = p['control']['converted']['mos_original'] >= 3 and s['clean_vc']['wer_small_en'] <= .10
        quality = s['defended']['asv_ge2e'] == speaker and p['defended']['mos_original'] >= 3 and s['defended']['wer_small_en'] <= .10
        changed = s['protected_vc']['asv_ge2e'] != speaker
        disruption = changed or p['converted']['mos_original'] < 3 or s['protected_vc']['wer_small_en'] > .10
        checks = {'clean_identity_ok': identity, 'clean_full_quality_ok': identity and control_quality,
                  'protected_quality_ok': quality, 'strict_joint': identity and quality and changed,
                  'paper_structure_identity_screen': identity and quality and disruption,
                  'paper_structure_full_screen': identity and control_quality and quality and disruption}
        if any(r[key] != value for key, value in checks.items()):
            raise ValueError('Stored supplemental gate inconsistent')
        if r['supplemental_protocol_sha256'] != file_sha256(args.evidence / 'supplemental-protocol.json'):
            raise ValueError('Supplemental protocol mismatch')
    output = {
        'protocol_sha256': file_sha256(args.evidence / 'recovery-protocol.json'),
        'selection_sha256': file_sha256(args.evidence / 'recovery-selection.json'),
        'primary_results_sha256': file_sha256(primary_path),
        'supplement_results_sha256': file_sha256(secondary_path),
        'counts_verified': True, 'selection': {'profile': selection['profile']['name'], 'checkpoint': selection['checkpoint']},
        'paper_reference': {'mos': 4.07, 'wer': .0784, 'defend_rate': .8006},
        'limitations': [
            '15 utterances, 10 speakers; two content repetitions are correlated, not 30 independent people.',
            'Macro WER and MOS are deduplicated by protected utterance; conversion ratios are per content pair.',
            'Original MOSNet checkpoint parity is verified; Voice Guard authors exact evaluator checkpoint is unknown.',
            'Eq10 structural interpretation uses WER<=.10; paper ASR threshold terminology is ambiguous.',
            'A zero full-clean denominator is N/A; do not attribute pre-existing conversion failures to perturbation.',
            'No same-utterance original-VG baseline in this round, so no causal improvement claim against old samples.',
            'No human MOS ratings collected; automatic prediction does not establish perceptual equivalence.',
        ], 'evaluations': {},
    }
    cohorts = {'all': set(protocol['validation']),
               'development_speakers_new_sentences': {u for u in protocol['validation'] if int(u[1:4]) <= 229},
               'speakers_not_used_in_attack_development': {u for u in protocol['validation'] if int(u[1:4]) >= 230}}
    for label, rows, supplement in [('ECAPA_Whisper_base', primary, False), ('GE2E_Whisper_small_en', secondary, True)]:
        output['evaluations'][label] = {
            cohort: {'pooled_correlated_pairs': summarize([r for r in rows if r['utterance'] in members], supplement),
                     'by_content': {content: summarize([r for r in rows if r['utterance'] in members and r['content'] == content], supplement)
                                    for content in protocol['evaluation_contents']['validation']}}
            for cohort, members in cohorts.items()}
    write_json(args.evidence / 'recovery-analysis.json', output)
    print(json.dumps(output['evaluations'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
