"""Recompute paired gates, waveform checksums and proxy diagnostics for this study."""
from collections import Counter
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_staged_recovery import summary, cluster_intervals
from tools.decide_blend_validation import eligible
from vguard.discriminant_recovery import acceptable
from vguard.metrics.staged import gates
from vguard.run_state import file_sha256, write_json


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(evidence, run):
    protocol, selection = read(evidence / 'protocol.json'), read(evidence / 'selection.json')
    decision = read(evidence / 'extension-decision.json')
    digest = file_sha256(evidence / 'protocol.json')
    if selection['protocol_sha256'] != digest or decision['protocol_sha256'] != digest:
        raise ValueError('Protocol metadata mismatch')
    if decision['selection_sha256'] != file_sha256(evidence / 'selection.json'):
        raise ValueError('Decision selection mismatch')
    development_rows = read(run / 'development-results.json')['rows']
    allowed, checks = eligible(selection['selected'], selection['baseline'], development_rows)
    if allowed != decision['validation_allowed'] or checks != decision['checks']:
        raise ValueError('Development gate does not recompute')
    if selection['development_results_sha256'] != file_sha256(run / 'development-results.json'):
        raise ValueError('Development changed after selection')
    output = {'counts_verified': True, 'protocol_sha256': digest,
        'selection_sha256': file_sha256(evidence / 'selection.json'),
        'decision_sha256': file_sha256(evidence / 'extension-decision.json'),
        'analysis_code_sha256': file_sha256(__file__), 'validation_allowed': allowed,
        'baseline': protocol['baseline']['name'],
        'challenger': selection['selected']['profile']['name'], 'splits': {},
        'evaluators': {'independent': ['GE2E', 'MOSNet', 'Whisper base', 'Whisper small.en'],
            'training_gradient': ['AdaIN native reference and decoded-mel enrollment LDA', 'wav2vec2-CTC'],
            'training_acceptance': ['ECAPA original-identity similarity', 'spectral distance', 'CTC', 'Native LDA identity margin']},
        'limitations': protocol['limitations']}
    if not allowed and (run / 'validation-state.json').exists():
        raise ValueError('Validation examined despite a failed gate')
    for split in (('development', 'validation') if allowed else ('development',)):
        profiles = [protocol['baseline']['name']] + ([p['name'] for p in protocol['profiles']]
            if split == 'development' else [selection['selected']['profile']['name']])
        rows = read(run / (split + '-results.json'))['rows']
        state = read(run / (split + '-state.json'))
        expected = {(u, c, p) for u in protocol[split] for c in protocol['evaluation_contents'] for p in profiles}
        if state['state'] != 'complete' or state['protocol_sha256'] != digest or state['rows'] != len(expected):
            raise ValueError('Split state incomplete or mismatched')
        if len(rows) != len(expected) or {(r['utterance'], r['content'], r['profile']) for r in rows} != expected:
            raise ValueError('Missing/duplicate split rows')
        if split == 'validation' and state['selection_sha256'] != file_sha256(evidence / 'selection.json'):
            raise ValueError('Validation used another selection')
        controls = {}; optimizations = {}; measured = {}
        for r in rows:
            u, c, name = r['utterance'], r['content'], r['profile']
            if r['protocol_sha256'] != digest: raise ValueError('Row protocol mismatch')
            clean = {k: r[k] for k in ('original', 'clean_vc')}
            pair = u, c
            if pair in controls and controls[pair] != clean: raise ValueError('Clean controls not paired')
            controls[pair] = clean
            control, job = run / 'controls' / u, run / split / name / u
            metadata = read(job / 'optimization.json')
            if r['optimization_sha256'] != file_sha256(job / 'optimization.json'):
                raise ValueError('Optimization changed after scoring')
            if metadata['protocol_sha256'] != digest: raise ValueError('Optimization protocol differs')
            paths = {'original': control / 'original.wav', 'clean_vc': control / (c + '.wav'),
                'defended': job / 'defended.wav', 'protected_vc': job / (c + '.wav')}
            for key, path in paths.items():
                if file_sha256(path) != r[key]['sha256']: raise ValueError('Measured WAV changed: ' + str(path))
                measured[path.as_posix()] = r[key]['sha256']
            for asv, asr in [('ge2e', 'small_en'), ('ecapa', 'base')]:
                if r['gates_' + asv] != gates(u, r['original'], r['clean_vc'], r['defended'],
                        r['protected_vc'], asv=asv, asr=asr):
                    raise ValueError('Gate does not recompute')
            optimizations[u, name] = metadata
        values = {name: {'all': summary([r for r in rows if r['profile'] == name]),
            'by_content': {c: summary([r for r in rows if r['profile'] == name and r['content'] == c])
                for c in protocol['evaluation_contents']}} for name in profiles}
        proxy = {}
        for name in profiles[1:]:
            entries = [v for (u, p), v in optimizations.items() if p == name]
            for entry in entries:
                if not entry['fallback_to_baseline']:
                    actual=entry['actual_pcm_proxy_checks']
                    if not acceptable(actual, entry['proxy_caps']) or actual['defense'] > entry['baseline_proxy_checks']['defense'] + .0001:
                        raise ValueError('Unaccepted export scored without fallback')
            proxy[name] = {'utterances': len(entries),
                'accepted_blocks': sum(v['accepted_blocks'] for v in entries),
                'rejected_blocks': sum(v['rejected_blocks'] for v in entries),
                'content_gradient_conflicts': sum(v['content_gradient_conflicts'] for v in entries),
                'post_export_fallbacks': sum(v['fallback_to_baseline'] for v in entries),
                'identical_to_baseline': sum(v['final_sha256_equals_baseline'] for v in entries),
                'mean_ctc_before': sum(v['baseline_proxy_checks']['ctc'] for v in entries) / len(entries),
                'mean_ctc_proposal_after': sum(v['actual_pcm_proxy_checks']['ctc'] for v in entries) / len(entries),
                'elapsed_optimization_s': sum(v['elapsed_s'] for v in entries)}
        waveform_checks = {}
        for name in profiles:
            snrs, peaks = [], []
            for u in protocol[split]:
                exported, sr = sf.read(run / split / name / u / 'defended.wav', dtype='float32')
                original, original_sr = sf.read(run / 'controls' / u / 'original.wav', dtype='float32')
                if exported.shape != original.shape or sr != original_sr or not np.isfinite(exported).all():
                    raise ValueError('Protected export shape/sample-rate/finite check failed')
                delta = exported - original
                snr = float(10 * np.log10(np.mean(original**2) / max(float(np.mean(delta**2)), 1e-20)))
                saved = next(r for r in rows if r['utterance'] == u and r['profile'] == name)
                if abs(snr - saved['snr_db']) > 1e-5:
                    raise ValueError('Stored SNR does not match export')
                snrs.append(snr); peaks.append(float(np.max(np.abs(exported))))
            waveform_checks[name] = {'snr_min_db': min(snrs), 'snr_max_db': max(snrs),
                'export_abs_peak_max': max(peaks), 'export_layout_and_finite_verified': True,
                'declared_snr_export_tolerance_passed': min(snrs) >= (protocol['baseline']['final_snr_db'] if name == profiles[0] else next(p['snr_db'] for p in protocol['profiles'] if p['name'] == name)) - .01}
        baseline_rows = [r for r in rows if r['profile'] == protocol['baseline']['name']]
        comparisons = {}
        for name in profiles[1:]:
            after = [r for r in rows if r['profile'] == name]
            lookup = {(r['utterance'], r['content']): r for r in after}
            transitions = {endpoint: dict(Counter(
                'both_pass' if a and b else 'baseline_only' if a else 'challenger_only' if b else 'both_fail'
                for r in baseline_rows for a, b in [(r['gates_ge2e'][endpoint],
                    lookup[r['utterance'], r['content']]['gates_ge2e'][endpoint])]))
                for endpoint in ('paper_full_screen', 'strict_joint')}
            comparisons[name] = {'intervals': cluster_intervals(baseline_rows, after, 'gates_ge2e'),
                'transitions': transitions}
            originals_before = {r['utterance']: r for r in baseline_rows}
            originals_after = {r['utterance']: r for r in after}
            comparisons[name]['new_original_quality_failures'] = [
                {'utterance': u, 'reference': before['defended']['reference'],
                    'baseline_hypothesis': before['defended']['hypothesis_small_en'],
                    'candidate_hypothesis': originals_after[u]['defended']['hypothesis_small_en'],
                    'baseline_wer': before['defended']['wer_small_en'],
                    'candidate_wer': originals_after[u]['defended']['wer_small_en'],
                    'baseline_mos': before['defended']['mos_original'],
                    'candidate_mos': originals_after[u]['defended']['mos_original'],
                    'candidate_identity': originals_after[u]['defended']['asv_ge2e']}
                for u, before in originals_before.items()
                if before['gates_ge2e']['protected_quality_ok']
                and not originals_after[u]['gates_ge2e']['protected_quality_ok']]
        output['splits'][split] = {'utterances': len(protocol[split]), 'rows': len(rows),
            'results_sha256': file_sha256(run / (split + '-results.json')),
            'measured_wav_paths': len(measured), 'profiles': values,
            'proxy_diagnostics': proxy, 'waveform_checks': waveform_checks,
            'paired_comparisons': comparisons}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('docs/evidence/2026-11-09-discriminant-blend'))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    run = args.run or Path(read(args.evidence / 'protocol.json')['output'])
    result = audit(args.evidence, run)
    path = args.evidence / 'analysis.json'
    if path.exists() and read(path) != result:
        previous = run.parent / 'initial-validation-analysis.json'
        if previous.exists(): raise ValueError('Preserve the prior analysis snapshot')
        shutil.copyfile(path, previous)
    write_json(path, result)
    print(json.dumps({'validation_allowed': result['validation_allowed'],
        'splits': {name: {'rows': value['rows'], 'profiles': {p: v['all']
            for p, v in value['profiles'].items()}} for name, value in result['splits'].items()}}, indent=2))


if __name__ == '__main__': main()
