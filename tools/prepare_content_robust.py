"""Freeze CTC-preserving reference-defense development and fresh validation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    evidence = Path('docs/evidence/2026-11-09-content-robust')
    destination = evidence / 'protocol.json'
    if destination.exists(): raise ValueError('Content protocol already frozen')
    first_path = Path('docs/evidence/2026-11-09-joint/protocol.json')
    prior = read(first_path)
    second = read('docs/evidence/2026-11-09-joint-anchored/protocol.json')
    development = prior['validation'][::2] + second['validation'][:2]
    if len(development) != 12 or len(set(development)) != 12:
        raise ValueError('Need twelve distinct observed development utterances')
    forbidden = set(prior['development'] + prior['validation'] + second['validation'])
    for path in ('docs/evidence/2026-11-09-quality/recovery-protocol.json',
                 'docs/evidence/2026-11-09-staged/final/protocol.json'):
        value = read(path)
        forbidden.update(value['validation']); forbidden.update(value.get('development', []))
    forbidden.update(r['utterance'] for r in read('docs/evidence/2026-11-09/frozen-protocol.json')['new_utterances'])
    staged = read('docs/evidence/2026-11-09-staged/final/protocol.json')
    forbidden.update(Path(name).stem for name in staged['enrollment_sha256'] if name.endswith('.wav'))
    with np.load('outputs/registry.npz') as bank: speakers = list(bank['names'])[:10]
    validation = []
    for speaker in speakers:
        files = sorted((Path('data/VCTK22') / str(speaker)).glob('*.wav'))
        if len(files) < 75: raise ValueError('Insufficient untouched sentences')
        validation.extend(p.stem for p in files[73:75])
    if set(validation) & forbidden or len(validation) != len(set(validation)):
        raise ValueError('New validation overlaps observed or enrollment inputs')
    assets = {name: sha for name, sha in prior['assets'].items() if not name.startswith('data/')}
    checkpoint = Path('checkpoints/ctc/wav2vec2_fairseq_base_ls960_asr_ls960.pth')
    assets[checkpoint.as_posix()] = file_sha256(checkpoint)
    for u in development + validation + prior['evaluation_contents'] + prior['baseline_optimization_contents']:
        for suffix in ('.wav', '.txt'):
            path = Path('data/VCTK22') / u.split('_')[0] / (u + suffix)
            assets[path.as_posix()] = file_sha256(path)
    smoke = read('outputs/content_robust/smoke.json')
    if not smoke['ctc_gradient_finite'] or not smoke['models_frozen'] or not smoke['unit_tests_passed']:
        raise ValueError('Real gradient and unit checks required before freezing')
    code = dict(prior['code_sha256'])
    for name in ('vguard/content_robust_recovery.py', 'scripts/run_content_robust.py',
            'tools/prepare_content_robust.py', 'tools/decide_content_validation.py',
            'tools/smoke_content_robust.py', 'tests/test_content_robust.py'):
        code[name] = file_sha256(name)
    # Existing frozen dependencies must still match their actual executed bytes.
    for name, sha in {**assets, **code}.items():
        if file_sha256(name) != sha: raise ValueError('Frozen dependency changed: ' + name)
    protocol = {**prior, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'output': 'outputs/content_robust/2026-11-09', 'seed': 2026110905,
        'type': 'CTC-preserving time-frequency allocated native-reference defense with EOT ablation',
        'development': development, 'validation': validation,
        'development_design': 'Twelve previously evaluated utterances from ten enrolled speakers, intentionally reused as development; no claim that these are untouched validation.',
        'validation_examined': False,
        'validation_selection': 'Sorted original sentence indices 73:75 for each of the first ten enrolled speakers; exclude all previous optimization, validation and enrollment sentences.',
        'optimization_contents': [],
        'profiles': [{'name': 'ctc_tf_snr20', 'eot': False}, {'name': 'ctc_tf_eot_snr20', 'eot': True}],
        'refinement': {'steps': 160, 'learning_rate': .00003, 'snr_db': 20., 'max_seconds': 420.},
        'gradient_design': 'Native AdaIN reference identity target retains the baseline objective; no output Griffin-Lim gradient. Frozen wav2vec2-CTC supplies original-content gradients. Normalize identity/content input gradients separately, build a smoothed STFT sensitivity gate in [.1,1], remove first-order conflicting CTC direction, and enforce existing masking/SNR/peak projection.',
        'proxy_acceptance': 'Every eight updates accept only if CTC <= baseline+.01, ECAPA original similarity >= baseline-.005, relative spectral distance <= baseline+.01, native identity objective <= baseline+.005 and mean training identity objective <= previous accepted+.0001. Rejected blocks restore the previous accepted waveform. Actual PCM16 caps are checked before independent scoring; failure falls back to the same baseline WAV.',
        'content_supervision': 'Ground-truth reference text, uppercase letters/apostrophe/spaces normalized for torchaudio WAV2VEC2_ASR_BASE_960H. Frozen evaluation Basic normalization and WER<=10% remain unchanged. Full reference context retained for CTC.',
        'eot_channels': ['identity', 'two fixed first-order image-source room responses', 'three-tap microphone response', '8 kHz down/up sampling'],
        'environment_limits': 'Fixed synthetic reference preprocessing, not a measured room, full acoustic simulator, platform recording or complete real-world channel distribution.',
        'warm_start': 'Rerun the same fixed native baseline for each utterance; both refinements use its exact PCM16 export. All checkpoint acceptance uses training proxies, never final GE2E/MOSNet/Whisper scores.',
        'validation_gate': 'Selected candidate must strictly improve development full-clean success count, not decrease strict success count or unique original-quality count, preserve every baseline-passing original, macro WER <= baseline+.005, and predicted MOS >= baseline-.05. Otherwise stop without scoring fresh validation.',
        'limits': {'development_hours': 1., 'validation_hours': 1.3, 'automatic_shutdown': False},
        'source_note': 'Inspired by general ideas described in the user-provided CSDN article, not a verified Kalypta implementation. No Kalypta performance, SDK or model provenance claim.',
        'source_url': 'https://blog.csdn.net/weixin_33139275/article/details/170178909',
        'prior_protocol_sha256': file_sha256(first_path),
        'smoke_sha256': file_sha256('outputs/content_robust/smoke.json'),
        'assets': assets, 'code_sha256': code,
        'limitations': ['Twelve observed development utterances; validation only if the fixed gate passes.',
            'Twenty fresh sentences from ten enrolled speakers, not unseen-pretraining people.',
            'CTC and local gradient constraints do not guarantee independent WER, identity or human quality.',
            'ECAPA participates in proxy acceptance and is not independent evidence.',
            'Fixed synthetic channels do not establish robustness to measured rooms or arbitrary preprocessing.',
            'No human listening study, second VC model, real-time generator, or paper-scale equivalence claim.']}
    write_json(destination, protocol)
    write_json(evidence / 'smoke.json', smoke)
    print(destination, file_sha256(destination), 'development', development, 'validation', validation)


if __name__ == '__main__': main()
