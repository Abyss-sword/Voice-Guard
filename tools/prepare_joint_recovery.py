"""Freeze the fourth experiment before any new validation audio is scored."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def main():
    destination = Path('docs/evidence/2026-11-09-joint/protocol.json')
    if destination.exists(): raise ValueError('Joint protocol already frozen')
    prior_path = Path('docs/evidence/2026-11-09-staged/final/protocol.json')
    prior = json.loads(prior_path.read_text())
    validation = []
    with np.load('outputs/registry.npz') as package:
        speakers = list(package['names'])[:10]
    for speaker in speakers:
        files = sorted((Path('data/VCTK22') / str(speaker)).glob('*.wav'))
        if len(files) < 72: raise ValueError('Need at least 72 original sentences for ' + str(speaker))
        validation.extend(p.stem for p in files[70:72])
    forbidden = set(prior['development']) | set(prior['validation'])
    old = json.loads(Path('docs/evidence/2026-11-09-quality/recovery-protocol.json').read_text())
    forbidden |= set(old['validation'])
    oldest = json.loads(Path('docs/evidence/2026-11-09/frozen-protocol.json').read_text())
    forbidden |= {r['utterance'] for r in oldest['new_utterances']}
    if set(validation) & forbidden or len(validation) != len(set(validation)):
        raise ValueError('Validation overlaps examined optimization sentences')
    if any(f'data/VCTK22/{u.split("_")[0]}/{u}.wav' in prior['enrollment_sha256'] for u in validation):
        raise ValueError('Validation overlaps enrollment')
    assets = {name: digest for name, digest in prior['assets'].items() if not name.startswith('data/')}
    optimization = ['p225_002', 'p225_003']
    evaluation = ['p226_002', 'p227_002', 'p225_003', 'p226_004']
    for u in prior['development'] + validation + optimization + evaluation:
        for suffix in ('.wav', '.txt'):
            path = Path('data/VCTK22') / u.split('_')[0] / (u + suffix)
            assets[path.as_posix()] = file_sha256(path)
    code = ['vguard/joint_recovery.py', 'scripts/run_joint_recovery.py',
        'tools/prepare_joint_recovery.py', 'tools/analyze_joint_recovery.py',
        'tools/analyze_staged_recovery.py', 'vguard/staged_recovery.py',
        'vguard/recovery.py', 'vguard/features.py', 'vguard/masking.py',
        'vguard/vc/adainvc.py', 'vguard/asv.py', 'vguard/audio.py',
        'vguard/metrics/staged.py', 'vguard/metrics/wer.py', 'vguard/metrics/mos_original.py',
        'vguard/run_state.py', 'third_party/ada_in_vc/model.py']
    smoke_path = Path('outputs/joint_recovery/smoke.json')
    smoke = json.loads(smoke_path.read_text())
    if not smoke['gradient_finite'] or smoke['waveform_correlation'] < .99:
        raise ValueError('Real waveform smoke check failed')
    profiles = [
        {'name': 'output_only_snr20', 'guarded': False},
        {'name': 'joint_guard_snr20', 'guarded': True},
    ]
    protocol = {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'output': 'outputs/joint_recovery/2026-11-09', 'seed': 2026110904,
        'type': 'Output-waveform identity separation and protected-original quality constraints',
        'development': prior['development'], 'validation': validation,
        'validation_examined': False, 'validation_selection': 'Local sorted sentences 70:72 for the first ten registered speakers, excluding all prior optimization and enrollment sentences; frozen before scores.',
        'optimization_contents': optimization, 'evaluation_contents': evaluation,
        'content_design': 'Original two source utterances retained as anchors; two fixed new utterances add distinct text. p225_003 is used for optimization, p226_004 text is held out. No source chosen by validation quality.',
        'profiles': profiles, 'baseline': {'name': 'native_emb_snr20', 'objective': 'embedding',
            'strong_steps': 3000, 'conceal_steps': 1500, 'strong_snr_db': 15., 'final_snr_db': 20.,
            'learning_rate': .001, 'final_margin_db': 0.},
        'baseline_optimization_contents': prior['optimization_contents'],
        'native_target_bank': prior['native_target_bank'], 'target_map': prior['target_map'],
        'refinement': {'steps': 240, 'learning_rate': .00015, 'snr_db': 20.,
            'gradient_iterations': 2, 'max_seconds': 600.},
        'gradient_design': 'Official 100-iteration Griffin-Lim forward with float64 phase loop; only last 2 iterations differentiated. ECAPA is a frozen optimization surrogate, not independent evaluation. Raw reference is guarded by ECAPA similarity/class margin, content-latent consistency and multi-resolution spectra; not direct ASR or human MOS constraints.',
        'warm_start': 'Same exported baseline waveform for both refinements, same refinement seed. No evaluation-based checkpoint choice; fixed final step.',
        'primary_evaluation': 'Held-out GE2E + original MOSNet + Whisper small.en; MOS>=3/WER<=.10 frozen Basic normalizer. English-normalized WER diagnostic only.',
        'secondary_evaluation': 'ECAPA/base retained but ECAPA participates in optimization and is explicitly non-independent.',
        'selection_rule': 'One global challenger selected on development only: maximize full-clean Eq10 success count, strict identity joint count, unique protected-quality count, lower macro WER, higher protected MOS, profile name. Compare this tuple to baseline and retain baseline recommendation unless strictly better. Selected challenger still gets one frozen paired validation for an honest test.',
        'reporting': 'Report clean-control coverage, complete joint endpoint, identity-only-screen endpoint, strict identity endpoint, raw counts, per-content/text strata, speaker-cluster paired intervals, and all failed candidates.',
        'limits': {'development_hours': .5, 'validation_hours': 1.3, 'automatic_shutdown': False},
        'prior_protocol_sha256': file_sha256(prior_path), 'smoke_sha256': file_sha256(smoke_path),
        'code_sha256': {p: file_sha256(p) for p in code}, 'assets': assets,
        'whisper_sha256': prior['whisper_sha256'], 'ge2e_weights_sha256': prior['ge2e_weights_sha256'],
        'limitations': ['Small enrolled-speaker sample; two new sentences per speaker.',
            'This is a new content protocol with paired baseline; do not compare its rates as if they used the previous content denominator.',
            'Last-two-iteration phase gradient is approximate; all final metrics use official exported conversions.',
            'No human listening study or second VC model; no claim of paper equivalence.']}
    # Also verify existing assets, rather than merely copying their old digests.
    for name, digest in assets.items():
        if file_sha256(name) != digest: raise ValueError('Asset changed: ' + name)
    write_json(destination, protocol)
    print(destination, file_sha256(destination), 'validation', validation, flush=True)


if __name__ == '__main__': main()
