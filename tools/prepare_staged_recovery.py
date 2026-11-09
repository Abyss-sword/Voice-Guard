"""Build native VC target references and freeze a fresh paired validation protocol."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json
from vguard.vc.adainvc import load_adainvc, get_spectrograms, normalize


def main():
    directory = Path('docs/evidence/2026-11-09-staged')
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / 'protocol.json'
    if frozen.exists(): raise ValueError('Protocol already frozen')
    output = Path('outputs/staged_recovery/2026-11-09')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model, config, attr = load_adainvc('checkpoints/adain-vc-official', 'cuda')
    with np.load('outputs/registry.npz') as package:
        names = list(package['names']); ecapa = package['embeddings']
    vectors, exemplars, enrollment = [], [], {}
    with torch.inference_mode():
        for name in names:
            files = sorted((Path('data/VCTK22') / name).glob('*.wav'))[:30]
            if len(files) != 30: raise ValueError('Need 30 enrollment references')
            embeddings = []
            for path in files:
                mel, _ = get_spectrograms(str(path), **config['preprocess'])
                native = torch.from_numpy(normalize(mel, attr)).T.unsqueeze(0).cuda()
                embeddings.append(model.speaker_encoder(native).squeeze().cpu().numpy())
                enrollment[path.as_posix()] = file_sha256(path)
            vectors.append(np.mean(embeddings, axis=0)); exemplars.append(np.stack(embeddings[:4]))
            print('native enrollment', name, flush=True)
    bank = output / 'native-target-bank.npz'
    np.savez(bank, names=np.asarray(names), embeddings=np.asarray(vectors), exemplars=np.asarray(exemplars))
    target_map = {}
    for index, name in enumerate(names):
        sims = ecapa @ ecapa[index] / (np.linalg.norm(ecapa, axis=1) * np.linalg.norm(ecapa[index]) + 1e-9)
        sims[index] = np.inf
        target_map[str(name)] = str(names[int(np.argmin(sims))])
    development = ['p225_040', 'p226_039', 'p227_033']
    validation = []
    for name in names[:10]:
        files = sorted((Path('data/VCTK22') / str(name)).glob('*.wav'))
        if len(files) < 62: raise ValueError('Download extra validation utterances first')
        validation.extend(p.stem for p in files[60:62])
    prior = json.loads(Path('docs/evidence/2026-11-09-quality/recovery-protocol.json').read_text())
    old = json.loads(Path('docs/evidence/2026-11-09/frozen-protocol.json').read_text())
    forbidden = set(prior['validation']) | set(development) | {r['utterance'] for r in old['new_utterances']}
    if set(validation) & forbidden: raise ValueError('New validation overlaps examined optimized utterances')
    if any(any(path.endswith('/' + u + '.wav') for path in enrollment) for u in validation):
        raise ValueError('Validation overlaps enrollment')
    contents = ['p226_002', 'p225_002', 'p227_002']
    profiles = [
        {'name': 'native_emb_snr20', 'objective': 'embedding', 'final_snr_db': 20.},
        {'name': 'native_fb_snr20', 'objective': 'feedback', 'final_snr_db': 20.},
        {'name': 'native_e2e_snr18', 'objective': 'e2e', 'final_snr_db': 18.},
    ]
    assets = {path: digest for path, digest in prior['assets'].items() if not path.startswith('data/')}
    for u in development + validation + contents:
        for suffix in ('.wav', '.txt'):
            path = Path('data/VCTK22') / u.split('_')[0] / (u + suffix)
            assets[path.as_posix()] = file_sha256(path)
    assets[bank.as_posix()] = file_sha256(bank)
    assets['outputs/paper_quality/ge2e-registry.npz'] = file_sha256('outputs/paper_quality/ge2e-registry.npz')
    code = ['vguard/staged_recovery.py', 'vguard/metrics/staged.py', 'scripts/run_staged_recovery.py',
            'vguard/recovery.py', 'vguard/features.py', 'vguard/masking.py', 'vguard/vc/adainvc.py',
            'vguard/metrics/wer.py', 'vguard/metrics/mos_original.py', 'vguard/asv.py',
            'vguard/audio.py', 'third_party/ada_in_vc/model.py']
    resource = Path.home() / '.cache/whisper'
    import resemblyzer
    protocol = {
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'output': output.as_posix(),
        'type': 'Third round, native enrollment and staged recovery with matched previous-method baseline',
        'development': development, 'validation': validation, 'seed': 2026110903,
        'optimization_contents': contents[:2], 'evaluation_contents': [contents[0], contents[2]],
        'profiles': profiles, 'strong_steps': 3000, 'conceal_steps': 1500, 'strong_snr_db': 15.,
        'learning_rate': .001, 'final_margin_db': 0., 'native_target_bank': bank.as_posix(),
        'target_selection': 'Least similar ECAPA enrollment identity; raw native AdaIN-VC references from first four enrolled utterances. ECAPA is not an independent target-selector check; GE2E supplies the primary independent identity result.',
        'target_map': target_map, 'enrollment_sha256': enrollment,
        'primary_evaluation': 'GE2E + original MOSNet + Whisper small.en BasicTextNormalizer WER',
        'secondary_evaluation': 'ECAPA + original MOSNet + Whisper base BasicTextNormalizer WER',
        'normalization_sensitivity': 'Whisper EnglishTextNormalizer WER also recorded, never used to select or change primary gates',
        'clean_gates': 'Identity-correct and additionally full MOS>=3/WER<=.10 controls both reported. Full-clean success is the primary Eq10 structural endpoint.',
        'selection_rule': 'Development only: maximize GE2E strict joint count, then full-clean Eq10 joint count, protected-quality count, lower protected macro WER, higher MOS, higher SNR budget, profile name. Baseline is reference, not a candidate. One global profile.',
        'baseline': {'method': 'Frozen second-round feedback_snr20/1500, Es target prototype selection, margin3dB',
                     'steps': 1500, 'source_protocol_sha256': file_sha256('docs/evidence/2026-11-09-quality/recovery-protocol.json')},
        'baseline_pairing': 'Same fresh validation utterances, PCM16 full context, same conversion seed and fixed contents; no historical different-utterance comparison.',
        'known_changes': 'MSE/native target context/two-stage optimization/projection continuation change together; matched baseline measures combined change, not separate causal attribution.',
        'limits': {'development_hours': .75, 'validation_hours': 1.25, 'automatic_shutdown': False},
        'code_sha256': {p: file_sha256(p) for p in code}, 'assets': assets,
        'whisper_sha256': {name: file_sha256(resource / (name + '.pt')) for name in ('base', 'small.en')},
        'ge2e_weights_sha256': file_sha256(Path(resemblyzer.__file__).parent / 'pretrained.pt'),
        'old_results_preserved': True, 'prior_validation_examined': True,
        'validation_independence_note': 'New sentences from ten enrolled speakers; five speakers not used for attack development, not guaranteed absent from VC pretraining.',
    }
    write_json(frozen, protocol)
    print('Frozen', frozen, file_sha256(frozen), 'validation', validation)


if __name__ == '__main__': main()
