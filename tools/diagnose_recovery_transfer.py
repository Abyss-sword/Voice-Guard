"""Measure where the existing perturbation loses identity changes, without optimization."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import torch
import librosa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.features import wav2mel_tensor
from vguard.run_state import file_sha256, write_json
from vguard.vc.adainvc import load_adainvc, get_spectrograms, normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=Path('outputs/registry_vc.npz'))
    parser.add_argument('--out', type=Path, default=Path('outputs/paper_quality/transfer-diagnosis.json'))
    args = parser.parse_args()
    if args.out.exists(): raise ValueError('Preserve prior diagnostic output; use a new path')
    torch.set_num_threads(1)
    root = Path('outputs/paper_quality/2026-11-09-recovery')
    protocol = json.loads(Path('docs/evidence/2026-11-09-quality/recovery-protocol.json').read_text())
    rows = json.loads((root / 'validation-results.json').read_text())['rows']
    secondary = {(r['utterance'], r['content']): r for r in json.loads((root / 'validation-supplement.json').read_text())['rows']}
    model, config, attr = load_adainvc(protocol['model_dir'], 'cuda')
    pp = config['preprocess']
    kwargs = {k: pp[k] for k in ('sample_rate', 'preemph', 'n_fft', 'hop_length', 'win_length', 'n_mels', 'ref_db', 'max_db', 'top_db')}
    with np.load(args.registry) as package:
        names = list(package['names']); registry = torch.as_tensor(package['embeddings'], device='cuda')
    def identify(embedding):
        sims = torch.nn.functional.cosine_similarity(registry, embedding.reshape(1, -1), dim=1)
        return {'prediction': str(names[int(sims.argmax())]), 'norm': float(embedding.norm()),
                'similarities': {str(n): float(v) for n, v in zip(names, sims)}}
    def mel(path):
        value, _ = get_spectrograms(str(path), **pp)
        return torch.from_numpy(normalize(value, attr)).T.unsqueeze(0).cuda()
    results = []
    with torch.inference_mode():
        for row in rows:
            u, c = row['utterance'], row['content']
            clean = root / 'controls' / u
            defended = root / 'validation' / row['profile'] / u / f"step_{row['checkpoint']}"
            source = Path(protocol['vctk']) / c.split('_')[0] / (c + '.wav')
            mu, _ = model.content_encoder(mel(source))
            record = {'utterance': u, 'content': c, 'input': {}, 'decoded_mel': {}, 'resynthesized': {}, 'feature_alignment': {}}
            for label, path in [('clean', clean / 'original.wav'), ('protected', defended / 'defended.wav')]:
                native = mel(path)
                wave, sr = sf.read(path, dtype='float32')
                crop, interval = librosa.effects.trim(wave, top_db=pp['top_db'])
                differentiable = wav2mel_tensor(torch.as_tensor(crop, device='cuda'), attr=attr, **kwargs)
                embedding = model.speaker_encoder(native)
                record['input'][label] = identify(embedding)
                decoded = model.decoder(mu, embedding)
                record['decoded_mel'][label] = identify(model.speaker_encoder(decoded))
                converted = (clean if label == 'clean' else defended) / (c + '.wav')
                record['resynthesized'][label] = identify(model.speaker_encoder(mel(converted)))
                record['feature_alignment'][label] = {
                    'max_normalized_mel_abs_error': float((native - differentiable).abs().max()),
                    'mean_normalized_mel_abs_error': float((native - differentiable).abs().mean()),
                    'native_vs_differentiable_embedding_cosine': float(torch.nn.functional.cosine_similarity(embedding, model.speaker_encoder(differentiable))),
                    'trim_interval': list(map(int, interval)), 'wav_sha256': file_sha256(path)}
            record['independent'] = {'clean_ecapa': row['control']['converted']['asv'], 'protected_ecapa': row['converted']['asv'],
                'clean_ge2e': secondary[(u, c)]['scores']['clean_vc']['asv_ge2e'],
                'protected_ge2e': secondary[(u, c)]['scores']['protected_vc']['asv_ge2e']}
            results.append(record)
    summary = {}
    for stage in ('input', 'decoded_mel', 'resynthesized'):
        for label in ('clean', 'protected'):
            values = [r for r in results if r['content'] == protocol['evaluation_contents']['validation'][0]] if stage == 'input' else results
            summary[stage + '_' + label] = {'identity_correct': sum(r[stage][label]['prediction'] == r['utterance'].split('_')[0] for r in values),
                'total': len(values), 'mean_norm': float(np.mean([r[stage][label]['norm'] for r in values]))}
    write_json(args.out, {'summary': summary, 'rows': results,
        'diagnostic_only': True, 'old_validation_is_now_examined': True,
        'registry_sha256': file_sha256(args.registry), 'source_sha256': file_sha256(__file__),
        'registry_norms': {str(n): float(v.norm()) for n, v in zip(names, registry)},
        'new_experiments_require_new_validation': True})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
