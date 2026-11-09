"""Post-validation mechanism audit: native Es versus independent GE2E identities.

This does not change waveforms, evaluator scores, selection, or primary gates.
"""
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
    evidence = Path('docs/evidence/2026-11-09-staged/final')
    protocol = json.loads((evidence / 'protocol.json').read_text())
    root = Path(protocol['output'])
    state = json.loads((root / 'validation-state.json').read_text())
    if state['state'] != 'complete': raise ValueError('Complete frozen validation required')
    if state['protocol_sha256'] != file_sha256(evidence / 'protocol.json'):
        raise ValueError('Validation protocol differs')
    destination = root / 'native-validation-audit.json'
    if destination.exists(): raise ValueError('Preserve completed audit')
    torch.set_num_threads(1)
    model, config, attr = load_adainvc('checkpoints/adain-vc-official', 'cuda')
    with np.load(protocol['native_target_bank']) as bank:
        names, embeddings = list(bank['names']), bank['embeddings']
    cache = {}
    @torch.inference_mode()
    def classify(path, expected):
        digest = file_sha256(path)
        if digest != expected: raise ValueError('Scored WAV changed')
        if digest not in cache:
            mel, _ = get_spectrograms(str(path), **config['preprocess'])
            value = torch.from_numpy(normalize(mel, attr)).T.unsqueeze(0).cuda()
            vector = model.speaker_encoder(value).squeeze().cpu().numpy()
            sims = embeddings @ vector / (np.linalg.norm(embeddings, axis=1)*np.linalg.norm(vector)+1e-9)
            cache[digest] = {'identity': str(names[int(sims.argmax())]),
                'similarities': {str(n): float(s) for n, s in zip(names, sims)}, 'sha256': digest}
        return cache[digest]
    result = []
    for row in json.loads((root / 'validation-results.json').read_text())['rows']:
        u, c, p = row['utterance'], row['content'], row['profile']
        control, protected = root / 'controls' / u, root / 'validation' / p / u
        paths = {'original': control / 'original.wav', 'clean_vc': control / (c+'.wav'),
            'defended': protected / 'defended.wav', 'protected_vc': protected / (c+'.wav')}
        result.append({'utterance': u, 'content': c, 'profile': p,
            'native': {key: classify(path, row[key]['sha256']) for key, path in paths.items()},
            'ge2e': {key: row[key]['asv_ge2e'] for key in paths}})
    summaries = {}
    for profile in sorted({r['profile'] for r in result}):
        rows = [r for r in result if r['profile'] == profile]
        unique = {r['utterance']: r for r in rows}
        expected = lambda r: r['utterance'].split('_')[0]
        summaries[profile] = {'utterances': len(unique), 'conversion_pairs': len(rows),
            'native_original_identity_retained': sum(r['native']['original']['identity']==expected(r) for r in unique.values()),
            'native_protected_original_identity_retained': sum(r['native']['defended']['identity']==expected(r) for r in unique.values()),
            'native_clean_converted_identity_retained': sum(r['native']['clean_vc']['identity']==expected(r) for r in rows),
            'native_protected_converted_identity_retained': sum(r['native']['protected_vc']['identity']==expected(r) for r in rows),
            'ge2e_protected_converted_identity_retained': sum(r['ge2e']['protected_vc']==expected(r) for r in rows),
            'native_only_converted_identity_changed': sum(r['native']['protected_vc']['identity']!=expected(r) and r['ge2e']['protected_vc']==expected(r) for r in rows),
            'ge2e_only_converted_identity_changed': sum(r['native']['protected_vc']['identity']==expected(r) and r['ge2e']['protected_vc']!=expected(r) for r in rows)}
    write_json(destination, {'protocol_sha256': file_sha256(evidence / 'protocol.json'),
        'audit_code_sha256': file_sha256(__file__), 'summaries': summaries, 'rows': result,
        'purpose': 'Post-validation mechanism audit only; no waveform changes or configuration selection. Native correctness is not an independent protection success.'})
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__': main()
