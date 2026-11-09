"""Inspect native speaker geometry on already observed development audio only."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json
from vguard.vc.adainvc import load_adainvc, get_spectrograms, normalize


def main():
    output = Path('outputs/geometry/diagnosis.json')
    if output.exists(): raise ValueError('Preserve prior diagnostic output')
    torch.set_num_threads(1)
    evidence = Path('docs/evidence/2026-11-09-content-robust')
    protocol = json.loads((evidence / 'protocol.json').read_text())
    previous = json.loads((Path(protocol['output']) / 'development-results.json').read_text())['rows']
    model, config, attr = load_adainvc('checkpoints/adain-vc-official', 'cuda')
    with np.load(protocol['native_target_bank']) as p:
        names = list(p['names'])
        bank = torch.tensor(p['embeddings'], device='cuda') if 'embeddings' in p else torch.tensor(p['exemplars'].mean(1), device='cuda')
    rows = []
    with torch.no_grad():
        for u in protocol['development']:
            own = names.index(u.split('_')[0])
            entry = {'utterance': u}
            for label, path in [('original', Path(protocol['output']) / 'controls' / u / 'original.wav'),
                    ('baseline', Path(protocol['output']) / 'development' / protocol['baseline']['name'] / u / 'defended.wav')]:
                mel, _ = get_spectrograms(str(path), **config['preprocess'])
                emb = model.speaker_encoder(torch.tensor(normalize(mel, attr).T[None], device='cuda'))
                sims = F.cosine_similarity(emb, bank)
                indices = sims.argsort(descending=True).tolist()
                nearest_other = next(i for i in indices if i != own)
                target = names.index(protocol['target_map'][u.split('_')[0]])
                entry[label] = {'norm': float(emb.norm()), 'prediction': names[indices[0]],
                    'own_cosine': float(sims[own]), 'target_cosine': float(sims[target]),
                    'nearest_other': names[nearest_other], 'nearest_other_cosine': float(sims[nearest_other]),
                    'margin': float(sims[own]-sims[nearest_other]), 'wav_sha256': file_sha256(path)}
            entry['baseline_observations'] = [{'content': r['content'],
                'ge2e_owner_retained': r['protected_vc']['asv_ge2e'] == u.split('_')[0],
                'original_quality': r['gates_ge2e']['protected_quality_ok']}
                for r in previous if r['utterance'] == u and r['profile'] == protocol['baseline']['name']]
            rows.append(entry)
    result = {'diagnostic_only': True, 'development_already_observed': True,
        'source_sha256': file_sha256(__file__), 'rows': rows,
        'baseline_native_owner_retained': sum(r['baseline']['prediction'] == r['utterance'].split('_')[0] for r in rows),
        'mean_baseline_margin': float(np.mean([r['baseline']['margin'] for r in rows])),
        'mean_baseline_target_cosine': float(np.mean([r['baseline']['target_cosine'] for r in rows]))}
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
