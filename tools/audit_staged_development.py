"""Exploratory strong-vs-hidden diagnostics on development only, after selection."""
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.metrics.staged import StagedEvaluator, gates
from vguard.run_state import file_sha256, utterance_seed, write_json
from vguard.vc.adainvc import load_adainvc, convert


def main():
    evidence = Path('docs/evidence/2026-11-09-staged')
    protocol = json.loads((evidence / 'protocol.json').read_text())
    selection = json.loads((evidence / 'selection.json').read_text())
    root = Path(protocol['output'])
    selected = selection['selected']['profile']['name']
    if json.loads((root / 'development-state.json').read_text())['state'] != 'complete':
        raise ValueError('Complete development and frozen selection required')
    destination = root / 'strong-audit'; destination.mkdir(exist_ok=True)
    output = destination / 'results.json'
    if output.exists(): raise ValueError('Preserve original diagnostic')
    torch.set_num_threads(1)
    evaluator = StagedEvaluator(root / 'strong-audit-cache.json')
    model, config, attr = load_adainvc('checkpoints/adain-vc-official', 'cuda')
    primary = json.loads((root / 'development-results.json').read_text())['rows']
    rows = []
    for row in primary:
        if row['profile'] != selected: continue
        u, c = row['utterance'], row['content']
        strong = root / 'development' / selected / u / 'strong.wav'
        content = Path('data/VCTK22') / c.split('_')[0] / (c + '.wav')
        original_ref = row['original']['reference']
        strong_score = evaluator.score(strong, original_ref)
        folder = destination / u; folder.mkdir(exist_ok=True)
        converted = folder / (c + '.wav')
        seed = utterance_seed(protocol['seed'], u + ':' + c)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        convert(model, config, attr, str(content), str(strong), str(converted), 'cuda')
        converted_score = evaluator.score(converted, content.with_suffix('.txt').read_text().strip())
        result = {'utterance': u, 'content': c, 'profile': selected, 'strong': strong_score,
            'strong_vc': converted_score,
            'strong_ge2e_gates': gates(u, row['original'], row['clean_vc'], strong_score, converted_score),
            'final_ge2e_gates': row['gates_ge2e'], 'final_mos': row['defended']['mos_original'],
            'final_wer': row['defended']['wer_small_en']}
        rows.append(result)
        write_json(output, {'rows': rows, 'selection_sha256': file_sha256(evidence / 'selection.json'),
            'purpose': 'Exploratory mechanism diagnosis on three development utterances only; not used to select or retune the frozen candidate.',
            'validation_examined': False})
        print(u, c, result['strong_ge2e_gates'], flush=True)


if __name__ == '__main__': main()
