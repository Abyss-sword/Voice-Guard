"""Locate clean-content loss: original, GL reconstruction and self-conversion.

This diagnostic never selects a method/content or changes the frozen protocol.
"""
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.audio import load_audio
from vguard.metrics.staged import StagedEvaluator
from vguard.run_state import file_sha256, write_json
from vguard.vc.adainvc import load_adainvc, get_spectrograms, melspectrogram2wav, convert


def main():
    evidence = Path('docs/evidence/2026-11-09-joint')
    destination = evidence / 'clean-chain-diagnostic.json'
    if destination.exists(): raise ValueError('Preserve existing clean diagnostic')
    protocol = json.loads((evidence / 'protocol.json').read_text())
    output = Path(protocol['output']) / 'clean-chain-diagnostic'
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model, config, attr = load_adainvc('checkpoints/adain-vc-official', 'cuda')
    pp = config['preprocess']
    evaluator = StagedEvaluator(output / 'scores.json')
    rows = []
    for content in protocol['evaluation_contents']:
        source = Path('data/VCTK22') / content.split('_')[0] / (content + '.wav')
        reference = source.with_suffix('.txt').read_text().strip()
        source_dir = output / content; source_dir.mkdir(exist_ok=True)
        mel, _ = get_spectrograms(str(source), **pp)
        waves = {'original': load_audio(source, pp['sample_rate']),
            'mel_griffin_lim': melspectrogram2wav(mel, **pp)}
        for name, wave in waves.items():
            path = source_dir / (name + '.wav')
            if not path.exists(): sf.write(path, wave, pp['sample_rate'], subtype='PCM_16')
        self_path = source_dir / 'self_conversion.wav'
        if not self_path.exists(): convert(model, config, attr, str(source), str(source), str(self_path), 'cuda')
        for name in ('original', 'mel_griffin_lim', 'self_conversion'):
            path = source_dir / (name + '.wav')
            score = evaluator.score(path, reference)
            row = {'content': content, 'stage': name, 'path': path.as_posix(), 'score': score}
            rows.append(row)
            print(json.dumps({'content': content, 'stage': name, 'wer': score['wer_small_en'],
                'mos': score['mos_original'], 'hypothesis': score['hypothesis_small_en']}, ensure_ascii=False), flush=True)
    summaries = {}
    for name in ('original', 'mel_griffin_lim', 'self_conversion'):
        values = [r['score'] for r in rows if r['stage'] == name]
        summaries[name] = {'samples': len(values), 'mean_mos': float(np.mean([s['mos_original'] for s in values])),
            'mean_wer': float(np.mean([s['wer_small_en'] for s in values])),
            'content_quality_pass': sum(s['mos_original'] >= 3 and s['wer_small_en'] <= .1 for s in values)}
    write_json(destination, {'protocol_sha256': file_sha256(evidence / 'protocol.json'),
        'code_sha256': file_sha256(__file__), 'rows': rows, 'summaries': summaries,
        'note': 'Diagnostic only, all four frozen content sources included. These stages are not additive causal effects; final clean cross-speaker controls are reported separately. Frozen gates/content/method selection remain unchanged.'})


if __name__ == '__main__': main()
