"""Freeze then evaluate fixed unoptimized reference channels on observed sentences.

This is a development robustness diagnostic, not fresh-sentence validation.
Every method and channel uses its matching clean reference conversion.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.content_robust_recovery import first_order_rir
from vguard.metrics.staged import StagedEvaluator, gates
from vguard.run_state import file_sha256, write_json
from vguard.vc.adainvc import load_adainvc, convert
from tools.analyze_staged_recovery import rate

EVIDENCE = Path('docs/evidence/2026-11-09-content-robust')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def channel_summary(rows):
    """Summarize channel gates without inventing post-channel masking metrics."""
    unique = {r['utterance']: r for r in rows}
    identity = sum(r['gates_ge2e']['clean_identity_ok'] for r in rows)
    full = sum(r['gates_ge2e']['clean_full_quality_ok'] for r in rows)
    return {'utterances': len(unique), 'conversion_pairs': len(rows),
        'protected_quality_count': sum(r['gates_ge2e']['protected_quality_ok'] for r in unique.values()),
        'protected_mos_mean': float(np.mean([r['defended']['mos_original'] for r in unique.values()])),
        'protected_wer_macro': float(np.mean([r['defended']['wer_small_en'] for r in unique.values()])),
        'strict_joint': rate(sum(r['gates_ge2e']['strict_joint'] for r in rows), identity),
        'paper_identity_screen': rate(sum(r['gates_ge2e']['paper_identity_screen'] for r in rows), identity),
        'paper_full_screen': rate(sum(r['gates_ge2e']['paper_full_screen'] for r in rows), full)}


def freeze():
    path = EVIDENCE / 'channel-protocol.json'
    if path.exists(): raise ValueError('Channel protocol already frozen')
    protocol = read(EVIDENCE / 'protocol.json')
    if (Path(protocol['output']) / 'channel-results.json').exists():
        raise ValueError('Channel outcomes already observed')
    write_json(path, {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'main_protocol_sha256': file_sha256(EVIDENCE / 'protocol.json'),
        'code_sha256': file_sha256(__file__),
        'utterances': protocol['development'][:4], 'content': 'p226_004',
        'profiles': [protocol['baseline']['name']] + [p['name'] for p in protocol['profiles']],
        'channels': ['identity', 'heldout_room', 'mp3_64k'],
        'room': {'dimensions': [6., 4.5, 2.8], 'source': [.8, 1.2, 1.4],
            'microphone': [4., 3., 1.3], 'reflection': .27,
            'gain': 'Normalize kernel by its L1 norm, identically for every reference'},
        'reporting': 'All conditions and candidates; paired clean controls; zero denominator N/A; original quality once per reference. No method/threshold selection based on this diagnostic.',
        'limitations': ['Four already observed development utterances, one content source.',
            'First-order synthetic room, not a measured room or full Treble simulator.',
            'Independent final evaluators, but no claim of held-out-sentence generalization.',
            'Same fixed MOS/WER/identity gates. A degraded clean control is excluded, not counted as defense.']})
    print(path)


def transform(source, target, channel, sr, specification):
    if channel == 'mp3_64k':
        encoded = target.with_suffix('.mp3')
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(source),
            '-ac', '1', '-ar', str(sr), '-c:a', 'libmp3lame', '-b:a', '64k', str(encoded)], check=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(encoded),
            '-ac', '1', '-ar', str(sr), '-c:a', 'pcm_s16le', str(target)], check=True)
        return
    wave, actual_sr = sf.read(source, dtype='float32')
    if actual_sr != sr or wave.ndim != 1: raise ValueError('Unexpected channel input')
    if channel == 'heldout_room':
        room = specification['room']
        kernel = first_order_rir(sr, room['dimensions'], room['source'],
            room['microphone'], room['reflection'])
        kernel = kernel / kernel.abs().sum()
        wave = AF.fftconvolve(torch.from_numpy(wave), kernel, mode='full')[:len(wave)].numpy()
    elif channel != 'identity': raise ValueError('Unknown fixed probe channel')
    sf.write(target, wave, sr, subtype='PCM_16')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze: freeze(); return
    protocol, specification = read(EVIDENCE / 'protocol.json'), read(EVIDENCE / 'channel-protocol.json')
    if specification['main_protocol_sha256'] != file_sha256(EVIDENCE / 'protocol.json'):
        raise ValueError('Main protocol changed')
    if specification['code_sha256'] != file_sha256(__file__): raise ValueError('Probe execution source changed')
    run = Path(protocol['output'])
    if read(run / 'development-state.json')['state'] != 'complete':
        raise ValueError('Complete development required')
    state_path = run / 'channel-state.json'
    if state_path.exists() and read(state_path)['state'] == 'complete':
        print('Channel probe complete; preserving files'); return
    state = {'state': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': file_sha256(EVIDENCE / 'channel-protocol.json')}
    write_json(state_path, state)
    torch.set_num_threads(1)
    model, config, attr = load_adainvc('checkpoints/adain-vc-official', 'cuda')
    sr = config['preprocess']['sample_rate']
    evaluator = StagedEvaluator(run / 'channel-score-cache.json')
    content = specification['content']
    content_path = Path('data/VCTK22') / content.split('_')[0] / (content + '.wav')
    content_text = content_path.with_suffix('.txt').read_text().strip()
    rows = []
    for u in specification['utterances']:
        original_input = run / 'controls' / u / 'original.wav'
        reference = (Path('data/VCTK22') / u.split('_')[0] / (u + '.txt')).read_text().strip()
        for channel in specification['channels']:
            folder = run / 'robustness' / channel / u
            folder.mkdir(parents=True, exist_ok=True)
            original = folder / 'original.wav'
            if not original.exists(): transform(original_input, original, channel, sr, specification)
            clean_vc = folder / 'clean-vc.wav'
            if not clean_vc.exists(): convert(model, config, attr, str(content_path), str(original), str(clean_vc), 'cuda')
            clean_score, original_score = evaluator.score(clean_vc, content_text), evaluator.score(original, reference)
            for name in specification['profiles']:
                record = folder / (name + '.json')
                defended_input = run / 'development' / name / u / 'defended.wav'
                if record.exists():
                    row = read(record)
                    if row['input_sha256'] != file_sha256(defended_input): raise ValueError('Probe input changed')
                    rows.append(row); continue
                protected = folder / (name + '-reference.wav')
                converted = folder / (name + '-vc.wav')
                transform(defended_input, protected, channel, sr, specification)
                convert(model, config, attr, str(content_path), str(protected), str(converted), 'cuda')
                defended_score, converted_score = evaluator.score(protected, reference), evaluator.score(converted, content_text)
                paths = {'original': original, 'clean_vc': clean_vc,
                    'defended': protected, 'protected_vc': converted}
                row = {'utterance': u, 'content': content, 'channel': channel, 'profile': name,
                    'original': original_score, 'clean_vc': clean_score,
                    'defended': defended_score, 'protected_vc': converted_score,
                    'paths': {key: value.as_posix() for key, value in paths.items()},
                    'input_sha256': file_sha256(defended_input),
                    'channel_protocol_sha256': file_sha256(EVIDENCE / 'channel-protocol.json'),
                    'gates_ge2e': gates(u, original_score, clean_score, defended_score, converted_score),
                    'gates_ecapa': gates(u, original_score, clean_score, defended_score, converted_score, asv='ecapa', asr='base')}
                write_json(record, row); rows.append(row)
                write_json(run / 'channel-results.json', {'rows': rows})
                print(u, channel, name, row['gates_ge2e'], flush=True)
    state.update(state='complete', rows=len(rows), completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(state_path, state)
    analysis = {'protocol_sha256': state['protocol_sha256'], 'rows': len(rows),
        'development_diagnostic_only': True,
        'by_channel': {channel: {name: channel_summary([r for r in rows
            if r['channel'] == channel and r['profile'] == name])
            for name in specification['profiles']} for channel in specification['channels']},
        'limitations': specification['limitations']}
    write_json(EVIDENCE / 'channel-analysis.json', analysis)


if __name__ == '__main__': main()
