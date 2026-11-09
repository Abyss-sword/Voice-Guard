"""Run or resume the frozen third-round development and paired validation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.audio import load_audio, restore_protected_crop
from vguard.masking import generate_th, perturbation_psd
from vguard.metrics.staged import StagedEvaluator, gates
from vguard.recovery import recover_waveform
from vguard.run_state import file_sha256, utterance_seed, write_json
from vguard.staged_recovery import staged_recover
from vguard.vc.adainvc import load_adainvc, get_spectrograms, normalize, convert

PROTOCOL = Path('docs/evidence/2026-11-09-staged/protocol.json')
BASELINE = {'name': 'previous_feedback_snr20', 'objective': 'baseline', 'final_snr_db': 20.}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def select(protocol):
    root = Path(protocol['output'])
    if read(root / 'development-state.json')['state'] != 'complete':
        raise ValueError('Development must be complete')
    rows = read(root / 'development-results.json')['rows']
    expected = {(u, p['name'], c) for u in protocol['development'] for p in [BASELINE] + protocol['profiles'] for c in protocol['evaluation_contents']}
    if len(rows) != len(expected) or {(r['utterance'], r['profile'], r['content']) for r in rows} != expected:
        raise ValueError('Development incomplete or duplicated')
    summaries = []
    for profile in protocol['profiles']:
        values = [r for r in rows if r['profile'] == profile['name']]
        unique = {r['utterance']: r for r in values}
        summaries.append({'profile': profile, 'strict_joint': sum(r['gates_ge2e']['strict_joint'] for r in values),
            'full_paper_joint': sum(r['gates_ge2e']['paper_full_screen'] for r in values),
            'protected_quality': sum(r['gates_ge2e']['protected_quality_ok'] for r in unique.values()),
            'mean_wer': float(np.mean([r['defended']['wer_small_en'] for r in unique.values()])),
            'mean_mos': float(np.mean([r['defended']['mos_original'] for r in unique.values()]))})
    chosen = max(summaries, key=lambda r: (r['strict_joint'], r['full_paper_joint'], r['protected_quality'],
        -r['mean_wer'], r['mean_mos'], r['profile']['final_snr_db'], r['profile']['name']))
    path = PROTOCOL.parent / 'selection.json'
    if path.exists() or (root / 'validation-state.json').exists(): raise ValueError('Selection cannot be changed')
    write_json(path, {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': file_sha256(PROTOCOL),
        'development_results_sha256': file_sha256(root / 'development-results.json'),
        'selected': chosen, 'all_candidates': summaries, 'validation_examined': False,
        'selection_rule': protocol['selection_rule']})
    print(json.dumps(chosen, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['development', 'select', 'validation'], required=True)
    parser.add_argument('--max-hours', type=float, default=.75)
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    if args.split == 'select':
        select(protocol); return
    for name, digest in {**protocol['assets'], **protocol['code_sha256']}.items():
        if file_sha256(name) != digest: raise ValueError('Frozen input changed: ' + name)
    for name, digest in protocol['whisper_sha256'].items():
        if file_sha256(Path.home() / '.cache/whisper' / (name + '.pt')) != digest:
            raise ValueError('Whisper checkpoint changed')
    import resemblyzer
    if file_sha256(Path(resemblyzer.__file__).parent / 'pretrained.pt') != protocol['ge2e_weights_sha256']:
        raise ValueError('GE2E weights changed')
    root = Path(protocol['output'])
    state_path = root / (args.split + '-state.json')
    if state_path.exists() and read(state_path)['state'] == 'complete':
        if read(state_path)['protocol_sha256'] != file_sha256(PROTOCOL): raise ValueError('Completed split protocol differs')
        print('Split already complete; preserving files'); return
    selection_path = PROTOCOL.parent / 'selection.json'
    if args.split == 'validation':
        selection = read(selection_path)
        if selection['protocol_sha256'] != file_sha256(PROTOCOL): raise ValueError('Selection mismatch')
        if selection['selected']['profile'] not in protocol['profiles']: raise ValueError('Unregistered selected profile')
        profiles = [BASELINE, selection['selected']['profile']]
    else:
        profiles = [BASELINE] + protocol['profiles']
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {'state': 'running', 'split': args.split, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': file_sha256(PROTOCOL),
        'selection_sha256': file_sha256(selection_path) if args.split == 'validation' else None}
    write_json(state_path, state)
    torch.set_num_threads(1)
    model, config, attr = load_adainvc('checkpoints/adain-vc-official', 'cuda')
    pp = config['preprocess']; sr = pp['sample_rate']
    kwargs = {k: pp[k] for k in ('sample_rate', 'preemph', 'n_fft', 'hop_length', 'win_length', 'n_mels', 'ref_db', 'max_db', 'top_db')}
    evaluator = StagedEvaluator(root / 'score-cache.json')
    with np.load(protocol['native_target_bank']) as package:
        names = list(package['names']); exemplars = package['exemplars']
    with np.load('outputs/registry_vc.npz') as package:
        old_names = list(package['names']); old_embeddings = package['embeddings']
    contents = []
    for u in protocol['optimization_contents']:
        value, _ = get_spectrograms(str(Path('data/VCTK22') / u.split('_')[0] / (u + '.wav')), **pp)
        contents.append(torch.from_numpy(normalize(value, attr)).T.unsqueeze(0).cuda())
    rows = []
    def bounded():
        if time.monotonic() - started > args.max_hours * 3600: raise TimeoutError('Wall-time budget reached')
    try:
        for u in protocol[args.split]:
            bounded(); speaker = u.split('_')[0]
            source = Path('data/VCTK22') / speaker / (u + '.wav')
            reference = source.with_suffix('.txt').read_text().strip()
            full = load_audio(source, sr)
            crop, interval = librosa.effects.trim(full, top_db=pp['top_db'])
            theta, psd = generate_th(crop, sr, pp['n_fft'], pp['hop_length'], pp['win_length'])
            clean_dir = root / 'controls' / u; clean_dir.mkdir(parents=True, exist_ok=True)
            clean_wave = clean_dir / 'original.wav'
            if not clean_wave.exists(): sf.write(clean_wave, full, sr, subtype='PCM_16')
            original = evaluator.score(clean_wave, reference)
            controls = {}
            for c in protocol['evaluation_contents']:
                content = Path('data/VCTK22') / c.split('_')[0] / (c + '.wav')
                ref = content.with_suffix('.txt').read_text().strip()
                path = clean_dir / (c + '.wav')
                seed = utterance_seed(protocol['seed'], u + ':' + c)
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                if not path.exists(): convert(model, config, attr, str(content), str(clean_wave), str(path), 'cuda')
                controls[c] = evaluator.score(path, ref)
            for profile in profiles:
                bounded()
                job = root / args.split / profile['name'] / u; job.mkdir(parents=True, exist_ok=True)
                protected_path = job / 'defended.wav'
                optimization_path = job / 'optimization.json'
                if not optimization_path.exists():
                    seed = utterance_seed(protocol['seed'], u + ':native' if profile['objective'] != 'baseline' else u + ':baseline')
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    def save(label, wave):
                        bounded()
                        restored = restore_protected_crop(full, crop, wave, *map(int, interval))
                        path = protected_path if label == 'final' else job / 'strong.wav'
                        sf.write(path, restored, sr, subtype='PCM_16')
                    if profile['objective'] == 'baseline':
                        own = old_embeddings[old_names.index(speaker)]
                        similarities = old_embeddings @ own / (np.linalg.norm(old_embeddings, axis=1) * np.linalg.norm(own) + 1e-9)
                        target_index = int(similarities.argmin()); target_name = str(old_names[target_index])
                        _, trace = recover_waveform(model, crop, old_embeddings[target_index], contents, theta, psd,
                            attr, kwargs, objective='feedback', steps=1500, learning_rate=.0003,
                            snr_db=20., margin_db=3., checkpoints=(1500,), callback=lambda step, wave: save('final', wave))
                        details = {'trace': trace, 'evaluators_used_in_gradient_loss': False}
                    else:
                        target_name = protocol['target_map'][speaker]
                        _, details = staged_recover(model, crop, exemplars[names.index(target_name)], contents,
                            theta, psd, attr, pp, objective=profile['objective'], strong_steps=protocol['strong_steps'],
                            conceal_steps=protocol['conceal_steps'], strong_snr_db=protocol['strong_snr_db'],
                            final_snr_db=profile['final_snr_db'], learning_rate=protocol['learning_rate'],
                            final_margin_db=protocol['final_margin_db'], callback=save)
                    details.update(seed=seed, profile=profile, target_identity=target_name,
                        crop_interval=list(map(int, interval)), defended_sha256=file_sha256(protected_path),
                        protocol_sha256=file_sha256(PROTOCOL))
                    write_json(optimization_path, details)
                saved = read(optimization_path)
                if saved['protocol_sha256'] != file_sha256(PROTOCOL): raise ValueError('Saved optimization protocol differs')
                if saved['defended_sha256'] != file_sha256(protected_path): raise ValueError('Protected WAV changed')
                defended = evaluator.score(protected_path, reference)
                decoded, _ = sf.read(protected_path, dtype='float32')
                original_export, _ = sf.read(clean_wave, dtype='float32')
                delta = decoded - original_export
                excess = np.maximum(perturbation_psd(delta[interval[0]:interval[1]], pp['n_fft'], pp['hop_length'], pp['win_length'], psd).T - theta, 0)
                for c in protocol['evaluation_contents']:
                    bounded()
                    record_path = job / (c + '.json')
                    if record_path.exists():
                        row = read(record_path)
                        if row['protocol_sha256'] != file_sha256(PROTOCOL): raise ValueError('Saved record protocol differs')
                        rows.append(row); continue
                    content = Path('data/VCTK22') / c.split('_')[0] / (c + '.wav')
                    converted_path = job / (c + '.wav')
                    seed = utterance_seed(protocol['seed'], u + ':' + c)
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    convert(model, config, attr, str(content), str(protected_path), str(converted_path), 'cuda')
                    converted = evaluator.score(converted_path, content.with_suffix('.txt').read_text().strip())
                    row = {'utterance': u, 'content': c, 'profile': profile['name'], 'original': original,
                        'clean_vc': controls[c], 'defended': defended, 'protected_vc': converted,
                        'gates_ge2e': gates(u, original, controls[c], defended, converted),
                        'gates_ecapa': gates(u, original, controls[c], defended, converted, asv='ecapa', asr='base'),
                        'snr_db': float(10 * np.log10(np.mean(original_export**2) / np.mean(delta**2))),
                        'mask_exceed_fraction': float(np.mean(excess > 0)), 'mask_excess_mean_db': float(excess.mean()),
                        'protocol_sha256': file_sha256(PROTOCOL), 'optimization_sha256': file_sha256(optimization_path)}
                    write_json(record_path, row); rows.append(row)
                    write_json(root / (args.split + '-results.json'), {'rows': rows})
                    print(json.dumps({'utterance': u, 'profile': profile['name'], 'content': c,
                        'ge2e': row['gates_ge2e'], 'mos': defended['mos_original']}, ensure_ascii=False), flush=True)
        state.update(state='complete', rows=len(rows))
    except BaseException as error:
        state.update(state='interrupted_or_failed', error=str(error)); raise
    finally:
        state.update(elapsed_s=round(time.monotonic() - started, 2), updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)


if __name__ == '__main__': main()
