"""Run frozen CTC-preserving reference-defense candidates and gated validation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import sys
import time
import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.audio import load_audio, restore_protected_crop
from vguard.joint_recovery import SurrogateSpeaker
from vguard.content_robust_recovery import ContentCTC, refine_content, check_export, passes_proxy_caps
from vguard.masking import generate_th, perturbation_psd
from vguard.metrics.staged import StagedEvaluator, gates
from vguard.run_state import file_sha256, utterance_seed, write_json
from vguard.staged_recovery import staged_recover
from vguard.vc.adainvc import load_adainvc, get_spectrograms, normalize, convert

PROTOCOL = Path('docs/evidence/2026-11-09-content-robust/protocol.json')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def rank(summary):
    return (summary['full_joint'], summary['strict_joint'], summary['protected_quality'],
        -summary['mean_wer'], summary['mean_mos'])


def select(protocol):
    root = Path(protocol['output'])
    state = read(root / 'development-state.json')
    digest = file_sha256(PROTOCOL)
    if state['state'] != 'complete' or state['protocol_sha256'] != digest:
        raise ValueError('Matching complete development required')
    rows = read(root / 'development-results.json')['rows']
    profiles = [protocol['baseline']] + protocol['profiles']
    expected = {(u, c, p['name']) for u in protocol['development'] for c in protocol['evaluation_contents'] for p in profiles}
    if len(rows) != len(expected) or {(r['utterance'], r['content'], r['profile']) for r in rows} != expected:
        raise ValueError('Missing/duplicate development rows')
    if any(r['protocol_sha256'] != digest for r in rows): raise ValueError('Development protocol mismatch')
    summaries = []
    for profile in profiles:
        values = [r for r in rows if r['profile'] == profile['name']]
        unique = {r['utterance']: r for r in values}
        summaries.append({'profile': profile,
            'full_joint': sum(r['gates_ge2e']['paper_full_screen'] for r in values),
            'strict_joint': sum(r['gates_ge2e']['strict_joint'] for r in values),
            'protected_quality': sum(r['gates_ge2e']['protected_quality_ok'] for r in unique.values()),
            'mean_wer': float(np.mean([r['defended']['wer_small_en'] for r in unique.values()])),
            'mean_mos': float(np.mean([r['defended']['mos_original'] for r in unique.values()]))})
    baseline = summaries[0]
    challenger = max(summaries[1:], key=lambda s: (*rank(s), s['profile']['name']))
    destination = PROTOCOL.parent / 'selection.json'
    if destination.exists() or (root / 'validation-state.json').exists():
        raise ValueError('Selection already frozen or validation examined')
    write_json(destination, {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': digest, 'development_results_sha256': file_sha256(root / 'development-results.json'),
        'selected': challenger, 'baseline': baseline, 'all_candidates': summaries,
        'recommended_after_development': challenger['profile']['name'] if rank(challenger) > rank(baseline) else baseline['profile']['name'],
        'validation_examined': False, 'selection_rule': protocol['selection_rule']})
    print(json.dumps(read(destination), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['development', 'select', 'validation'], required=True)
    parser.add_argument('--max-hours', type=float, default=.5)
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    if args.split == 'select': select(protocol); return
    digest = file_sha256(PROTOCOL)
    for name, sha in {**protocol['assets'], **protocol['code_sha256']}.items():
        if file_sha256(name) != sha: raise ValueError('Frozen input changed: ' + name)
    import resemblyzer
    if file_sha256(Path(resemblyzer.__file__).parent / 'pretrained.pt') != protocol['ge2e_weights_sha256']:
        raise ValueError('GE2E checkpoint changed')
    for name, sha in protocol['whisper_sha256'].items():
        if file_sha256(Path.home() / '.cache/whisper' / (name + '.pt')) != sha:
            raise ValueError('Whisper checkpoint changed')
    if args.split == 'validation':
        decision_path = PROTOCOL.parent / 'extension-decision.json'
        decision = read(decision_path)
        if (not decision['validation_allowed'] or decision['protocol_sha256'] != digest
                or decision['selection_sha256'] != file_sha256(PROTOCOL.parent / 'selection.json')):
            raise ValueError('Matching passed development gate required')
    root = Path(protocol['output']); root.mkdir(parents=True, exist_ok=True)
    state_path = root / (args.split + '-state.json')
    selection_path = PROTOCOL.parent / 'selection.json'
    if state_path.exists():
        old = read(state_path)
        if old['protocol_sha256'] != digest: raise ValueError('Existing state uses another protocol')
        if old['state'] == 'complete': print('Already complete; preserving files'); return
    if args.split == 'development':
        profiles = [protocol['baseline']] + protocol['profiles']
    else:
        selection = read(selection_path)
        if selection['protocol_sha256'] != digest or selection['validation_examined']:
            raise ValueError('Invalid frozen selection')
        if selection['selected']['profile'] not in protocol['profiles']:
            raise ValueError('Unregistered challenger')
        profiles = [protocol['baseline'], selection['selected']['profile']]
    state = {'state': 'running', 'protocol_sha256': digest,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'selection_sha256': file_sha256(selection_path) if args.split == 'validation' else None}
    write_json(state_path, state)
    started = time.monotonic()
    def bounded():
        if time.monotonic() - started > args.max_hours * 3600:
            raise TimeoutError('Split wall-time budget reached')
    torch.set_num_threads(1)
    model, config, attr = load_adainvc('checkpoints/adain-vc-official', 'cuda')
    pp = config['preprocess']; sr = pp['sample_rate']
    evaluator = StagedEvaluator(root / 'score-cache.json')
    surrogate = SurrogateSpeaker(evaluator.ecapa, sr).cuda()
    ctc = ContentCTC().cuda()
    registry_names, registry = evaluator.registries['ecapa']
    with np.load(protocol['native_target_bank']) as package:
        names = list(package['names']); exemplars = package['exemplars'].copy()
    def contents(utterances):
        result = []
        for u in utterances:
            mel, _ = get_spectrograms(str(Path('data/VCTK22') / u.split('_')[0] / (u + '.wav')), **pp)
            result.append(torch.from_numpy(normalize(mel, attr)).T[None].cuda())
        return result
    baseline_contents = contents(protocol['baseline_optimization_contents'])
    refinement_contents = contents(protocol['optimization_contents'])
    rows = []
    try:
        for u in protocol[args.split]:
            bounded(); speaker = u.split('_')[0]
            source = Path('data/VCTK22') / speaker / (u + '.wav')
            reference = source.with_suffix('.txt').read_text().strip()
            full = load_audio(source, sr)
            crop, interval = librosa.effects.trim(full, top_db=pp['top_db'])
            theta, psd = generate_th(crop, sr, pp['n_fft'], pp['hop_length'], pp['win_length'])
            control = root / 'controls' / u; control.mkdir(parents=True, exist_ok=True)
            original_path = control / 'original.wav'
            if not original_path.exists(): sf.write(original_path, full, sr, subtype='PCM_16')
            original = evaluator.score(original_path, reference)
            controls = {}
            for c in protocol['evaluation_contents']:
                bounded()
                content_path = Path('data/VCTK22') / c.split('_')[0] / (c + '.wav')
                path = control / (c + '.wav')
                if not path.exists(): convert(model, config, attr, str(content_path), str(original_path), str(path), 'cuda')
                controls[c] = evaluator.score(path, content_path.with_suffix('.txt').read_text().strip())
            for profile in profiles:
                bounded()
                name = profile['name']
                job = root / args.split / name / u; job.mkdir(parents=True, exist_ok=True)
                defended_path = job / 'defended.wav'; optimization_path = job / 'optimization.json'
                if not optimization_path.exists():
                    seed = utterance_seed(protocol['seed'], u + ':baseline' if name == protocol['baseline']['name'] else u + ':refinement')
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    def save(value):
                        bounded()
                        restored = restore_protected_crop(full, crop, value, *map(int, interval))
                        sf.write(defended_path, restored, sr, subtype='PCM_16')
                    if name == protocol['baseline']['name']:
                        baseline = protocol['baseline']
                        target_name = protocol['target_map'][speaker]
                        _, details = staged_recover(model, crop, exemplars[names.index(target_name)],
                            baseline_contents, theta, psd, attr, pp, objective='embedding',
                            **{k: baseline[k] for k in ('strong_steps', 'conceal_steps', 'strong_snr_db',
                                'final_snr_db', 'learning_rate', 'final_margin_db')},
                            callback=lambda label, value: save(value) if label == 'final' else bounded())
                    else:
                        baseline_path = root / args.split / protocol['baseline']['name'] / u / 'defended.wav'
                        baseline_meta = read(baseline_path.with_name('optimization.json'))
                        if baseline_meta['defended_sha256'] != file_sha256(baseline_path):
                            raise ValueError('Baseline warm start changed')
                        initial, initial_sr = sf.read(baseline_path, dtype='float32')
                        if initial_sr != sr or len(initial) != len(full): raise ValueError('Warm start layout changed')
                        target_name = protocol['target_map'][speaker]
                        targets = exemplars[names.index(target_name)]
                        _, details = refine_content(model, crop, initial[interval[0]:interval[1]],
                            theta, psd, attr, pp, targets, ctc, reference, full, interval,
                            surrogate, eot=profile['eot'], callback=save, **protocol['refinement'])
                        exported, export_sr = sf.read(defended_path, dtype='float32')
                        if export_sr != sr or len(exported) != len(full):
                            raise ValueError('PCM export layout differs')
                        actual = check_export(model, crop, exported[interval[0]:interval[1]],
                            full, interval, pp, attr, targets, ctc, reference, surrogate)
                        details['actual_pcm_proxy_checks'] = actual
                        details['fallback_to_baseline'] = not passes_proxy_caps(actual, details['proxy_caps'])
                        if details['fallback_to_baseline']:
                            shutil.copyfile(baseline_path, defended_path)
                        details['final_sha256_equals_baseline'] = file_sha256(defended_path) == file_sha256(baseline_path)
                        details['warm_start_sha256'] = file_sha256(baseline_path)
                    details.update(seed=seed, profile=profile, protocol_sha256=digest,
                        crop_interval=list(map(int, interval)), defended_sha256=file_sha256(defended_path))
                    write_json(optimization_path, details)
                saved = read(optimization_path)
                if saved['protocol_sha256'] != digest or saved['defended_sha256'] != file_sha256(defended_path):
                    raise ValueError('Saved optimization mismatch')
                defended = evaluator.score(defended_path, reference)
                exported, _ = sf.read(defended_path, dtype='float32')
                original_export, _ = sf.read(original_path, dtype='float32')
                delta = exported - original_export
                excess = np.maximum(perturbation_psd(delta[interval[0]:interval[1]], pp['n_fft'],
                    pp['hop_length'], pp['win_length'], psd).T - theta, 0)
                for c in protocol['evaluation_contents']:
                    bounded()
                    record = job / (c + '.json')
                    if record.exists():
                        row = read(record)
                        if row['protocol_sha256'] != digest or row['optimization_sha256'] != file_sha256(optimization_path):
                            raise ValueError('Saved row mismatch')
                        rows.append(row); continue
                    content_path = Path('data/VCTK22') / c.split('_')[0] / (c + '.wav')
                    path = job / (c + '.wav')
                    convert(model, config, attr, str(content_path), str(defended_path), str(path), 'cuda')
                    converted = evaluator.score(path, content_path.with_suffix('.txt').read_text().strip())
                    noise_power = float(np.mean(delta**2))
                    row = {'utterance': u, 'content': c, 'profile': name,
                        'original': original, 'clean_vc': controls[c], 'defended': defended, 'protected_vc': converted,
                        'gates_ge2e': gates(u, original, controls[c], defended, converted),
                        'gates_ecapa': gates(u, original, controls[c], defended, converted, asv='ecapa', asr='base'),
                        'snr_db': float(10 * np.log10(np.mean(original_export**2) / max(noise_power, 1e-20))),
                        'mask_exceed_fraction': float(np.mean(excess > 0)), 'mask_excess_mean_db': float(excess.mean()),
                        'protocol_sha256': digest, 'optimization_sha256': file_sha256(optimization_path)}
                    write_json(record, row); rows.append(row)
                    write_json(root / (args.split + '-results.json'), {'rows': rows})
                    print(json.dumps({'utterance': u, 'profile': name, 'content': c,
                        'ge2e': row['gates_ge2e'], 'mos': defended['mos_original']}, ensure_ascii=False), flush=True)
        state.update(state='complete', rows=len(rows))
    except BaseException as error:
        state.update(state='interrupted_or_failed', error=str(error), rows=len(rows)); raise
    finally:
        state.update(elapsed_s=round(time.monotonic() - started, 2), updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)


if __name__ == '__main__': main()
