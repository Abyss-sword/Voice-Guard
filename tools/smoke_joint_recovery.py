"""Check the real output-waveform gradient on development data only."""
from pathlib import Path
import sys
import time
import numpy as np
import torch
import librosa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.asv import load_ecapa
from vguard.audio import load_audio
from vguard.joint_recovery import WaveformReconstructor, SurrogateSpeaker, refine_joint
from vguard.masking import generate_th
from vguard.run_state import write_json
from vguard.vc.adainvc import load_adainvc, get_spectrograms, normalize, melspectrogram2wav


def main():
    torch.set_num_threads(1)
    started = time.monotonic()
    model, config, attr = load_adainvc('checkpoints/adain-vc-official', 'cuda')
    pp = config['preprocess']
    file = 'data/VCTK22/p225/p225_040.wav'
    content_file = 'data/VCTK22/p225/p225_003.wav'
    wave, _ = librosa.effects.trim(load_audio(file, pp['sample_rate']), top_db=pp['top_db'])
    content, _ = get_spectrograms(content_file, **pp)
    target, _ = get_spectrograms(file, **pp)
    content = torch.from_numpy(normalize(content, attr)).T[None].cuda()
    target = torch.from_numpy(normalize(target, attr)).T[None].cuda()
    with torch.no_grad():
        decoded = model.inference(content, target)
    decoded = decoded.detach().requires_grad_(True)
    reconstructor = WaveformReconstructor(pp, attr, gradient_iterations=2).cuda()
    reconstruction = reconstructor(decoded)
    official = melspectrogram2wav(decoded.detach().cpu().numpy()[0].T * attr['std'] + attr['mean'], **pp)
    trimmed, _ = librosa.effects.trim(reconstruction.detach().cpu().numpy())
    length = min(len(trimmed), len(official))
    correlation = float(np.corrcoef(trimmed[:length], official[:length])[0, 1])
    relative_error = float(np.linalg.norm(trimmed[:length] - official[:length]) / np.linalg.norm(official[:length]))
    surrogate = SurrogateSpeaker(load_ecapa('cuda'), pp['sample_rate']).cuda()
    embedding = surrogate(reconstruction)
    embedding[0, 0].backward()
    if decoded.grad is None or not torch.isfinite(decoded.grad).all() or decoded.grad.abs().sum() == 0:
        raise ValueError('ECAPA output gradient failed')
    with np.load('outputs/registry.npz') as package:
        names, registry = list(package['names']), package['embeddings'].copy()
    theta, psd = generate_th(wave, pp['sample_rate'], pp['n_fft'], pp['hop_length'], pp['win_length'])
    result, details = refine_joint(model, wave, wave.copy(), [content], theta, psd, attr, pp,
        surrogate, registry, names.index('p225'), steps=3, guarded=True)
    report = {'development_only': True, 'official_samples': len(official), 'surrogate_samples_trimmed': len(trimmed),
        'waveform_correlation': correlation, 'relative_waveform_error': relative_error,
        'mel_gradient_l1': float(decoded.grad.abs().sum()), 'gradient_finite': True,
        'details': details, 'elapsed_s': time.monotonic() - started,
        'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated()}
    if not np.isfinite(relative_error) or relative_error > .1 or correlation < .99:
        raise ValueError('Synthesis parity insufficient: ' + str(report))
    write_json('outputs/joint_recovery/smoke.json', report)
    print(report, flush=True)


if __name__ == '__main__': main()
