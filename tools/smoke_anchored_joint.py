"""Three real development-only updates for the relative-output anchor extension."""
from pathlib import Path
import sys
import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.anchored_joint_recovery import refine_anchored
from vguard.joint_recovery import SurrogateSpeaker
from vguard.asv import load_ecapa
from vguard.audio import load_audio
from vguard.masking import generate_th
from vguard.run_state import write_json, file_sha256
from vguard.vc.adainvc import load_adainvc, get_spectrograms, normalize


def main():
    torch.set_num_threads(1)
    model, config, attr = load_adainvc('checkpoints/adain-vc-official', 'cuda')
    pp = config['preprocess']
    wave, _ = librosa.effects.trim(load_audio('data/VCTK22/p225/p225_040.wav', pp['sample_rate']), top_db=pp['top_db'])
    mel, _ = get_spectrograms('data/VCTK22/p225/p225_003.wav', **pp)
    mel = torch.from_numpy(normalize(mel, attr)).T[None].cuda()
    with np.load('outputs/registry.npz') as bank:
        names, registry = list(bank['names']), bank['embeddings'].copy()
    theta, psd = generate_th(wave, pp['sample_rate'], pp['n_fft'], pp['hop_length'], pp['win_length'])
    proxy = SurrogateSpeaker(load_ecapa('cuda'), pp['sample_rate']).cuda()
    result, details = refine_anchored(model, wave, wave.copy(), [mel], theta, psd, attr, pp,
        proxy, registry, names.index('p225'), steps=3)
    if not np.isfinite(result).all() or np.max(np.abs(result-wave)) == 0: raise ValueError('No real anchored update')
    write_json('outputs/joint_anchored/smoke.json', {'development_only': True, 'details': details,
        'code_sha256': file_sha256('vguard/anchored_joint_recovery.py'),
        'model_parameters_frozen': all(not p.requires_grad and p.grad is None for p in model.parameters())})


if __name__ == '__main__': main()
