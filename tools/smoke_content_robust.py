"""Check real CTC input gradients and one constrained development-only block."""
from pathlib import Path
import sys
import unittest
import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.content_robust_recovery import ContentCTC, refine_content
from vguard.joint_recovery import SurrogateSpeaker
from vguard.asv import load_ecapa
from vguard.audio import load_audio
from vguard.masking import generate_th
from vguard.run_state import file_sha256, write_json
from vguard.vc.adainvc import load_adainvc


def main():
    torch.set_num_threads(1); torch.manual_seed(2026110905)
    test_result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover('tests', pattern='test_content_robust.py'))
    if not test_result.wasSuccessful(): raise ValueError('Content unit checks failed')
    model, config, attr = load_adainvc('checkpoints/adain-vc-official', 'cuda')
    pp = config['preprocess']; sr = pp['sample_rate']
    source = Path('data/VCTK22/p225/p225_104.wav')
    full = load_audio(source, sr)
    wave, interval = librosa.effects.trim(full, top_db=pp['top_db'])
    baseline_path = Path('outputs/joint_recovery/2026-11-09/validation/native_emb_snr20/p225_104/defended.wav')
    initial, initial_sr = sf.read(baseline_path, dtype='float32')
    if initial_sr != sr or len(initial) != len(full): raise ValueError('Smoke warm start differs')
    theta, psd = generate_th(wave, sr, pp['n_fft'], pp['hop_length'], pp['win_length'])
    ctc = ContentCTC().cuda()
    probe = torch.tensor(full, device='cuda', requires_grad=True)
    reference = source.with_suffix('.txt').read_text().strip()
    loss = ctc.loss(probe, sr, reference)
    gradient, = torch.autograd.grad(loss, probe)
    finite = bool(torch.isfinite(gradient).all()) and float(gradient.abs().sum()) > 0
    if not finite: raise ValueError('Real CTC input gradient invalid')
    with np.load('outputs/staged_recovery/2026-11-09/native-target-bank.npz') as bank:
        target = bank['exemplars'][list(bank['names']).index('p236')].copy()
    proxy = SurrogateSpeaker(load_ecapa('cuda'), sr).cuda()
    result, details = refine_content(model, wave, initial[interval[0]:interval[1]],
        theta, psd, attr, pp, target, ctc, reference, full, interval, proxy, steps=8)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite real refinement')
    frozen = all(not p.requires_grad and p.grad is None
        for network in (model, ctc, proxy) for p in network.parameters())
    write_json('outputs/content_robust/smoke.json', {'development_only': True,
        'utterance': source.stem, 'ctc_gradient_finite': finite,
        'ctc_loss': float(loss.detach()), 'ctc_gradient_rms': float(gradient.square().mean().sqrt()),
        'models_frozen': frozen, 'unit_tests_passed': test_result.wasSuccessful(),
        'unit_test_count': test_result.testsRun, 'details': details,
        'code_sha256': file_sha256('vguard/content_robust_recovery.py'),
        'training_budget_estimate_per_160_steps_s': details['elapsed_s'] * 20,
        'human_mos_ratings': 0})


if __name__ == '__main__': main()
