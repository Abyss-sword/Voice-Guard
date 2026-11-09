import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn
from vguard.staged_recovery import NativeFeatures, staged_recover
from vguard.vc.adainvc import get_spectrograms, normalize
from vguard.masking import generate_th
from tests.test_quality_recovery import TinyModel


class WaveFeatures(nn.Module):
    def __init__(self, *args): super().__init__()
    def forward(self, wave): return wave.reshape(1, 1, -1)


class StagedRecoveryTests(unittest.TestCase):
    def test_native_features_match_inference_including_first_sample(self):
        pp = dict(sample_rate=16000, preemph=.97, n_fft=128, hop_length=32,
                  win_length=128, n_mels=16, ref_db=20, max_db=100, top_db=100)
        attr = {'mean': np.linspace(.1, .2, 16).astype(np.float32), 'std': np.ones(16, np.float32) * .5}
        wave = (.03 * np.random.default_rng(13).standard_normal(3000) + .1).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.wav'
            sf.write(path, wave, 16000, subtype='FLOAT')
            expected, _ = get_spectrograms(str(path), **pp)
        value = torch.tensor(wave, requires_grad=True)
        actual = NativeFeatures(pp, attr)(value)
        np.testing.assert_allclose(actual.detach().numpy()[0].T, normalize(expected, attr), atol=2e-5, rtol=1e-5)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertNotEqual(float(value.grad[0]), 0.)

    def test_both_stages_update_audio_and_final_budget_is_enforced(self):
        torch.manual_seed(5)
        model = TinyModel().eval().requires_grad_(False)
        wave = (.2 * np.sin(np.arange(600) * .13)).astype(np.float32)
        pp = dict(n_fft=128, hop_length=32, win_length=128)
        theta, psd = generate_th(wave, 16000, **pp)
        saved = {}
        with patch('vguard.staged_recovery.NativeFeatures', WaveFeatures):
            result, trace = staged_recover(model, wave, np.array([[1., -1., .5], [.8, -.8, .6]], np.float32),
                [torch.tensor(wave).reshape(1, 1, -1)], theta, psd, {}, pp,
                objective='e2e', strong_steps=4, conceal_steps=4,
                callback=lambda phase, value: saved.update({phase: value}))
        self.assertEqual(set(saved), {'strong', 'final'})
        self.assertTrue(np.isfinite(result).all())
        self.assertFalse(np.array_equal(wave, saved['strong']))
        self.assertFalse(np.array_equal(saved['strong'], result))
        self.assertLessEqual(np.mean((result-wave)**2), np.mean(wave**2)*.01 + 1e-9)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(np.isfinite(trace['exported_embedding_loss']))


if __name__ == '__main__': unittest.main()
