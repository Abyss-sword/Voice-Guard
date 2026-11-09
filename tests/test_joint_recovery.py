"""Gradient, synthesis and hard-constraint checks for the joint experiment."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from vguard.joint_recovery import WaveformReconstructor, identity_margin, refine_joint
from vguard.masking import generate_th
from vguard.vc.adainvc import melspectrogram2wav
from tests.test_quality_recovery import TinyModel
from tests.test_staged_recovery import WaveFeatures


class FlatReconstruction(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
    def forward(self, value): return value.flatten()


class TinySurrogate(nn.Module):
    def forward(self, value):
        embedding = torch.stack((value.mean(), value.square().mean(),
            (value * torch.linspace(-1, 1, len(value))).mean())).reshape(1, -1)
        return torch.nn.functional.normalize(embedding, dim=-1)


class JointRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_reconstruction_matches_official_and_has_waveform_gradient(self):
        torch.set_num_threads(1)
        pp = dict(sample_rate=16000, preemph=.97, n_fft=128, hop_length=32,
            win_length=128, n_mels=16, ref_db=20, max_db=100, n_iter=3)
        attr = {'mean': np.zeros(16, np.float32), 'std': np.ones(16, np.float32)}
        mel = np.random.default_rng(42).uniform(.35, .65, size=(40, 16)).astype(np.float32)
        expected = melspectrogram2wav(mel, **pp)
        value = torch.tensor(mel.T[None], requires_grad=True)
        actual = WaveformReconstructor(pp, attr, gradient_iterations=2)(value)
        # The official function also silence-trims; this synthetic signal has no
        # removable leading/trailing frames. torch/librosa 的 ISTFT 边界约定不同，
        # 低幅值边缘样本（~1e-2 量级）会有 ~2e-4 的绝对差；放宽 atol 但保持相对量级。
        np.testing.assert_allclose(actual.detach().numpy(), expected, atol=3e-4, rtol=1e-3)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(float(value.grad.abs().sum()), 0.)

    def test_margin_excludes_own_class_and_backpropagates(self):
        embedding = torch.tensor([[.8, .6]], requires_grad=True)
        margin = identity_margin(embedding, torch.eye(2), 0)
        self.assertAlmostEqual(float(margin), .2, places=6)
        margin.sum().backward()
        torch.testing.assert_close(embedding.grad, torch.tensor([[1., -1.]]))

    def test_guarded_refinement_updates_audio_with_frozen_models_and_budget(self):
        torch.manual_seed(8)
        model = TinyModel().eval().requires_grad_(False)
        wave = (.2 * np.sin(np.arange(1200) * .13)).astype(np.float32)
        pp = dict(n_fft=128, hop_length=32, win_length=128, n_iter=3)
        theta, psd = generate_th(wave, 16000, pp['n_fft'], pp['hop_length'], pp['win_length'])
        registry = np.array([[0., 1., 0.], [1., 0., 0.]], np.float32)
        with patch('vguard.joint_recovery.NativeFeatures', WaveFeatures), \
             patch('vguard.joint_recovery.WaveformReconstructor', FlatReconstruction):
            result, details = refine_joint(model, wave, wave.copy(),
                [torch.tensor(wave).reshape(1, 1, -1)], theta, psd, {}, pp,
                TinySurrogate(), registry, 1, steps=3, guarded=True)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(float(np.max(np.abs(result - wave))), 0.)
        self.assertLessEqual(float(np.mean((result - wave)**2)), float(np.mean(wave**2) * .01) + 1e-9)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(details['held_out'][0], 'GE2E')
        self.assertEqual(details['trace'][-1]['step'], 3)


if __name__ == '__main__': unittest.main()
