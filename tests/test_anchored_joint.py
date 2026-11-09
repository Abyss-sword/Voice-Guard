import unittest
from unittest.mock import patch
import numpy as np
import torch
from vguard.anchored_joint_recovery import refine_anchored
from vguard.masking import generate_th
from tests.test_joint_recovery import TinySurrogate, FlatReconstruction
from tests.test_quality_recovery import TinyModel
from tests.test_staged_recovery import WaveFeatures


class AnchoredJointTests(unittest.TestCase):
    def test_relative_anchors_require_stronger_output_and_preserve_hard_budget(self):
        torch.set_num_threads(1)
        torch.manual_seed(14)
        model = TinyModel().eval().requires_grad_(False)
        wave = (.2 * np.sin(np.arange(1200) * .13)).astype(np.float32)
        pp = dict(n_fft=128, hop_length=32, win_length=128, n_iter=3, top_db=15)
        theta, psd = generate_th(wave, 16000, pp['n_fft'], pp['hop_length'], pp['win_length'])
        registry = np.array([[0., 1., 0.], [1., 0., 0.]], np.float32)
        with patch('vguard.anchored_joint_recovery.NativeFeatures', WaveFeatures), \
             patch('vguard.anchored_joint_recovery.WaveformReconstructor', FlatReconstruction):
            result, details = refine_anchored(model, wave, wave.copy(),
                [torch.tensor(wave).reshape(1, 1, -1)], theta, psd, {}, pp,
                TinySurrogate(), registry, 1, steps=3, native_prototypes=registry)
        self.assertLessEqual(details['anchors'][0]['margin'], -.12)
        self.assertGreater(details['trace'][-1]['defense'], 0.)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(float(np.max(np.abs(result - wave))), 0.)
        self.assertLessEqual(float(np.mean((result - wave)**2)), float(np.mean(wave**2) * .01) + 1e-9)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__': unittest.main()
