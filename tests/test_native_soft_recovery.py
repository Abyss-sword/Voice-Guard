"""Check alpha-search bookkeeping independently of expensive real ASV inference."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from vguard.masking import generate_th
from vguard.native_soft_recovery import soft_recover
from tests.test_quality_recovery import TinyModel
from tests.test_staged_recovery import WaveFeatures


class NativeSoftRecoveryTests(unittest.TestCase):
    def run_search(self, succeeds):
        torch.manual_seed(7)
        model = TinyModel().eval().requires_grad_(False)
        wave = (.2*np.sin(np.arange(600)*.13)).astype(np.float32)
        pp = dict(n_fft=128, hop_length=32, win_length=128)
        theta, psd = generate_th(wave, 16000, **pp)
        original = torch.tensor([1., 0.])
        changed = torch.tensor([0., 1.]) if succeeds else original
        calls = [original] + [changed]*100
        saved = {}
        with patch('vguard.native_soft_recovery.NativeFeatures', WaveFeatures), \
             patch('vguard.native_soft_recovery.F.cosine_similarity', side_effect=calls):
            result, details = soft_recover(model, wave, np.array([[1., -1., .5]], np.float32),
                theta, psd, {}, pp, np.ones((2, 3), np.float32), strong_steps=2,
                conceal_steps=2, callback=lambda phase, value: saved.update({phase: value}))
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(np.max(np.abs(result-wave)), .100001)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(len([r for r in details['trace'] if r['phase']=='conceal']), 8)
        return result, details, saved

    def test_eight_successful_attempts_select_largest_alpha(self):
        _, details, _ = self.run_search(True)
        self.assertEqual(details['selected_alpha'], 128.)
        self.assertFalse(details['fallback_to_strong'])

    def test_unsuccessful_search_retains_strong_and_marks_fallback(self):
        result, details, saved = self.run_search(False)
        self.assertIsNone(details['selected_alpha'])
        self.assertTrue(details['fallback_to_strong'])
        np.testing.assert_array_equal(result, saved['strong'])


if __name__ == '__main__': unittest.main()
