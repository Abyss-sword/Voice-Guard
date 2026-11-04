"""Quality gates reject invalid metrics; spectral projection is measured after ISTFT."""
import unittest
import numpy as np
import torch

from vguard.masking import generate_th, perturbation_psd, project_masked_delta, MaskingProjection
from scripts.quality_sweep import quality_gate
from vguard.audio import restore_protected_crop


class QualityConstraintTests(unittest.TestCase):
    def test_preserves_quiet_prefix_and_suffix_and_rejects_wrong_crop(self):
        full = np.array([.0001,.0002,.1,-.2,.0003,.0004],dtype=np.float32)
        original = full[2:4].copy()
        restored = restore_protected_crop(full,original,original+.001,2,4)
        np.testing.assert_array_equal(restored[:2], full[:2])
        np.testing.assert_array_equal(restored[4:], full[4:])
        self.assertEqual(len(restored),len(full))
        with self.assertRaisesRegex(ValueError,'does not match'):
            restore_protected_crop(full,original+.01,original+.011,2,4)

    def test_quality_gate_requires_identity_and_finite_metrics(self):
        self.assertTrue(quality_gate(3, .1, True))
        for score, error, identity in ((2.99, 0, True), (4, .11, True), (4, 0, False),
                                       (float('nan'), 0, True), (4, float('inf'), True)):
            self.assertFalse(quality_gate(score, error, identity))

    def test_reconstructed_projection_reduces_mask_violation_and_obeys_peak_cap(self):
        rng = np.random.default_rng(7)
        time = np.arange(8000)/16000
        original = (.2*np.sin(2*np.pi*440*time) + .06*np.sin(2*np.pi*880*time)).astype(np.float32)
        cfg = dict(n_fft=1024, hop_length=256, win_length=1024)
        theta, psd_max = generate_th(original, sample_rate=16000, **cfg)
        delta = rng.normal(0, .03, len(original)).astype(np.float32)
        projected = project_masked_delta(delta, theta, psd_max, **cfg, iterations=30, margin_db=6)
        def excess(wave):
            psd = perturbation_psd(wave, **cfg, psd_max=psd_max).T
            return np.maximum(psd-theta, 0).mean()
        self.assertEqual(projected.shape, delta.shape)
        self.assertTrue(np.isfinite(projected).all())
        self.assertLessEqual(np.max(np.abs(projected)), np.max(np.abs(delta)))
        self.assertLess(excess(projected), .1*excess(delta))
        np.testing.assert_array_equal(project_masked_delta(np.zeros_like(delta), theta, psd_max, **cfg), 0)

    def test_torch_projection_matches_numpy_stft_units(self):
        rng = np.random.default_rng(3)
        original = rng.normal(0, .1, 4000).astype(np.float32)
        delta = rng.normal(0, .03, 4000).astype(np.float32)
        cfg = dict(n_fft=512, hop_length=128, win_length=400)
        theta, psd_max = generate_th(original, sample_rate=16000, **cfg)
        expected = project_masked_delta(delta, theta, psd_max, **cfg, iterations=1, margin_db=3)
        actual = MaskingProjection(theta, psd_max, **cfg)(torch.from_numpy(delta)).numpy()
        np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
