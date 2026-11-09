import unittest
import torch
from torch import nn

from vguard.content_robust_recovery import (ContentCTC, ReferenceChannels,
    allocated_gradient, first_order_rir, passes_proxy_caps, remove_content_conflict)
from tools.decide_content_validation import eligible


class ToyCTC(nn.Module):
    def __init__(self):
        super().__init__(); self.layer = nn.Conv1d(1, 4, 5, stride=4)

    def forward(self, wave):
        return self.layer(wave[:, None]).transpose(1, 2), None


class ContentRobustTests(unittest.TestCase):
    def test_frozen_ctc_retains_input_gradient_and_explicit_training_normalization(self):
        guard = ContentCTC(ToyCTC(), ('-', '|', 'A', 'B'))
        wave = torch.randn(1600, requires_grad=True)
        loss = guard.loss(wave, 16000, 'A, B!')
        loss.backward()
        self.assertEqual(guard.target('a, b!', 'cpu').tolist(), [2, 1, 3])
        self.assertTrue(torch.isfinite(wave.grad).all())
        self.assertGreater(float(wave.grad.abs().sum()), 0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in guard.parameters()))
        with self.assertRaisesRegex(ValueError, 'Empty'):
            guard.target('123!', 'cpu')

    def test_conflicting_direction_has_no_first_order_content_increase(self):
        content = torch.tensor([2., -1., .5])
        direction = remove_content_conflict(-content, content)
        self.assertGreaterEqual(float(direction @ content), -1e-6)
        helpful = content.clone()
        torch.testing.assert_close(remove_content_conflict(helpful, content), helpful)

    def test_allocation_avoids_content_sensitive_frequency(self):
        t = torch.arange(4096, dtype=torch.float32) / 16000
        low, high = torch.sin(2 * torch.pi * 250 * t), torch.sin(2 * torch.pi * 2000 * t)
        allocated, gate = allocated_gradient(low + high, high)
        before = abs(float((low + high) @ low)) / abs(float((low + high) @ high))
        after = abs(float(allocated @ low)) / abs(float(allocated @ high))
        self.assertGreater(after, before * 1.3)
        self.assertGreaterEqual(float(gate.min()), .09999)
        self.assertLessEqual(float(gate.max()), 1.)
        self.assertTrue(torch.isfinite(allocated).all())

    def test_reference_channels_preserve_length_and_backpropagate(self):
        channels = ReferenceChannels(24000)
        for index in range(len(channels.names)):
            wave = torch.randn(4096, requires_grad=True)
            output = channels(wave, index)
            self.assertEqual(output.shape, wave.shape)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(wave.grad).all())
            self.assertGreater(float(wave.grad.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, 'inside'):
            first_order_rir(24000, (3, 3, 3), (0, 1, 1), (2, 2, 2))

    def test_one_bad_proxy_or_nonfinite_value_rejects_export(self):
        values = {'ctc': 1., 'own_similarity': .98, 'spectrum': .1, 'native_defense': -.1}
        caps = {'ctc_max': 1.01, 'own_similarity_min': .975,
            'spectrum_max': .11, 'native_defense_max': -.095}
        self.assertTrue(passes_proxy_caps(values, caps))
        for key, bad in [('ctc', 1.02), ('own_similarity', .97),
                ('spectrum', .12), ('native_defense', -.09), ('ctc', float('nan'))]:
            self.assertFalse(passes_proxy_caps({**values, key: bad}, caps))

    def test_development_gate_cannot_hide_a_new_original_failure_in_equal_counts(self):
        baseline = {'profile': {'name': 'baseline'}, 'full_joint': 3, 'strict_joint': 5,
            'protected_quality': 2, 'mean_wer': .02, 'mean_mos': 4.}
        selected = {**baseline, 'profile': {'name': 'candidate'}, 'full_joint': 4}
        rows = [{'utterance': u, 'profile': name, 'gates_ge2e': {'protected_quality_ok': True}}
            for name, utterances in [('baseline', ('p225_a', 'p226_a')),
                                    ('candidate', ('p225_a', 'p227_a'))] for u in utterances]
        allowed, checks = eligible(selected, baseline, rows)
        self.assertFalse(allowed)
        self.assertFalse(checks['every_baseline_passing_original_preserved'])
        self.assertTrue(checks['original_quality_count_not_worse'])

    def test_development_gate_requires_joint_gain_and_bounded_quality(self):
        baseline = {'profile': {'name': 'baseline'}, 'full_joint': 3, 'strict_joint': 5,
            'protected_quality': 2, 'mean_wer': .02, 'mean_mos': 4.}
        selected = {**baseline, 'profile': {'name': 'candidate'}, 'full_joint': 4}
        rows = [{'utterance': u, 'profile': name, 'gates_ge2e': {'protected_quality_ok': True}}
            for name in ('baseline', 'candidate') for u in ('p225_a', 'p226_a')]
        self.assertTrue(eligible(selected, baseline, rows)[0])
        for key, bad in [('full_joint', 3), ('strict_joint', 4),
                ('mean_wer', .026), ('mean_mos', 3.94)]:
            self.assertFalse(eligible({**selected, key: bad}, baseline, rows)[0])


if __name__ == '__main__': unittest.main()
