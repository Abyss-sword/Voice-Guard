"""Regression for mismatched silence context in the production registry builder."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf
import torch
from torch import nn
from scripts.build_registry import vc_file_embedding
from vguard.vc.adainvc import get_spectrograms, normalize


class MeanEncoder(nn.Module):
    def forward(self, value): return value.mean(dim=2)


class NativeEnrollmentTests(unittest.TestCase):
    def test_enrollment_matches_trimmed_native_inference_with_stereo_resampling(self):
        sr = 22050
        speech = .2*np.sin(np.arange(5000)*.1)
        mono = np.concatenate((np.zeros(6000), speech, np.zeros(7000))).astype(np.float32)
        wave = np.stack([mono, .8*mono], axis=1)
        pp = dict(sample_rate=16000, preemph=.97, n_fft=128, hop_length=32,
            win_length=128, n_mels=16, ref_db=20, max_db=100, top_db=15)
        attr = {'mean': np.zeros(16, np.float32), 'std': np.ones(16, np.float32)}
        model = nn.Module(); model.speaker_encoder = MeanEncoder()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'with-silence.wav'; sf.write(path, wave, sr, subtype='FLOAT')
            actual = vc_file_embedding(model, {'preprocess': pp}, attr, path, 'cpu')
            native, _ = get_spectrograms(str(path), **pp)
            untrimmed, _ = get_spectrograms(str(path), **{**pp, 'top_db': 200})
        np.testing.assert_allclose(actual, normalize(native, attr).mean(axis=0), atol=1e-6)
        self.assertGreater(np.linalg.norm(actual-normalize(untrimmed, attr).mean(axis=0)), .05)


if __name__ == '__main__': unittest.main()
