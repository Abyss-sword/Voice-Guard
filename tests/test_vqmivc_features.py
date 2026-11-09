import unittest
from unittest.mock import patch
import numpy as np
import librosa
from vguard.vc.vqmivc import logmel, load_source_module


class VQMIVCFeaturesTests(unittest.TestCase):
    def test_modern_librosa_adapter_matches_pinned_original_feature_function(self):
        upstream=load_source_module('voiceguard_test_vqm_spectrogram','spectrogram.py')
        time=np.arange(16000)/16000
        wave=(.2*np.sin(2*np.pi*180*time)+.1*np.sin(2*np.pi*730*time)).astype('float32')
        native=librosa.filters.mel
        def compatibility(sr,n_fft,n_mels,fmin,fmax):
            return native(sr=sr,n_fft=n_fft,n_mels=n_mels,fmin=fmin,fmax=fmax)
        with patch.object(librosa.filters,'mel',compatibility):
            expected=upstream.logmelspectrogram(x=wave,fs=16000,n_mels=80,n_fft=400,
                n_shift=160,win_length=400,window='hann',fmin=80,fmax=7600)
        np.testing.assert_allclose(logmel(wave),expected,rtol=2e-5,atol=2e-5)


if __name__=='__main__':unittest.main()
