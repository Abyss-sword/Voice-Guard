"""Regression tests for real upload rates, missing ASV, and invalid API parameters."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import soundfile as sf

spec = importlib.util.spec_from_file_location("voiceguard_adapter", Path(__file__).parents[1]/"platform/service/adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class PlatformContractTests(unittest.TestCase):
    def test_auto_speaker_uses_upload_sample_rate_and_downmixes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"stereo.wav"
            wav = np.sin(2*np.pi*220*np.arange(22050)/22050).astype(np.float32)*.1
            sf.write(path, np.stack((wav,wav),axis=1),22050)
            pipe = Mock(pp={"sample_rate":24000})
            pipe._ecapa_asv.return_value = {"pred":"p225"}
            self.assertEqual(adapter._auto_speaker(pipe,str(path)),"p225")
            actual = pipe._ecapa_asv.call_args.args[0]
            self.assertEqual(actual.ndim,1)
            self.assertLessEqual(abs(len(actual)-24000),1)  # fractional resampling rounds up

    def test_rejects_invalid_api_parameters_instead_of_clamping_or_truthy_strings(self):
        for params in ({"eps":float("nan")},{"eps":0},{"eps":.4},
                       {"fast_mode":"false"},{"preserve_context":"yes"},{"method":"unknown"}):
            with self.assertRaises(ValueError): adapter.parse_parameters(params)
        self.assertEqual(adapter.parse_parameters({}),(.05,True,False,"vg"))
        self.assertEqual(adapter.parse_parameters({"method":"vg-projected","preserve_context":True}),
                         (.05,True,True,"vg-projected"))


if __name__ == "__main__": unittest.main()
