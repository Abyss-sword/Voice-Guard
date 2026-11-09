import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from vguard.channel import codec_channel


class CodecTests(unittest.TestCase):
    def test_zero_channel_control_has_no_quantization_but_pcm16_does(self):
        wave=np.array([.123456,-.321234,.000001],dtype=np.float32)
        np.testing.assert_array_equal(codec_channel(wave,24000,"none"),wave)
        quantized=codec_channel(wave,24000,"pcm16")
        self.assertGreater(np.max(np.abs(quantized-wave)),0)
        self.assertLessEqual(np.max(np.abs(quantized-wave)),1/32768)

    def test_failed_encode_raises_and_removes_temporary_directory(self):
        paths=[]
        def fail(fin,fout,args):
            paths.append(Path(fin).parent)
            return False
        with patch('vguard.channel._ffmpeg',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'ffmpeg channel failed'):
                codec_channel(np.zeros(1000,dtype=np.float32),24000,"mp3-32k")
        self.assertTrue(paths)
        self.assertFalse(paths[0].exists())


if __name__=='__main__': unittest.main()
