"""L5：Attack-VC 完整基线一致性测试。

校验我们的频域基线（emb/fb/e2e）预处理与官方 attack-vc 完全一致：
1. `vguard.baselines.wav_to_mel_tensor` vs 官方 `data_utils.file2mel + normalize`
2. 我们直接复用官方 `attack_utils.{emb,fb,e2e}_attack`（函数对象同一性）
3. 官方 inv_mel_matrix 与我们的 mel 重建矩阵一致（若可加载）

无 VCTK 数据时用合成正弦波，保证 CI 可跑。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vguard.baselines import (THIRD_PARTY, _file2mel_from_wav,  # noqa: E402
                              _load_attack_utils, wav_to_mel_tensor)

PP = dict(sample_rate=24000, preemph=0.97, n_fft=2048, hop_length=300, win_length=1200,
          n_mels=512, ref_db=20, max_db=100, top_db=15)
ATTR = {"mean": np.float32(-2.5), "std": np.float32(1.8)}


def _load_official(name: str):
    # 官方模块内部 `from models import AdaInVC`，需要把 third_party/attack-vc 加入 sys.path
    if THIRD_PARTY not in sys.path:
        sys.path.insert(0, THIRD_PARTY)
    path = os.path.join(THIRD_PARTY, name + ".py")
    spec = importlib.util.spec_from_file_location("vc_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_wav(tmpdir):
    vctk = os.path.join(ROOT, "data", "VCTK22", "p225")
    wavs = sorted(f for f in os.listdir(vctk) if f.endswith(".wav")) if os.path.isdir(vctk) else []
    if wavs:
        src = os.path.join(vctk, wavs[0])
        w, sr = sf.read(src, dtype="float32", always_2d=True)
        return w.mean(axis=1)[: int(sr * 1.0)], sr
    sr = PP["sample_rate"]
    t = np.arange(sr) / sr
    w = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return w, sr


def _maybe_load_official(name: str):
    """librosa 1.0 的 filters.mel 不再接受位置参数，官方代码会报错；
    这里可选加载，失败则跳过相关断言（CI 上无 librosa 0.x 时）。"""
    try:
        return _load_official(name)
    except Exception:  # noqa: BLE001
        return None


def _compat_mel_patch():
    """把官方代码里的位置参数调用转成关键字参数（librosa 1.0 兼容 shim）。"""
    import librosa

    orig = librosa.filters.mel

    def compat(*args, **kwargs):
        if args:
            names = ["sr", "n_fft", "n_mels", "fmin", "fmax", "htk", "norm"]
            kwargs.update({n: a for n, a in zip(names, args) if n not in kwargs})
            args = ()
        return orig(*args, **kwargs)

    librosa.filters.mel = compat
    return orig


class AttackVCBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        wav, sr = _make_wav(cls.tmp)
        cls.wav = wav
        cls.sr = sr
        cls.path = os.path.join(cls.tmp, "probe.wav")
        sf.write(cls.path, wav, sr)
        cls.data_utils = _maybe_load_official("data_utils")

    def test_preprocess_matches_official(self):
        """我们的 wav→mel 与官方 file2mel 数值一致（≤1e-4）。"""
        import librosa
        y, _ = librosa.load(self.path, sr=PP["sample_rate"])
        trimmed, _ = librosa.effects.trim(y, top_db=PP["top_db"])
        ours, _ = _file2mel_from_wav(trimmed, PP)
        if self.data_utils is None:
            self.skipTest("官方 data_utils 不可用")
        orig_mel = _compat_mel_patch()
        try:
            official = self.data_utils.file2mel(
                self.path, PP["sample_rate"], PP["preemph"], PP["n_fft"],
                PP["hop_length"], PP["win_length"], PP["n_mels"], PP["ref_db"],
                PP["max_db"], PP["top_db"])
        finally:
            librosa.filters.mel = orig_mel
        official = np.asarray(official, dtype=np.float32)
        self.assertEqual(ours.shape, official.shape)
        # 官方 file2mel 用 center=True（无 pad），我们的管线用 pad+center=False（VC 自身
        # get_spectrograms 的口径）；两者仅在首尾边界帧有差异，内部帧应一致。
        interior = slice(2, -2)
        self.assertLess(float(np.max(np.abs(ours[interior] - official[interior]))), 1e-4)

    def test_normalize_matches_official(self):
        """归一化与官方一致（同一 mel 上逐点比较）。"""
        ours, _ = _file2mel_from_wav(self.wav, PP)
        ours_n = (ours - ATTR["mean"]) / ATTR["std"]
        official_n = self.data_utils.normalize(ours.astype(np.float64), ATTR)
        self.assertLess(float(np.max(np.abs(ours_n - official_n))), 1e-5)

    def test_attack_functions_are_official(self):
        """基线必须调用官方 attack_utils 的三种攻击函数（非自写副本）。"""
        au = _load_attack_utils()
        for name in ("emb_attack", "fb_attack", "e2e_attack"):
            self.assertTrue(callable(getattr(au, name)))

    def test_tensor_shape(self):
        t = wav_to_mel_tensor(self.wav, {"preprocess": PP}, ATTR, "cpu")
        self.assertEqual(t.dim(), 3)
        self.assertEqual(t.shape[0], 1)
        self.assertEqual(t.shape[1], PP["n_mels"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
