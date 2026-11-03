"""掩蔽阈值 θx 与 Transform 的形状/数值合法性测试（无需 checkpoint）。

运行：python -m tests.test_masking
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from vguard.masking import Transform, generate_th

CFG = dict(sample_rate=16000, n_fft=1024, hop_length=256, win_length=1024)


def test_generate_th():
    rng = np.random.default_rng(1)
    wav = rng.standard_normal(16000).astype(np.float32)
    theta, psd_max = generate_th(wav, **CFG)
    assert theta.ndim == 2, theta.shape
    n_frames = 1 + len(wav) // 256
    assert theta.shape[0] == n_frames, (theta.shape, n_frames)
    assert np.isfinite(theta).all(), "theta 含 NaN/Inf"
    assert psd_max > 0
    print("[OK] test_generate_th", theta.shape)


def test_transform_shape():
    rng = np.random.default_rng(2)
    wav = rng.standard_normal(16000).astype(np.float32)
    theta, psd_max = generate_th(wav, **CFG)
    delta = torch.from_numpy(rng.standard_normal(16000).astype(np.float32))
    tr = Transform(n_fft=1024, hop_length=256, win_length=1024)
    out = tr(delta, psd_max)
    assert out.shape == theta.shape, (tuple(out.shape), theta.shape)
    assert torch.isfinite(out).all(), "Transform 输出含 NaN/Inf"
    print("[OK] test_transform_shape", tuple(out.shape))


if __name__ == "__main__":
    test_generate_th()
    test_transform_shape()
    print("ALL PASS")
