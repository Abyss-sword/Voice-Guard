"""H（torch）与 numpy 参考 file2mel 的数值对齐测试。

不依赖 checkpoint：用合成波形 + 固定 mel 参数 + attr(mean=0,std=1) 验证
STFT/mel/log/clip 这条链路的数值一致性。checkpoint 的真实 attr 对齐在下载后补。

运行：python -m tests.test_features
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import librosa
import torch

from vguard.features import PreEmphasis, wav2mel_tensor

CFG = dict(
    sample_rate=16000, preemph=0.97, n_fft=1024, hop_length=256,
    win_length=1024, n_mels=80, ref_db=20.0, max_db=100.0, top_db=30.0,
)


def numpy_file2mel(wav, **c):
    wav = np.append(wav[0], wav[1:] - c["preemph"] * wav[:-1])
    wav = np.pad(wav, c["n_fft"] // 2, mode="reflect")  # 旧版 librosa center=True 行为
    linear = librosa.stft(
        y=wav, n_fft=c["n_fft"], hop_length=c["hop_length"],
        win_length=c["win_length"], center=False,
    )
    mag = np.abs(linear)
    mel_basis = librosa.filters.mel(
        sr=c["sample_rate"], n_fft=c["n_fft"], n_mels=c["n_mels"]
    )
    mel = np.dot(mel_basis, mag)
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mel = np.clip((mel - c["ref_db"] + c["max_db"]) / c["max_db"], 1e-8, 1)
    return mel.T.astype(np.float32)  # (n_frames, n_mels)


def test_preemphasis():
    x = torch.randn(1, 1, 5000)
    y = PreEmphasis(0.97)(x).squeeze().numpy()
    xn = x.squeeze().numpy()
    expected = xn[1:] - 0.97 * xn[:-1]  # t>=1 与 numpy 一致
    assert np.allclose(y[1:], expected, atol=1e-6), "preemphasis t>=1 不一致"
    print("[OK] test_preemphasis")


def test_wav2mel_alignment():
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(16000).astype(np.float32)  # 1s
    wav[0] = wav[1] = 0.0  # 消除预加重第 0 采样的边界差异，让两条路径完全一致
    wav_t = torch.from_numpy(wav)
    attr = {"mean": np.float32(0.0), "std": np.float32(1.0)}

    mel_t = wav2mel_tensor(wav_t, attr=attr, **CFG).squeeze(0)   # (n_mels, n_frames)
    mel_n = numpy_file2mel(wav, **CFG).T                          # (n_mels, n_frames)

    diff = (mel_t - mel_n).abs().max().item()
    print(f"[INFO] max abs err = {diff:.3e}")
    assert diff < 1e-4, f"对齐误差过大：{diff}"
    print("[OK] test_wav2mel_alignment")


if __name__ == "__main__":
    test_preemphasis()
    test_wav2mel_alignment()
    print("ALL PASS")
