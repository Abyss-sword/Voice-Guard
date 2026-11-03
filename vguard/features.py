"""可微特征提取 H —— 与 VC 模型前处理逐位对齐。

这是官方 WHUB401/Voice-Guard ``data_utils.py`` 中 ``wav2mel_tensor`` 的忠实
重写（官方只发布了这一个核心函数，且硬编码了 ``.cuda()``，这里改成设备无关）。

H 的作用：让梯度从频域（mel 频谱）反传回时域波形，从而能直接优化 1×L 的扰动
δ（论文 Eq.2 的链式法则）。它必须和 VC 推理时的 numpy 前处理 ``file2mel``
（attack-vc ``data_utils.py``）数值一致，否则攻击会"打偏"。

与 numpy ``file2mel`` 的两处已知、可接受的差异：
1. 预加重的第 0 个采样：numpy 版 y[0]=x[0]；卷积版 y[0]=x[0]-coef*x[1]（reflect
   补零的边界效应）。仅 1 个采样、不可闻。
2. ``librosa.effects.trim`` 不可导：管线在 H 之外、进入 H 之前用 numpy 做一次
   trim，H 只在已裁剪的波形上纯可微计算（见 masking.file2wav_mask）。
"""
from __future__ import annotations

from typing import Dict

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreEmphasis(nn.Module):
    """时域预加重。conv1d 用的是互相关，故卷积核要翻转成 [-coef, 1]。"""

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            "flipped_filter", torch.FloatTensor([-coef, 1.0]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert len(input.size()) == 3, "input 必须是 (B, C, L)"
        input = F.pad(input, (1, 0), "reflect")  # 左补 1 帧保持长度一致
        return F.conv1d(input, self.flipped_filter)


class InversePreEmphasis(nn.Module):
    """逆预加重（一阶 IIR），用于 Griffin-Lim 重建后的波形恢复。"""

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.rnn = nn.RNN(1, 1, 1, bias=False, batch_first=True)
        self.rnn.weight_ih_l0.data.fill_(1.0)
        self.rnn.weight_hh_l0.data.fill_(self.coef)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x, _ = self.rnn(input.transpose(1, 2))
        return x.transpose(1, 2)


def normalize_tensor(mel: torch.Tensor, attr: Dict) -> torch.Tensor:
    mean = torch.as_tensor(attr["mean"], device=mel.device, dtype=mel.dtype)
    std = torch.as_tensor(attr["std"], device=mel.device, dtype=mel.dtype)
    return (mel - mean) / std


def wav2mel_tensor(
    wav: torch.Tensor,
    sample_rate: int,
    preemph: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    ref_db: float,
    max_db: float,
    top_db: float,  # H 内部不使用，仅用于与调用方 kwargs 兼容（trim 在 H 之外做）
    attr: Dict,
) -> torch.Tensor:
    """可微：波形 -> 归一化 mel 频谱。

    Returns:
        shape ``(1, n_mels, n_frames)``（官方代码最终 ``.T.unsqueeze(0)`` 的布局）。
    """
    preemp = PreEmphasis(coef=preemph).to(wav.device)
    wav = wav.unsqueeze(0).unsqueeze(0)              # (1, 1, L)
    preemp_wav = preemp(wav).squeeze(0).squeeze(0)   # (L,)

    window = torch.hann_window(win_length, device=wav.device)
    # 与 librosa.stft 默认参数对齐：center=True, pad_mode='reflect',
    # normalized=False, onesided=True, hann 窗；return_complex=False 返回 (real, imag)。
    linear = torch.stft(
        input=preemp_wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )  # (n_freq, n_frames, 2)
    mag = torch.sqrt(linear.pow(2).sum(-1) + 1e-9)

    mel_basis = torch.as_tensor(
        librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels),
        device=wav.device,
        dtype=mag.dtype,
    )
    mel = torch.matmul(mel_basis, mag)               # (n_mels, n_frames)

    mel = 20 * torch.log10(torch.maximum(torch.full_like(mel, 1e-5), mel))
    mel = torch.clip((mel - ref_db + max_db) / max_db, 1e-8, 1.0)
    mel = mel.T                                      # (n_frames, n_mels)
    mel = normalize_tensor(mel, attr)
    return mel.T.unsqueeze(0)                        # (1, n_mels, n_frames)
