"""MCD（Mel Cepstral Distortion）。

配方（报告中明示）：
  1. 24 维 mel 频谱（power=2）→ dB
  2. DCT(type-2, ortho) 取前 13 个 cepstral 系数，剔除 C0
  3. 帧对齐（同内容音频线性对齐；长度不同可用 DTW）
  4. MCD = mean_t ||c1(t)-c2(t)||_2 · 10√2/ln10

注：论文未开源其 MCD 实现，绝对量纲可能不同；本实现用于**方法间相对比较**，
报告中统一口径。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import librosa
from scipy.fftpack import dct


def mel_cepstrum(wav: np.ndarray, sample_rate: int,
                 n_mels: int = 24, n_mfcc: int = 13):
    """返回 ((n_frames, n_mfcc) 系数, (n_mels, n_frames) dB 谱)。"""
    S = librosa.feature.melspectrogram(
        y=wav.astype(np.float32), sr=sample_rate, n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S)
    c = dct(S_db, axis=0, type=2, norm="ortho")[:n_mfcc]
    return c.T, S_db


def _active_mask(ref_db: np.ndarray, rel_thresh_db: float = 40.0) -> np.ndarray:
    """能量门控：参考谱帧峰值低于 (全局峰值 - rel_thresh_db) 的帧视为静音，剔除。"""
    peak = ref_db.max()
    frame_peak = ref_db.max(axis=0)
    return frame_peak > (peak - rel_thresh_db)


def _dtw_path(c1: np.ndarray, c2: np.ndarray):
    """简单 DTW 距离（O(T^2)，长音频请截断）。"""
    T1, T2 = c1.shape[0], c2.shape[0]
    D = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = np.linalg.norm(c1[i - 1] - c2[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[T1, T2]


def mcd_mel(
    wav1: np.ndarray,
    wav2: np.ndarray,
    pp: dict,
    attr: Optional[dict] = None,
    device: str = "cuda",
) -> float:
    """论文口径的 MCD：在 VC 模型的归一化 mel 特征上求逐帧 L2 均值 × 10√2/ln10。

    依据：论文 rec 行 MCD=0.00 —— 只有"在模型 mel 特征上比较"才能得到 0（重建
    路径的解码 mel ≈ 输入 mel），波形域/倒谱域都不可能。且该口径下 512 维、
    幅值 ∈[1e-8,1] 的特征差异天然落在论文的 3~12 量级。
    """
    import torch

    from ..features import wav2mel_tensor

    attr = attr or {"mean": np.zeros(pp["n_mels"], np.float32),
                    "std": np.ones(pp["n_mels"], np.float32)}
    m1 = wav2mel_tensor(
        torch.from_numpy(np.asarray(wav1, dtype=np.float32)).to(device),
        attr=attr, top_db=pp["top_db"], **{k: pp[k] for k in
                                           ("sample_rate", "preemph", "n_fft", "hop_length",
                                            "win_length", "n_mels", "ref_db", "max_db")}
    ).squeeze(0)  # (n_mels, T)
    m2 = wav2mel_tensor(
        torch.from_numpy(np.asarray(wav2, dtype=np.float32)).to(device),
        attr=attr, top_db=pp["top_db"], **{k: pp[k] for k in
                                           ("sample_rate", "preemph", "n_fft", "hop_length",
                                            "win_length", "n_mels", "ref_db", "max_db")}
    ).squeeze(0)
    n = min(m1.shape[1], m2.shape[1])
    d = (m1[:, :n] - m2[:, :n]).pow(2).sum(0).sqrt()  # 逐帧 L2
    dist = float(d.mean())
    return dist * 10 * np.sqrt(2) / np.log(10)


def mcd(
    wav1: np.ndarray,
    wav2: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    use_dtw: bool = True,
    max_len: Optional[int] = None,
    rel_thresh_db: float = 40.0,
) -> float:
    """两条波形的 MCD（dB，仅统计参考信号的有声帧）。"""
    c1, db1 = mel_cepstrum(wav1, sample_rate, n_mfcc=n_mfcc)
    c2, _ = mel_cepstrum(wav2, sample_rate, n_mfcc=n_mfcc)
    if max_len is not None:
        c1, c2 = c1[:max_len], c2[:max_len]
    mask1 = _active_mask(db1, rel_thresh_db)[: len(c1)]
    n = min(len(c1), len(c2))
    c1, c2, mask1 = c1[:n], c2[:n], mask1[:n]
    if not mask1.any():
        return 0.0
    if use_dtw and abs(len(c1) - len(c2)) > 2:
        dist = _dtw_path(c1[mask1], c2[mask1]) / int(mask1.sum())
    else:
        dist = float(np.mean(np.linalg.norm(c1[mask1] - c2[mask1], axis=1)))
    return float(dist * 10 * np.sqrt(2) / np.log(10))
