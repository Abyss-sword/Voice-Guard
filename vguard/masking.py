"""心理声学掩蔽阈值（Appendix A）+ 可微扰动 PSD。

- 掩蔽阈值 θx 用 numpy 一次性算好（无需梯度），忠实转录自官方
  ``generate_masking_threshold.py``。
- ``Transform`` 是官方仓库**缺失**的类：它计算扰动 δ 的 PSD，并归一化到与 θx
  相同的 96 dB 基准（用原始音频的峰值 ``psd_max``），使 hinge 损失
  ``mean(ReLU(PSD_delta - theta_x))``（论文 Eq.7）成立。
"""
from __future__ import annotations

import math

import librosa
import numpy as np
import torch
import torch.nn as nn
from scipy import signal


# --------------------------------------------------------------------------- #
# numpy 参考实现（不可导），转录自 generate_masking_threshold.py
# --------------------------------------------------------------------------- #
def compute_PSD_matrix(audio: np.ndarray, n_fft: int, hop_length: int, win_length: int):
    # 官方用旧版 librosa.stft 的 center=True（= 显式 np.pad reflect n_fft//2）；
    # librosa 1.0 改了默认 padding，这里显式复现旧行为。
    audio = np.pad(audio, n_fft // 2, mode="reflect")
    win = np.sqrt(8.0 / 3.0) * librosa.stft(
        audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False
    )
    z = np.abs(win / win_length)
    psd_max = np.max(z * z)
    psd = 10 * np.log10(z * z + 1e-19)
    PSD = 96 - np.max(psd) + psd
    return PSD, psd_max


def Bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def quiet(f):
    return (
        3.64 * (f * 0.001) ** -0.8
        - 6.5 * np.exp(-0.6 * (0.001 * f - 3.3) ** 2)
        + 0.001 * (0.001 * f) ** 4
        - 12
    )


def two_slops(bark_psd, delta_TM, bark_maskee):
    Ts = []
    for tone_mask in range(bark_psd.shape[0]):
        bark_masker = bark_psd[tone_mask, 0]
        dz = bark_maskee - bark_masker
        zero_index = np.argmax(dz > 0)
        sf = np.zeros(len(dz))
        sf[:zero_index] = 27 * dz[:zero_index]
        sf[zero_index:] = (-27 + 0.37 * max(bark_psd[tone_mask, 1] - 40, 0)) * dz[zero_index:]
        T = bark_psd[tone_mask, 1] + delta_TM[tone_mask] + sf
        Ts.append(T)
    return Ts


def compute_th(PSD, barks, ATH, freqs):
    length = len(PSD)
    masker_index = signal.argrelextrema(PSD, np.greater)[0]

    # 官方代码用 np.delete(0)/np.delete(length-1) 删边界掩蔽者（实际是死代码，
    # argrelextrema 不会返回端点）；这里用等价写法，输出完全一致。
    if 0 in masker_index:
        masker_index = np.delete(masker_index, np.where(masker_index == 0)[0])
    if length - 1 in masker_index:
        masker_index = np.delete(masker_index, np.where(masker_index == length - 1)[0])
    num_local_max = len(masker_index)

    p_k = pow(10, PSD[masker_index] / 10.0)
    p_k_prev = pow(10, PSD[masker_index - 1] / 10.0)
    p_k_post = pow(10, PSD[masker_index + 1] / 10.0)
    P_TM = 10 * np.log10(p_k_prev + p_k + p_k_post)

    _BARK, _PSD, _INDEX = 0, 1, 2
    bark_psd = np.zeros([num_local_max, 3])
    bark_psd[:, _BARK] = barks[masker_index]
    bark_psd[:, _PSD] = P_TM
    bark_psd[:, _INDEX] = masker_index

    # 忠实转录官方 while 循环（含其索引移动的原始行为）
    for i in range(num_local_max):
        next_ = i + 1
        if next_ >= bark_psd.shape[0]:
            break
        while bark_psd[next_, _BARK] - bark_psd[i, _BARK] < 0.5:
            if quiet(freqs[int(bark_psd[i, _INDEX])]) > bark_psd[i, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=0)
            if next_ == bark_psd.shape[0]:
                break
            if bark_psd[i, _PSD] < bark_psd[next_, _PSD]:
                bark_psd = np.delete(bark_psd, (i), axis=0)
            else:
                bark_psd = np.delete(bark_psd, (next_), axis=0)
            if next_ == bark_psd.shape[0]:
                break

    delta_TM = 1 * (-6.025 - 0.275 * bark_psd[:, 0])
    Ts = two_slops(bark_psd, delta_TM, barks)
    Ts = np.array(Ts)
    theta_x = 10 * np.log10(np.sum(pow(10, Ts / 10.0), axis=0) + pow(10, ATH / 10.0))
    return theta_x


def perturbation_psd(delta: np.ndarray, n_fft: int, hop_length: int, win_length: int,
                     psd_max: float) -> np.ndarray:
    """扰动 PSD（numpy 版），按原始音频峰值 psd_max 归一化到 θx 同一 96 dB 尺度。"""
    d = np.pad(np.asarray(delta, dtype=np.float32), n_fft // 2, mode="reflect")
    win = np.sqrt(8.0 / 3.0) * librosa.stft(
        d, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False)
    z = np.abs(win / win_length)
    psd = 10 * np.log10(z * z + 1e-19)
    ref = 96 - 10 * np.log10(float(psd_max) + 1e-19)
    return ref + psd  # (n_freq, n_frames)


def generate_th(audio: np.ndarray, sample_rate: int, n_fft: int, hop_length: int, win_length: int):
    """返回 (theta_xs, psd_max)。theta_xs 形状 (n_frames, n_freq)。"""
    PSD, psd_max = compute_PSD_matrix(
        audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    barks = Bark(freqs)

    ATH = np.zeros(len(barks)) - np.inf
    bark_ind = np.argmax(barks > 1)
    ATH[bark_ind:] = quiet(freqs[bark_ind:])

    theta_xs = []
    for i in range(PSD.shape[1]):
        theta_xs.append(compute_th(PSD[:, i], barks, ATH, freqs))
    theta_xs = np.array(theta_xs)
    return theta_xs, psd_max


def file2wav_mask(audio_path: str, sample_rate: int, n_fft: int, hop_length: int,
                  win_length: int, top_db: float):
    """载入 -> trim -> clip(-1,1) -> 算 θx。返回 (wav, theta_xs, psd_max)。

    对应官方 ``data_utils.file2wav_mask``：trim/clip 在这里用 numpy 做一次，
    之后 H 在返回的 wav 上纯可微。
    """
    wav, _ = librosa.load(audio_path, sr=sample_rate)
    wav, _ = librosa.effects.trim(wav, top_db=top_db)
    wav = np.clip(wav, -1, 1)
    theta_xs, psd_max = generate_th(
        wav, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    return wav, theta_xs, psd_max


# --------------------------------------------------------------------------- #
# 官方缺失的类：可微扰动 PSD（Eq.7 所需）
# --------------------------------------------------------------------------- #
class Transform(nn.Module):
    """计算扰动 δ 的归一化 PSD，输出与 θx 同布局，供 hinge 损失使用。

    重建自官方 ``attack_utils.py`` 的调用点（该类从未发布）：
        psd_transformer = Transform(**kwargs)
        logits_delta = psd_transformer(adv_inp - vc_tgt, vc_tgt_psd_max)
        loss_th = mean(ReLU(logits_delta - vc_tgt_mask))
    用原始音频的峰值 ``psd_max`` 做 96 dB 归一化，使扰动水平与掩蔽阈值同尺度。
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, delta_wav: torch.Tensor, psd_max: float) -> torch.Tensor:
        window = torch.hann_window(self.win_length, device=delta_wav.device)
        linear = torch.stft(
            delta_wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )  # (n_freq, n_frames, 2)
        mag = torch.sqrt(linear.pow(2).sum(-1) + 1e-19)
        z = math.sqrt(8.0 / 3.0) * mag / self.win_length
        psd = 10 * torch.log10(z.pow(2) + 1e-19)
        ref = 96 - 10 * math.log10(float(psd_max) + 1e-19)
        PSD = ref + psd                    # (n_freq, n_frames)
        return PSD.T                       # (n_frames, n_freq) 与 theta_xs 对齐
