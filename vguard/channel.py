"""信道模拟：真实分发链路（ffmpeg 编解码）与可微 EOT 信道（训练用）。

真实信道（W6 加分项验证）：mp3/aac/opus 码率、8kHz 电话（G.711 μ-law）、响度归一化。
EOT 信道（训练用，可微）：随机重采样 / 低通 / 量化噪声 / 增益。
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 真实信道（ffmpeg）
# --------------------------------------------------------------------------- #
def _ffmpeg(in_path: str, out_path: str, codec_args) -> bool:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", in_path, *codec_args, out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def _read_write(wav: np.ndarray, sr: int) -> (str, str):
    import soundfile as sf

    d = tempfile.mkdtemp(prefix="vgch_")
    fin = os.path.join(d, "in.wav")
    fout = os.path.join(d, "out.wav")
    sf.write(fin, wav, sr)
    return fin, fout


def codec_channel(wav: np.ndarray, sr: int, spec: str) -> np.ndarray:
    """spec: 'mp3-128k' / 'mp3-32k' / 'aac-64k' / 'opus-24k' / 'g711'（8k μ-law）/ 'resample8k'"""
    import soundfile as sf

    fin, fout = _read_write(wav, sr)
    tmp = os.path.join(os.path.dirname(fin), "enc")
    ok = False
    if spec.startswith("mp3"):
        br = spec.split("-")[1]
        ok = _ffmpeg(fin, tmp + ".mp3", ["-codec:a", "libmp3lame", "-b:a", br])
        if ok:
            ok = _ffmpeg(tmp + ".mp3", fout, ["-ar", str(sr)])
    elif spec.startswith("aac"):
        br = spec.split("-")[1]
        ok = _ffmpeg(fin, tmp + ".m4a", ["-codec:a", "aac", "-b:a", br])
        if ok:
            ok = _ffmpeg(tmp + ".m4a", fout, ["-ar", str(sr)])
    elif spec.startswith("opus"):
        br = spec.split("-")[1]
        ok = _ffmpeg(fin, tmp + ".opus", ["-codec:a", "libopus", "-b:a", br])
        if ok:
            ok = _ffmpeg(tmp + ".opus", fout, ["-ar", str(sr)])
    elif spec == "g711":
        ok = _ffmpeg(fin, tmp + ".wav", ["-ar", "8000", "-ac", "1", "-codec:a", "pcm_mulaw"])
        if ok:
            ok = _ffmpeg(tmp + ".wav", fout, ["-ar", str(sr)])
    elif spec == "resample8k":
        ok = _ffmpeg(fin, tmp + ".wav", ["-ar", "8000"])
        if ok:
            ok = _ffmpeg(tmp + ".wav", fout, ["-ar", str(sr)])
    else:
        raise ValueError(spec)
    if not ok:
        raise RuntimeError(f"ffmpeg channel failed: {spec}")
    w, _ = sf.read(fout)
    return w.astype(np.float32)


def normalize_loudness(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = np.abs(wav).max()
    if m < 1e-6:
        return wav
    return (wav / m * peak).astype(np.float32)


# --------------------------------------------------------------------------- #
# 可微 EOT 信道（训练用）
# --------------------------------------------------------------------------- #
def differentiable_channel(x: torch.Tensor, sr: int, rng=None) -> torch.Tensor:
    """随机采样一条可微信道：重采样/低通/量化/增益。x: (L,)，输出保持长度 L。"""
    import torchaudio.functional as F_ta

    rng = rng or torch.Generator()  # torch.randint/rand 只接受 CPU 生成器
    L = len(x)
    x = x.clone()

    # 1) 随机重采样（往返，之后对齐回原长）
    rates = [sr, 22050, 16000, 12000, 8000]
    r = rates[int(torch.randint(0, len(rates), (1,), generator=rng))]
    if r != sr:
        x = F_ta.resample(x.unsqueeze(0), sr, r).squeeze(0)
        x = F_ta.resample(x.unsqueeze(0), r, sr).squeeze(0)
        if len(x) > L:
            x = x[:L]
        elif len(x) < L:
            x = F.pad(x, (0, L - len(x)))

    # 2) 随机低通（STFT 域）
    if torch.rand(1, generator=rng).item() < 0.7:
        n_fft = 2048
        cutoff = float(torch.randint(4, 10, (1,), generator=rng)) * 1000.0
        X = torch.stft(x, n_fft=n_fft, hop_length=300, win_length=1200,
                       window=torch.hann_window(1200, device=x.device),
                       center=True, pad_mode="reflect", normalized=False,
                       onesided=True, return_complex=True)
        freqs = torch.linspace(0, sr / 2, X.shape[-2], device=x.device)
        mask = (freqs <= cutoff).float().view(-1, 1)
        X = X * mask
        x = torch.istft(X, n_fft=n_fft, hop_length=300, win_length=1200,
                        window=torch.hann_window(1200, device=x.device),
                        center=True, length=L)

    # 3) 随机量化噪声（代理）
    if torch.rand(1, generator=rng).item() < 0.7:
        q = 10 ** float(torch.randint(-30, -14, (1,), generator=rng)) * 0.05
        x = x + (torch.rand(x.shape, generator=rng, device=x.device) - 0.5) * q

    # 4) 随机增益
    g_db = float(torch.randint(-30, 31, (1,), generator=rng))
    x = x * 10 ** (g_db / 20.0)

    return x
