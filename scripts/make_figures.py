"""案例可视化：波形对比 / 掩蔽阈值 vs 扰动 PSD / 频谱对比（报告与前端用）。

用法：
    python scripts/make_figures.py --utt-dir outputs/exp02/p225_040 \
        --sample-rate 24000 --n-fft 2048 --hop 300 --win 1200
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from vguard.masking import generate_th, perturbation_psd  # noqa: E402


def fig_waveforms(w_orig, w_def, delta, sr, out):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    t0 = np.arange(len(w_orig)) / sr
    t1 = np.arange(len(delta)) / sr
    ax.plot(t0, w_orig, lw=0.6, label="original")
    ax.plot(t0[: len(w_def)], w_def[: len(t0)], lw=0.6, alpha=0.85, label="defended")
    ax.plot(t1, delta * 5, lw=0.7, alpha=0.7, label="perturbation x5")
    ax.legend(loc="upper right")
    ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_masking(w_orig, delta, sr, n_fft, hop, win, out):
    theta, psd_max = generate_th(w_orig.astype(np.float32), sample_rate=sr,
                                 n_fft=n_fft, hop_length=hop, win_length=win)
    PSD_d = perturbation_psd(delta, n_fft=n_fft, hop_length=hop,
                             win_length=win, psd_max=psd_max)
    n = min(len(theta), len(PSD_d.T))
    theta_mean = theta[:n].mean(axis=0)
    psd_mean = PSD_d.T[:n].mean(axis=0)
    freqs = np.linspace(0, sr / 2, len(theta_mean))
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.semilogx(freqs[1:], theta_mean[1:], label="masking threshold θx (mean)")
    ax.semilogx(freqs[1:], psd_mean[1:], label="perturbation PSD (mean)", alpha=0.85)
    ax.set_xlabel("frequency (Hz, log)")
    ax.set_ylabel("dB (normalized)")
    ax.legend()
    ax.set_title("Perturbation hidden under the masking threshold")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_spectrograms(w_orig, w_def, sr, out):
    import librosa
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
    for ax, w, title in [(axes[0], w_orig, "original"), (axes[1], w_def, "defended")]:
        S = librosa.feature.melspectrogram(y=w.astype(np.float32), sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S)
        im = ax.imshow(S_db, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utt-dir", required=True)
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=300)
    ap.add_argument("--win", type=int, default=1200)
    args = ap.parse_args()

    w_orig, sr = sf.read(os.path.join(args.utt_dir, "original_trimmed.wav"))
    w_def, _ = sf.read(os.path.join(args.utt_dir, "defended.wav"))
    delta = w_def[: len(w_orig)] - w_orig
    delta = delta.astype(np.float32)
    w_orig = w_orig.astype(np.float32)

    fig_waveforms(w_orig, w_def, delta, sr,
                  os.path.join(args.utt_dir, "fig_waveforms.png"))
    fig_masking(w_orig, delta, sr, args.n_fft, args.hop, args.win,
                os.path.join(args.utt_dir, "fig_masking.png"))
    fig_spectrograms(w_orig, w_def, sr,
                     os.path.join(args.utt_dir, "fig_spectrograms.png"))
    print("figures saved to", args.utt_dir)


if __name__ == "__main__":
    main()
