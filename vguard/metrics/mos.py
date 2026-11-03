"""MOS 评估：MOSNet（HF 移植版，1NEYRON1/mosnet）。

论文用 MOSNet 评主观质量；此处用 HuggingFace 上的等价移植（原始权重在 Google
Drive 不可达）。模型文件已 curl 到 checkpoints/mosnet-hf/。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def load_mosnet(savedir: str = "checkpoints/mosnet-hf", device: Optional[str] = None):
    import torch
    from transformers import AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(savedir, trust_remote_code=True).to(device).eval()
    return model


def mos(model, wav: np.ndarray, sample_rate: int, target_sr: int = 16000) -> float:
    """单条波形 -> MOS（模型内部 STFT 512/256/512，输入重采样到 16k）。"""
    wav = np.asarray(wav, dtype=np.float32)
    if sample_rate != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=target_sr)
    scores = model.predict([wav])
    return float(scores[0])


def mos_batch(model, wavs: List[np.ndarray], sample_rate: int, target_sr: int = 16000):
    import librosa

    out = []
    for w in wavs:
        w = np.asarray(w, dtype=np.float32)
        if sample_rate != target_sr:
            w = librosa.resample(w, orig_sr=sample_rate, target_sr=target_sr)
        out.append(w)
    return [float(s) for s in model.predict(out)]
