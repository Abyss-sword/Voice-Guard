"""MOS 评估：MOSNet（HF 移植版，1NEYRON1/mosnet）。

使用第三方 HuggingFace 移植，不承诺与论文权重或人工 MOS 等价。
先用 tools/download_models.py mosnet 下载固定提交至 checkpoints/mosnet-hf/。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def load_mosnet(savedir: str = "checkpoints/mosnet-hf", device: Optional[str] = None):
    import os

    # 本机 C:\Users\<user>\.cache 有 ACL 问题（WinError 5），把 HF 缓存重定向到项目目录
    _hf_home = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".hfcache"))
    os.environ.setdefault("HF_HOME", _hf_home)
    os.environ.setdefault("HF_MODULES_CACHE", os.path.join(_hf_home, "modules"))

    import torch
    from transformers import AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # 离线加载（auto_map 已改为本地模块名；禁止走 hub）
    model = AutoModel.from_pretrained(
        savedir, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
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
