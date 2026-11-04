"""Audio inputs for attacks must share the VC model's sample rate and channels."""
from __future__ import annotations

import numpy as np
import soundfile as sf


def load_audio(path, sample_rate: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if not wav.size or not np.isfinite(wav).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    if sr != sample_rate:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    return np.asarray(wav, dtype=np.float32)
