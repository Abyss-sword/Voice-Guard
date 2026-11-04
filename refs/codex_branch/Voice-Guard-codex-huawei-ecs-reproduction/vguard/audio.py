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


def restore_protected_crop(full, original_crop, protected_crop, start, end):
    """Keep every original sample outside the attack crop, including quiet speech.

    Account for PCM16 quantization in historical saved crops. Reject a crop from
    another waveform instead of silently inserting its perturbation at a wrong offset.
    """
    full = np.asarray(full, dtype=np.float32)
    original_crop = np.asarray(original_crop, dtype=np.float32)
    protected_crop = np.asarray(protected_crop, dtype=np.float32)
    if (full.ndim != 1 or original_crop.ndim != 1 or protected_crop.shape != original_crop.shape
            or not 0 <= start < end <= len(full) or end-start != len(original_crop)
            or not all(np.isfinite(wave).all() for wave in (full, original_crop, protected_crop))):
        raise ValueError("Invalid or misaligned protected crop")
    if not np.allclose(full[start:end], original_crop, rtol=0, atol=1/32768+2e-6):
        raise ValueError("Saved original crop does not match the full source")
    restored = full.copy()
    restored[start:end] += protected_crop-original_crop
    return restored
