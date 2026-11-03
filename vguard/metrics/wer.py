"""Whisper 识别：CER / WER（论文用 Whisper 评语音内容完整性）。"""
from __future__ import annotations

from typing import Optional


def load_whisper(model_name: str = "base", device: Optional[str] = None):
    import whisper

    if device is None:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    return whisper.load_model(model_name, device=device)


def _normalize(text: str) -> str:
    """与 whisper 的 BasicTextNormalizer 对齐（英文小写、去标点、数字展开）。"""
    from whisper.normalizers import BasicTextNormalizer

    return BasicTextNormalizer()(text)


def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def cer(hyp: str, ref: str) -> float:
    h, r = _normalize(hyp), _normalize(ref)
    if not r:
        return float("inf")
    return _edit_distance(h, r) / len(r)


def wer(hyp: str, ref: str) -> float:
    h, r = _normalize(hyp).split(), _normalize(ref).split()
    if not r:
        return float("inf")
    return _edit_distance(h, r) / len(r)


def transcribe(model, wav_path: str) -> str:
    return model.transcribe(wav_path, language="en")["text"].strip()
