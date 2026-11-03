"""ASV（自动说话人验证）封装：ECAPA-TDNN 嵌入 + 余弦注册库。

官方仓库的 asr_model/（GE2E 判别器 + 嵌入库）未发布，这里用 speechbrain 的
ECAPA-TDNN（VoxCeleb 预训练）替代：对 VCTK 的 20 位说话人做注册（均值嵌入），
用 argmax 余弦相似度做身份判定。攻击与评测共用同一注册库，与官方机制一致。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _patch_symlink_fallback():
    """Windows 无软链接权限时，把 os.symlink 兜底为复制（speechbrain 需要）。"""
    import os
    import shutil

    if hasattr(os, "_symlink_patched"):
        return
    os._symlink_patched = True
    _orig = os.symlink

    def _symlink(src, dst, *a, **kw):
        try:
            return _orig(src, dst, *a, **kw)
        except OSError:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    os.symlink = _symlink


def load_ecapa(device: Optional[str] = None, savedir: str = "checkpoints/ecapa-local"):
    """加载 ECAPA-TDNN。优先本地目录（checkpoints/ecapa-local，已用 curl 下载好），
    否则走 HF（需 HF_ENDPOINT=hf-mirror.com，且本机 .cache 有 ACL 问题时需 HF_HOME 重定向）。
    """
    _patch_symlink_fallback()
    from speechbrain.inference.speaker import EncoderClassifier

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    savedir = os.path.abspath(savedir)
    os.makedirs(savedir, exist_ok=True)
    if os.path.exists(os.path.join(savedir, "hyperparams.yaml")):
        return EncoderClassifier.from_hparams(
            source=savedir, savedir=savedir, run_opts={"device": device}
        )
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        run_opts={"device": device},
    )


def embed_wav(classifier, wav: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """单条波形 -> (D,) 嵌入。ECAPA 内部重采样到 16k。"""
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.dataio.dataio import read_audio

    # 写入临时或直接构造 tensor：用 torch 转 16k 后 encode_batch
    x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    if sample_rate != 16000:
        import torchaudio.functional as F
        x = F.resample(x, sample_rate, 16000)
    with torch.no_grad():
        emb = classifier.encode_batch(x.to(classifier.device))
    return emb.squeeze().cpu().numpy()


def build_registry(
    classifier,
    speaker_wavs: Dict[str, List[np.ndarray]],
) -> Tuple[List[str], np.ndarray]:
    """每位说话人用其全部样本的均值嵌入注册。

    Args:
        speaker_wavs: {speaker_name: [wav, ...]}（wav 为 float32，采样率一致）
    Returns:
        (names, embeddings) — embeddings (N, D)
    """
    names = sorted(speaker_wavs.keys())
    embs = []
    for n in names:
        vs = [embed_wav(classifier, w, 16000) for w in speaker_wavs[n]]
        embs.append(np.mean(vs, axis=0))
    return names, np.stack(embs).astype(np.float32)
