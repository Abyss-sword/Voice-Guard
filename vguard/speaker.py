"""说话人注册表：官方缺失的 asr_model 模块的替代。

官方 attack_utils.py 依赖：
    from asr_model.find_most_unlikely_speaker import speaker_name   # 说话人名单
    speaker_embedding_dict = np.load('your-speaker_encoder_embedding_dict-path', ...)  # {name: emb}
    speaker_name[argmax(cosine(adv_emb, speaker_embedding_tensor))]  # 判定身份
    # least-likely 目标：对每位被保护说话人，选其余说话人中余弦相似度最低者

这里抽象成 SpeakerRegistry，可对接 GE2E 判别器（论文）或 ECAPA 兜底。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class SpeakerRegistry:
    """N 位说话人的嵌入表（N × D，float32）。"""

    def __init__(self, embeddings: np.ndarray, names: List[str]):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        assert embeddings.ndim == 2 and embeddings.shape[0] == len(names), (
            embeddings.shape, len(names))
        self.names = list(names)
        self._np = embeddings
        self.embeddings = torch.from_numpy(embeddings)

    @classmethod
    def from_dict(cls, d: Dict[str, np.ndarray]) -> "SpeakerRegistry":
        names = sorted(d.keys())
        embs = np.stack([np.asarray(d[n], dtype=np.float32).reshape(-1) for n in names])
        return cls(embs, names)

    def to(self, device) -> "SpeakerRegistry":
        self.embeddings = self.embeddings.to(device)
        return self

    @property
    def device(self):
        return self.embeddings.device

    def predict(self, emb: torch.Tensor) -> Tuple[str, torch.Tensor]:
        """argmax 余弦相似度 -> (姓名, 相似度向量)。"""
        sims = torch.cosine_similarity(
            emb.reshape(1, -1), self.embeddings, dim=1
        )
        return self.names[int(torch.argmax(sims))], sims

    def least_likely(self, target_name: str, target_emb: torch.Tensor) -> Tuple[str, np.ndarray]:
        """目标说话人之外，余弦相似度最低的说话人（论文的 least-likely 概念）。"""
        sims = torch.cosine_similarity(
            target_emb.reshape(1, -1), self.embeddings, dim=1
        ).detach().cpu().numpy()
        order = np.argsort(sims)
        for idx in order:
            if self.names[idx] != target_name:
                return self.names[idx], self._np[idx]
        raise ValueError(f"registry 中没有 target 以外的说话人: {target_name}")


class ProbeASV:
    """闭集 ASV：嵌入 + 线性分类头（对应论文在 VCTK 上训练的 GE2E 判别器）。

    用法：
        probe = ProbeASV(names, weight, bias, embed_fn)
        name, score = probe.predict(wav)   # embed_fn(wav) -> (D,) numpy
    """

    def __init__(self, names, weight, bias, embed_fn):
        self.names = list(names)
        self.w = torch.from_numpy(np.asarray(weight, dtype=np.float32))
        self.b = torch.from_numpy(np.asarray(bias, dtype=np.float32))
        self.embed_fn = embed_fn

    def to(self, device):
        self.w = self.w.to(device)
        self.b = self.b.to(device)
        return self

    def predict(self, wav: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        e = self.embed_fn(wav, sample_rate)
        e = torch.from_numpy(np.asarray(e, dtype=np.float32)).to(self.w.device)
        e = F.normalize(e, dim=0)
        logits = self.w @ e + self.b
        i = int(torch.argmax(logits))
        return self.names[i], float(logits[i])
