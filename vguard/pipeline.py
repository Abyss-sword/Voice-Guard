"""可复用管线：保护 + 转换 + 指标，供 CLI、Gradio 前端与 FastAPI 服务共用。"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch

from .attack import voice_guard_attack
from .masking import file2wav_mask
from .speaker import SpeakerRegistry
from .vc.adainvc import convert as vc_convert
from .vc.adainvc import load_adainvc


class Pipeline:
    """Voice Guard 保护服务（单例加载模型，串行执行）。"""

    def __init__(self, model_dir: str, registry_path: str, attack_registry_path: str,
                 ecapa_dir: str = "checkpoints/ecapa-local", device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, self.config, self.attr = load_adainvc(model_dir, device=self.device)
        self.pp = self.config["preprocess"]
        self.feature_kwargs = {k: self.pp[k] for k in
                               ("sample_rate", "preemph", "n_fft", "hop_length",
                                "win_length", "n_mels", "ref_db", "max_db", "top_db")}
        reg = np.load(registry_path, allow_pickle=True)
        self.eval_registry = SpeakerRegistry(reg["embeddings"], list(reg["names"])).to(self.device)
        atk = np.load(attack_registry_path, allow_pickle=True)
        self.attack_registry = SpeakerRegistry(atk["embeddings"], list(atk["names"])).to(self.device)
        self._eval_embs = np.asarray(reg["embeddings"], dtype=np.float32)
        self._ecapa_dir = ecapa_dir
        self._ecapa = None

    @property
    def speakers(self):
        return self.attack_registry.names

    def least_likely(self, target: str) -> str:
        e = self.attack_registry.embeddings[self.attack_registry.names.index(target)]
        y, _ = self.attack_registry.least_likely(target, e)
        return y

    def _ecapa_asv(self, wav: np.ndarray) -> Dict:
        from .asv import embed_wav, load_ecapa
        if self._ecapa is None:
            self._ecapa = load_ecapa(savedir=self._ecapa_dir)
        e = embed_wav(self._ecapa, wav, self.pp["sample_rate"])
        sims = np.asarray([
            float(np.dot(e, x) / (np.linalg.norm(e) * np.linalg.norm(x) + 1e-9))
            for x in self._eval_embs])
        i = int(np.argmax(sims))
        return {"pred": self.eval_registry.names[i],
                "sim": {n: round(float(s), 3) for n, s in zip(self.eval_registry.names, sims)}}

    def protect(self, defend_path: str, target_speaker: str, adv_path: Optional[str],
                eps: float = 0.1, iters1: int = 3000, iters2: int = 1500,
                out_dir: str = "outputs/web", progress: bool = True) -> Dict:
        """执行保护并保存产物，返回报告 dict。"""
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()

        if adv_path is None:
            adv_path = os.path.join(out_dir, "_adv_tmp.wav")
            y = self.least_likely(target_speaker)
            raise NotImplementedError("需要提供 least-likely 音频路径（adv_path）")

        wav, theta, psd_max = file2wav_mask(
            defend_path, sample_rate=self.pp["sample_rate"], n_fft=self.pp["n_fft"],
            hop_length=self.pp["hop_length"], win_length=self.pp["win_length"],
            top_db=self.pp["top_db"])
        adv_wav, _ = sf.read(adv_path)
        adv_wav = adv_wav.astype(np.float32)

        res = voice_guard_attack(
            self.model, torch.from_numpy(wav), vc_tgt_mask=theta,
            vc_tgt_psd_max=float(psd_max), adv_tgt=torch.from_numpy(adv_wav),
            registry=self.attack_registry, vc_tgt_speaker=target_speaker, eps=eps,
            n_iters1=iters1, n_iters2=iters2, c_init=1.0, attr=self.attr,
            progress=progress, **self.feature_kwargs)
        adv_np = res["adv_wav"].numpy().astype(np.float32)
        sr = self.pp["sample_rate"]

        defended = os.path.join(out_dir, "defended.wav")
        original = os.path.join(out_dir, "original_trimmed.wav")
        delta = os.path.join(out_dir, "delta.wav")
        sf.write(defended, adv_np, sr)
        sf.write(original, wav.astype(np.float32), sr)
        sf.write(delta, (adv_np - wav).astype(np.float32), sr)

        # 转换对比（内容源 = least-likely 音频）
        conv_clean = os.path.join(out_dir, "conv_from_clean.wav")
        conv_def = os.path.join(out_dir, "conv_from_defended.wav")
        vc_convert(self.model, self.config, self.attr, adv_path, original, conv_clean, self.device)
        vc_convert(self.model, self.config, self.attr, adv_path, defended, conv_def, self.device)

        asv = {}
        try:
            asv["defended"] = self._ecapa_asv(adv_np)
            c1, _ = sf.read(conv_clean)
            asv["conv_clean"] = self._ecapa_asv(c1.astype(np.float32))
            c2, _ = sf.read(conv_def)
            asv["conv_def"] = self._ecapa_asv(c2.astype(np.float32))
        except Exception as e:  # noqa: BLE001
            asv["error"] = str(e)[:200]

        report = {
            "target_speaker": target_speaker,
            "eps": eps, "iters1": iters1, "iters2": iters2,
            "c_final": res["c"], "defense_flag": res["flag"], "stage": res["stage"],
            "duration_s": round(len(wav) / sr, 2),
            "elapsed_s": round(time.time() - t0, 1),
            "asv": asv,
            "files": {"defended": defended, "original": original,
                      "delta": delta, "conv_clean": conv_clean, "conv_def": conv_def},
        }
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report
