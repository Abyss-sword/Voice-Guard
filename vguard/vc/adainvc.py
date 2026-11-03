"""AdaIN-VC（官方 NTU 权重）加载、预处理与重建。

官方链路（jjery2243542/adaptive_voice_conversion）：
  wav -> trim(top_db=15) -> 预加重 -> STFT(2048/300/1200) -> 512 维 mel
      -> 20log10 -> clip 归一化 -> (x-mean)/std -> AE.inference(src, tgt)
      -> 反归一化 -> mel->线性谱(伪逆) -> Griffin-Lim(n_iter=100) -> 去预加重 -> trim

Voice Guard 的攻击 H（features.wav2mel_tensor）与该预处理公式一致（除预加重
第 0 采样边界），故可端到端复现。
"""
from __future__ import annotations

import copy
import importlib.util
import os
import pickle
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import yaml

THIRD_PARTY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "ada_in_vc")
)


def _load_module(name: str, rel_path: str):
    path = os.path.join(THIRD_PARTY, rel_path)
    if THIRD_PARTY not in sys.path:
        sys.path.insert(0, THIRD_PARTY)
    if name == "adainvc_official_model":
        # 官方 model.py 只从 utils import cc；官方 utils.py 还 import tensorboardX，
        # 这里注入最小 utils 模块绕开该依赖。
        import types

        def cc(net):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return net.to(device)

        utils_mod = types.ModuleType("utils")
        utils_mod.cc = cc
        sys.modules["utils"] = utils_mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_adainvc(model_dir: str, device: str = "cuda"):
    """加载官方 AdaIN-VC。model_dir 含 config.yaml / attr.pkl / vctk_model.ckpt。"""
    import librosa  # noqa: F401  (model.py 不依赖，但统一环境校验)

    cfg_path = os.path.join(model_dir, "config.yaml")
    attr_path = os.path.join(model_dir, "attr.pkl")
    ckpt_path = os.path.join(model_dir, "vctk_model.ckpt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(model_dir, "model.ckpt")

    attr = pickle.load(open(attr_path, "rb"))
    config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    model_mod = _load_module("adainvc_official_model", "model.py")
    model = model_mod.AE(config).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if any(k.startswith("module.") for k in ckpt):
        ckpt = {k[7:]: v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.eval()
    return model, config, attr


# --------------------------------------------------------------------------- #
# 官方预处理 / 重建（转录自 preprocess/tacotron/utils.py，去掉 tensorflow 依赖）
# --------------------------------------------------------------------------- #
def get_spectrograms(fpath: str, **pp) -> Tuple[np.ndarray, np.ndarray]:
    """官方 get_spectrograms：返回 (mel, mag)，均为 (T, n_bins) float32。"""
    import librosa

    y, _ = librosa.load(fpath, sr=pp["sample_rate"])
    y, _ = librosa.effects.trim(y, top_db=pp["top_db"])
    y = np.append(y[0], y[1:] - pp["preemph"] * y[:-1])

    # 与训练时的旧版 librosa 行为一致：center=True 等价于显式 np.pad reflect n_fft//2
    # （librosa 1.0 改变了默认 padding，故这里显式实现）
    y = np.pad(y, pp["n_fft"] // 2, mode="reflect")
    linear = librosa.stft(
        y=y, n_fft=pp["n_fft"], hop_length=pp["hop_length"],
        win_length=pp["win_length"], center=False,
    )
    mag = np.abs(linear)

    mel_basis = librosa.filters.mel(sr=pp["sample_rate"], n_fft=pp["n_fft"], n_mels=pp["n_mels"])
    mel = np.dot(mel_basis, mag)

    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    mel = np.clip((mel - pp["ref_db"] + pp["max_db"]) / pp["max_db"], 1e-8, 1)
    mag = np.clip((mag - pp["ref_db"] + pp["max_db"]) / pp["max_db"], 1e-8, 1)

    return mel.T.astype(np.float32), mag.T.astype(np.float32)


def inv_mel_matrix(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    import librosa

    m = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    p = np.matmul(m, m.T)
    d = [1.0 / x if np.abs(x) > 1e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m.T, np.diag(d))


def griffin_lim(spect: np.ndarray, n_fft: int, hop_length: int, win_length: int,
                n_iter: int = 100) -> np.ndarray:
    import librosa

    X_best = copy.deepcopy(spect)
    for _ in range(n_iter):
        X_t = librosa.istft(X_best, hop_length=hop_length, win_length=win_length, window="hann")
        est = librosa.stft(X_t, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spect * phase
    X_t = librosa.istft(X_best, hop_length=hop_length, win_length=win_length, window="hann")
    return np.real(X_t)


def melspectrogram2wav(mel: np.ndarray, **pp) -> np.ndarray:
    """官方 melspectrogram2wav：mel (T, n_mels) -> wav。"""
    import librosa
    from scipy.signal import lfilter

    mel = mel.T
    mel = (np.clip(mel, 0, 1) * pp["max_db"]) - pp["max_db"] + pp["ref_db"]
    mel = np.power(10.0, mel * 0.05)

    m = inv_mel_matrix(pp["sample_rate"], pp["n_fft"], pp["n_mels"])
    mag = np.dot(m, mel)

    wav = griffin_lim(mag, pp["n_fft"], pp["hop_length"], pp["win_length"], n_iter=pp["n_iter"])
    wav = lfilter([1], [1, -pp["preemph"]], wav)
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)


def normalize(mel: np.ndarray, attr: Dict) -> np.ndarray:
    return (mel - attr["mean"]) / attr["std"]


def denormalize(mel: np.ndarray, attr: Dict) -> np.ndarray:
    return mel * attr["std"] + attr["mean"]


def speaker_embed(model, config, attr, wav: np.ndarray, device: str = "cuda") -> np.ndarray:
    """用 AdaIN-VC 的说话人编码器 Es 提取 128 维嵌入（攻击期身份判定的注册库）。"""
    from ..features import wav2mel_tensor

    pp = config["preprocess"]
    feature_kwargs = {k: pp[k] for k in
                      ("sample_rate", "preemph", "n_fft", "hop_length",
                       "win_length", "n_mels", "ref_db", "max_db", "top_db")}
    x = torch.from_numpy(wav.astype(np.float32)).to(device)
    with torch.no_grad():
        mel = wav2mel_tensor(x, attr=attr, **feature_kwargs)
        emb = model.speaker_encoder(mel)
    return emb.squeeze().cpu().numpy()


def convert(model, config, attr, source: str, target: str, out: str, device: str = "cuda") -> np.ndarray:
    """官方推理链路：source 提供内容，target 提供声纹。"""
    import soundfile as sf

    pp = config["preprocess"]
    src_mel, _ = get_spectrograms(source, **pp)
    tar_mel, _ = get_spectrograms(target, **pp)
    src_mel = torch.from_numpy(normalize(src_mel, attr)).T.unsqueeze(0).to(device)
    tar_mel = torch.from_numpy(normalize(tar_mel, attr)).T.unsqueeze(0).to(device)

    with torch.no_grad():
        dec = model.inference(src_mel, tar_mel)      # (1, 512, T')
        dec = dec.transpose(1, 2).squeeze(0)         # (T', 512)
    dec = denormalize(dec.cpu().numpy(), attr)
    wav = melspectrogram2wav(dec, **pp)
    sf.write(out, wav, pp["sample_rate"])
    return wav
