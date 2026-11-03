"""频域基线攻击（emb / fb / e2e，转录自 attack-vc）与 VG w/o P 消融。

- emb/fb/e2e：在 mel 域加扰动 δ=ε·tanh(w)，再经声码器回波形（论文中的跨域损耗来源）。
  直接复用 third_party/attack-vc/attack_utils.py 的三个函数（接口与官方 AE 一致）。
- VG w/o P：时域扰动但不加心理声学掩蔽损失（论文 Tab.2 消融）。
"""
from __future__ import annotations

import importlib.util
import os
from typing import Dict

import numpy as np
import torch

from .attack import attack_flag_of, get_att_dis  # noqa: F401
from .features import wav2mel_tensor
from .vc.adainvc import denormalize, get_spectrograms, load_adainvc, melspectrogram2wav, normalize

THIRD_PARTY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "third_party", "attack-vc")
)


def _load_attack_utils():
    spec = importlib.util.spec_from_file_location(
        "attackvc_attack_utils", os.path.join(THIRD_PARTY, "attack_utils.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wav_to_mel_tensor(wav: np.ndarray, config: Dict, attr: Dict, device: str):
    """波形 -> 归一化 mel (1, n_mels, T)（与 attack-vc 的 file2mel+normalize 一致）。"""
    pp = config["preprocess"]
    # 与官方 get_spectrograms 相同的预处理（trim+preemph+pad+stft+mel+log+clip）
    mel, _ = _file2mel_from_wav(wav, pp)
    mel = normalize(mel, attr)
    return torch.from_numpy(mel).T.unsqueeze(0).to(device)


def _file2mel_from_wav(wav: np.ndarray, pp: Dict):
    import librosa

    y = np.append(wav[0], wav[1:] - pp["preemph"] * wav[:-1])
    y = np.pad(y, pp["n_fft"] // 2, mode="reflect")
    linear = librosa.stft(y=y, n_fft=pp["n_fft"], hop_length=pp["hop_length"],
                          win_length=pp["win_length"], center=False)
    mag = np.abs(linear)
    mb = librosa.filters.mel(sr=pp["sample_rate"], n_fft=pp["n_fft"], n_mels=pp["n_mels"])
    mel = 20 * np.log10(np.maximum(1e-5, np.dot(mb, mag)))
    mel = np.clip((mel - pp["ref_db"] + pp["max_db"]) / pp["max_db"], 1e-8, 1)
    return mel.T.astype(np.float32), mag.T.astype(np.float32)


def freq_domain_attack(
    method: str,
    model, config, attr,
    defend_wav: np.ndarray,
    adv_wav: np.ndarray,
    source_wav: np.ndarray,
    eps: float,
    n_iters: int,
    device: str = "cuda",
):
    """method in {emb, fb, e2e}。返回 (defended_wav, adv_mel)。"""
    assert method in ("emb", "fb", "e2e")
    pp = config["preprocess"]
    au = _load_attack_utils()

    vc_tgt = wav_to_mel_tensor(defend_wav, config, attr, device)
    adv_tgt = wav_to_mel_tensor(adv_wav, config, attr, device)

    if method == "emb":
        adv_mel = au.emb_attack(model, vc_tgt, adv_tgt, eps, n_iters)
    else:
        vc_src = wav_to_mel_tensor(source_wav, config, attr, device)
        if method == "fb":
            adv_mel = au.fb_attack(model, vc_src, vc_tgt, adv_tgt, eps, n_iters)
        else:
            adv_mel = au.e2e_attack(model, vc_src, vc_tgt, adv_tgt, eps, n_iters)

    adv_mel = adv_mel.squeeze(0).T.detach().cpu().numpy()
    adv_mel = denormalize(adv_mel, attr)
    wav = melspectrogram2wav(adv_mel, **pp)
    return wav.astype(np.float32), adv_mel


def vg_wo_p(
    model, config, attr,
    defend_wav: np.ndarray,
    adv_wav: np.ndarray,
    registry,
    target_speaker: str,
    eps: float,
    n_iters1: int,
    n_iters2: int,
    device: str = "cuda",
    progress: bool = True,
):
    """VG w/o P：时域 emb 攻击，两个阶段都不加掩蔽损失。"""
    from tqdm import trange

    pp = config["preprocess"]
    feature_kwargs = {k: pp[k] for k in
                      ("sample_rate", "preemph", "n_fft", "hop_length",
                       "win_length", "n_mels", "ref_db", "max_db", "top_db")}
    x = torch.from_numpy(defend_wav.astype(np.float32)).to(device).detach()
    y = torch.from_numpy(adv_wav.astype(np.float32)).to(device).detach()

    with torch.no_grad():
        org_emb = model.speaker_encoder(wav2mel_tensor(x, attr=attr, **feature_kwargs))
        tgt_emb = model.speaker_encoder(wav2mel_tensor(y, attr=attr, **feature_kwargs))

    ptb = torch.zeros_like(x).normal_(0, 1).requires_grad_(True)
    opt = torch.optim.Adam([ptb])
    criterion = torch.nn.MSELoss()

    for phase, iters in ((1, n_iters1), (2, n_iters2)):
        for i in trange(1, iters + 1, desc=f"vg-woP-s{phase}", disable=not progress):
            adv_inp = x + 2.0 * eps * ptb.tanh()
            adv_emb = model.speaker_encoder(wav2mel_tensor(adv_inp, attr=attr, **feature_kwargs))
            loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    adv_inp = (x + 2.0 * eps * ptb.tanh()).detach()
    with torch.no_grad():
        adv_emb = model.speaker_encoder(wav2mel_tensor(adv_inp, attr=attr, **feature_kwargs))
        flag = attack_flag_of(adv_emb, registry, target_speaker)
    return adv_inp.cpu().numpy().astype(np.float32), flag
