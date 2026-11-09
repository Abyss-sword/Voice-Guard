"""Voice Guard 攻击主流程：Eq.3 / Eq.7 / Eq.8 + Algorithm 1。

忠实转录自官方 WHUB401/Voice-Guard ``attack_utils.py``（emb 防御流），
并补全官方缺失的依赖（Transform 已补在 masking.py，说话人注册表在 speaker.py）。

流程（Algorithm 1）：
  阶段1（"强"）：只优化 L_vc = MSE(emb(x+δ), emb(y)) - 0.1·MSE(emb(x+δ), emb(x))
                 Adam(默认 lr=1e-3)，T1=3000 步，δ = 2·ε·tanh(w)。
  阶段2（"隐"）：loss = L_vc + c·mean(ReLU(PSD_δ - θx))，T2=1500 步，
                 c 做 8 轮二分（初始 1.0，成功乘 2，失败与上次成功值折半），
                 目标是找到"仍能防御成功"的最大 c（噪声藏得最深）。

与官方代码一致的两处细节：
- 扰动幅值系数是 2·ε（论文写 ε·tanh(w)，代码是 vc_tgt + 2*eps*tanh(ptb)）。
- 阶段2 的 while 循环在成功分支多自增一次 i（官方原样，保留）。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import trange

from .features import wav2mel_tensor
from .masking import Transform
from .speaker import SpeakerRegistry


def refine_masked(model, original, initial_delta, adv_target, theta, psd_max, attr,
                  feature_kwargs, steps=1000, learning_rate=0.0003, margin_db=3.0,
                  eps=0.05, checkpoint_fn=None):
    """Experimental projected optimization. The original Voice Guard stays unchanged.

    Refine the same speaker-encoder surrogate inside a masking constraint. The
    independent ECAPA evaluator, MOS model and Whisper are not part of this loss.
    """
    from .masking import MaskingProjection
    if steps < 1 or learning_rate <= 0 or eps <= 0:
        raise ValueError("Positive optimization settings required")
    device = next(model.parameters()).device
    original = torch.as_tensor(original, dtype=torch.float32, device=device)
    adv_target = torch.as_tensor(adv_target, dtype=torch.float32, device=device)
    projector = MaskingProjection(theta, psd_max, **{key: feature_kwargs[key]
        for key in ("n_fft", "hop_length", "win_length")}, margin_db=margin_db).to(device)
    delta = torch.as_tensor(initial_delta, dtype=torch.float32, device=device).clone()
    with torch.no_grad():
        for _ in range(30):
            delta = projector(delta).clamp(-2*eps, 2*eps)
        original_embedding = model.speaker_encoder(wav2mel_tensor(original, attr=attr, **feature_kwargs))
        target_embedding = model.speaker_encoder(wav2mel_tensor(adv_target, attr=attr, **feature_kwargs))
    delta = delta.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)
    trace = []
    for step in range(1, steps+1):
        embedding = model.speaker_encoder(wav2mel_tensor(original+delta, attr=attr, **feature_kwargs))
        loss = torch.nn.functional.mse_loss(embedding, target_embedding) - .1*torch.nn.functional.mse_loss(embedding, original_embedding)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.copy_(projector(delta).clamp(-2*eps, 2*eps))
        if step % 100 == 0 or step == steps:
            trace.append({"step": step, "surrogate_loss": float(loss.detach())})
            print(trace[-1], flush=True)
            if checkpoint_fn is not None:
                checkpoint_fn(step, (original+delta).detach().cpu().numpy())
    with torch.no_grad():
        for _ in range(30):
            delta.copy_(projector(delta).clamp(-2*eps, 2*eps))
    return (original+delta).detach().cpu().numpy(), trace


def get_att_dis(target: Tensor, behaviored: Tensor) -> Tensor:
    """target 与注册库每条嵌入的余弦相似度分布（official get_att_dis）。"""
    attention_distribution = []
    for i in range(behaviored.size(0)):
        score = torch.cosine_similarity(target, behaviored[i].view(1, -1))
        attention_distribution.append(score)
    return torch.stack(attention_distribution, dim=0)


def attack_flag_of(adv_emb: Tensor, registry: SpeakerRegistry, target_name: str) -> bool:
    """防御成功 = argmax 余弦相似度不再命中被保护说话人。"""
    sims = get_att_dis(adv_emb, registry.embeddings)
    pred = registry.names[int(torch.argmax(sims))]
    return pred != target_name


def _stage1(
    model: nn.Module,
    vc_tgt: Tensor,
    org_emb: Tensor,
    tgt_emb: Tensor,
    registry: SpeakerRegistry,
    vc_tgt_speaker: str,
    eps: float,
    n_iters: int,
    attr: Dict,
    feature_kwargs: Dict,
    progress: bool = True,
    flag_fn=None,
    progress_cb=None,
) -> Tuple[Tensor, bool]:
    """阶段1：只求破坏力。返回 (ptb, 最终是否防御成功)。"""
    ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True)
    opt = torch.optim.Adam([ptb])  # 官方默认 lr=1e-3
    criterion = nn.MSELoss()

    attack_flag = False
    rng = trange(1, n_iters + 1, desc="stage1", disable=not progress)
    for i in rng:
        adv_inp = vc_tgt + 2.0 * eps * ptb.tanh()
        adv_emb = model.speaker_encoder(wav2mel_tensor(adv_inp, attr=attr, **feature_kwargs))

        if flag_fn is not None:
            attack_flag = flag_fn(adv_emb.detach())
        else:
            attack_flag = attack_flag_of(adv_emb, registry, vc_tgt_speaker)

        loss_emb_l2 = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb)
        loss = loss_emb_l2

        if i % 100 == 0:
            rng.set_postfix(loss=float(loss), flag=attack_flag)
        if progress_cb is not None:
            _stride = max(1, min(50, n_iters // 20))
            if i % _stride == 0 or i == n_iters:
                progress_cb(0.5 * i / max(n_iters, 1), "stage1")

        opt.zero_grad()
        loss.backward()
        opt.step()

    return ptb, attack_flag


def band_gain_from_theta(theta: torch.Tensor, tau_db: float = 6.0) -> torch.Tensor:
    """由掩蔽阈 θx 导出频带增益 m(f)∈(0,1]：阈值高的频带允许更多扰动能量。

    theta: (n_frames, n_freq) 的掩蔽阈（dB，与 PSD 同尺度）。
    返回 (n_freq,) 的频率增益，归一化到最大值 1。
    """
    mean_db = theta.mean(dim=0)                    # (n_freq,)
    z = (mean_db - mean_db.median()) / max(tau_db, 1e-6)
    g = torch.sigmoid(z)
    return g / g.max().clamp_min(1e-6)


def apply_band_gain(pe: torch.Tensor, gain: torch.Tensor, n_fft: int,
                    hop_length: int, win_length: int) -> torch.Tensor:
    """把频率增益作用到时域扰动上（STFT 域逐频带缩放后重建）。"""
    win = torch.hann_window(win_length, device=pe.device)
    squeezed = pe.dim() == 1
    spec = torch.stft(pe, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      window=win, center=True, pad_mode="reflect", return_complex=True)
    # spec: 1-D 输入 -> (freq, frames)；多维输入 -> (..., freq, frames)
    spec = spec * gain.reshape((-1,) + (1,) * (spec.dim() - 1))
    out = torch.istft(spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      window=win, center=True, length=pe.numel())
    return out.reshape(pe.shape) if squeezed else out


def _stage2(
    model: nn.Module,
    vc_tgt: Tensor,
    vc_tgt_mask: Tensor,
    vc_tgt_psd_max: float,
    org_emb: Tensor,
    tgt_emb: Tensor,
    registry: SpeakerRegistry,
    vc_tgt_speaker: str,
    eps: float,
    n_iters: int,
    c: float,
    stage1_ptb: Tensor,
    psd_transformer: Transform,
    attr: Dict,
    feature_kwargs: Dict,
    progress: bool = True,
    channel_fn=None,
    flag_fn=None,
    imp_channel_fn=None,
    imp_weight: float = 0.0,
    progress_cb=None,
    band_gain: Tensor = None,
) -> Tuple[bool, Tensor, int]:
    """阶段2：加掩蔽约束。官方语义：迭代中一旦防御失败立即返回。

    channel_fn（EOT，可选）：对 adv_inp 施加可微信道后再算身份与损失，
    使扰动对信道稳健；掩蔽损失仍作用于原始 adv_inp。
    flag_fn（可选）：自定义"防御成功"判定（如 Es 线性探头），默认注册库 argmax。
    imp_channel_fn / imp_weight（防御侧 EOT，可选）：在 adv_inp 上施加可微信道后
    约束其身份嵌入仍接近原说话人（F_imp 信道项），防止信道把防御音频自身的
    声纹破坏；imp_weight=0 时行为与官方完全一致。
    """
    ptb = stage1_ptb.detach().clone().requires_grad_(True)
    vc_tgt = vc_tgt.detach().clone().requires_grad_(False)
    opt = torch.optim.Adam([ptb])
    criterion = nn.MSELoss()
    relu = nn.ReLU()

    attack_flag = False
    adv_inp = vc_tgt
    rng = trange(1, int(n_iters) + 1, desc=f"stage2(c={c:.3g})", disable=not progress)
    for i in rng:
        adv_inp = vc_tgt + 2.0 * eps * (apply_band_gain(ptb.tanh(), band_gain, feature_kwargs["n_fft"],
                                                         feature_kwargs["hop_length"],
                                                         feature_kwargs["win_length"])
                                        if band_gain is not None else ptb.tanh())
        adv_inp_eval = channel_fn(adv_inp) if channel_fn is not None else adv_inp
        adv_emb = model.speaker_encoder(wav2mel_tensor(adv_inp_eval, attr=attr, **feature_kwargs))

        attack_flag = flag_fn(adv_emb.detach()) if flag_fn is not None else \
            attack_flag_of(adv_emb, registry, vc_tgt_speaker)

        logits_delta = psd_transformer(adv_inp - vc_tgt, vc_tgt_psd_max)
        loss_th = torch.mean(relu(logits_delta - vc_tgt_mask))

        loss_emb_l2 = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb)
        loss = loss_emb_l2 + c * loss_th

        # 防御侧 EOT：信道后的防御音频仍应保持原说话人身份（保护 F_imp）
        if imp_channel_fn is not None and imp_weight > 0:
            adv_imp = imp_channel_fn(adv_inp)
            imp_emb = model.speaker_encoder(wav2mel_tensor(adv_imp, attr=attr, **feature_kwargs))
            loss_imp_ch = criterion(imp_emb, org_emb.detach())
            loss = loss + imp_weight * loss_imp_ch

        if i % 100 == 0:
            rng.set_postfix(loss=float(loss), th=float(loss_th), flag=attack_flag)
        if progress_cb is not None:
            _stride = max(1, min(50, n_iters // 20))
            if i % _stride == 0 or i == n_iters:
                progress_cb(0.5 + 0.5 * min(1.0, i / max(n_iters, 1)), f"stage2(c={c:.2g})")

        if attack_flag is False:  # 官方语义：一旦不防御成功即返回
            return attack_flag, adv_inp, i

        opt.zero_grad()
        loss.backward()
        opt.step()

    return attack_flag, adv_inp, int(n_iters)


def voice_guard_attack(
    model: nn.Module,
    vc_tgt: Tensor,
    vc_tgt_mask: Tensor,
    vc_tgt_psd_max: float,
    adv_tgt: Tensor,
    registry: SpeakerRegistry,
    vc_tgt_speaker: str,
    eps: float,
    n_iters1: int = 3000,
    n_iters2: int = 1500,
    c_init: float = 1.0,
    attr: Dict = None,
    progress: bool = True,
    channel_fn=None,
    flag_fn=None,
    imp_channel_fn=None,
    imp_weight: float = 0.0,
    progress_cb=None,
    band_alloc: bool = False,
    **feature_kwargs,
) -> Dict:
    """Voice Guard 完整攻击。返回 dict：{"adv_wav", "c", "flag", "steps"}。

    channel_fn（EOT，可选）：阶段2 中对扰动后音频施加可微信道（信道稳健性训练）。
    flag_fn（可选）：自定义防御成功判定 adv_emb -> bool（如 Es 线性探头）。
    imp_channel_fn / imp_weight（防御侧 EOT，可选）：阶段2 增加"信道后仍保留原身份"
    的正则项（F_imp 信道稳健），默认关闭。
    progress_cb（可选）：fn(fraction 0..1, desc)，供前端实时进度条。
    """
    attr = attr or {"mean": np.float32(0.0), "std": np.float32(1.0)}
    device = next(model.parameters()).device
    vc_tgt = vc_tgt.to(device).detach()
    adv_tgt = adv_tgt.to(device).detach()
    vc_tgt_mask = torch.as_tensor(vc_tgt_mask, device=device, dtype=torch.float32)

    with torch.no_grad():
        org_emb = model.speaker_encoder(wav2mel_tensor(vc_tgt, attr=attr, **feature_kwargs))
        tgt_emb = model.speaker_encoder(wav2mel_tensor(adv_tgt, attr=attr, **feature_kwargs))

    psd_transformer = Transform(
        n_fft=feature_kwargs["n_fft"],
        hop_length=feature_kwargs["hop_length"],
        win_length=feature_kwargs["win_length"],
    ).to(device)

    # L3：按掩蔽阈分配频带能量（阈值高的频带允许更多扰动）
    band_gain = band_gain_from_theta(vc_tgt_mask.to(device)) if band_alloc else None

    # ---- 阶段1 ----
    stage1_ptb, stage1_flag = _stage1(
        model, vc_tgt, org_emb, tgt_emb, registry, vc_tgt_speaker,
        eps, n_iters1, attr, feature_kwargs, progress, flag_fn, progress_cb,
    )

    if stage1_flag is False:
        # 官方：阶段1 一步都没成功（step==1）直接返回当前结果
        return {"adv_wav": (vc_tgt + 2.0 * eps * stage1_ptb.tanh()).detach().cpu(),
                "c": float(c_init), "flag": False, "stage": 1}

    # ---- 阶段2：c 的 8 轮二分（官方 while 语义，成功分支多自增一次）----
    c = float(c_init)
    final_adv_inp = vc_tgt + 2.0 * eps * stage1_ptb.tanh()
    attack_step_2_c = 0.0
    false_c = 0.0
    i = 0
    while i < 8:
        flag, adv_inp, _ = _stage2(
            model, vc_tgt, vc_tgt_mask, vc_tgt_psd_max, org_emb, tgt_emb,
            registry, vc_tgt_speaker, eps, n_iters2, c, stage1_ptb,
            psd_transformer, attr, feature_kwargs, progress, channel_fn, flag_fn,
            imp_channel_fn, imp_weight, progress_cb, band_gain,
        )
        if flag and false_c == 0:
            attack_step_2_c = c
            c = c * 2.0
            final_adv_inp = adv_inp
            i = i + 1
        elif flag and false_c != 0:
            attack_step_2_c = c
            c = (c + false_c) / 2.0
            final_adv_inp = adv_inp
            i = i + 1
        else:
            false_c = c
            c = (attack_step_2_c + c) / 2.0
        i = i + 1

    return {"adv_wav": final_adv_inp.detach().cpu(),
            "c": float(attack_step_2_c), "flag": attack_step_2_c > 0, "stage": 2}
