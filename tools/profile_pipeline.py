"""L12 性能剖析：定位时域攻击的热点（STFT/mel/VC 重建/评估）并给出加速建议。

用法：
    python tools/profile_pipeline.py --model-dir checkpoints/adain-vc-official \
        --vctk data/VCTK22 --registry outputs/registry.npz \
        --attack-registry outputs/registry_vc.npz --device cuda
    python tools/profile_pipeline.py --stage protect --iters1 200 --iters2 100

输出：分阶段耗时表（ms）+ 占比，便于决定优化点（如缓存 mel 滤波器、减少 GL 迭代）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402


def timed(name, fn, rows):
    t0 = time.perf_counter()
    out = fn()
    dt = (time.perf_counter() - t0) * 1000
    rows.append((name, dt))
    print(f"  {name:<28}{dt:>9.1f} ms", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="checkpoints/adain-vc-official")
    ap.add_argument("--vctk", default="data/VCTK22")
    ap.add_argument("--registry", default="outputs/registry.npz")
    ap.add_argument("--attack-registry", default="outputs/registry_vc.npz")
    ap.add_argument("--speaker", default="p225")
    ap.add_argument("--utt-offset", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters1", type=int, default=300)
    ap.add_argument("--iters2", type=int, default=150)
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args()

    from vguard.features import wav2mel_tensor
    from vguard.masking import file2wav_mask
    from vguard.pipeline import Pipeline

    rows: list[tuple[str, float]] = []
    print("== 初始化 ==")
    pipe = timed("Pipeline 初始化（含模型加载）", lambda: Pipeline(
        args.model_dir, args.registry, args.attack_registry,
        ecapa_dir="checkpoints/ecapa-local", device=args.device), rows)
    pp = pipe.pp
    sr = pp["sample_rate"]

    wavs = sorted(f for f in os.listdir(os.path.join(args.vctk, args.speaker)) if f.endswith(".wav"))
    defend = os.path.join(args.vctk, args.speaker, wavs[args.utt_offset])
    src_dir = os.path.join(args.vctk, "p226")
    adv = os.path.join(src_dir, sorted(f for f in os.listdir(src_dir) if f.endswith(".wav"))[2])
    x, _ = sf.read(defend, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    print(f"测试句：{os.path.basename(defend)}（{len(x)/sr:.2f}s）")

    print("== 组件级 ==")
    model = pipe.model
    attr = pipe.attr
    feat_kwargs = dict(n_fft=pp["n_fft"], hop_length=pp["hop_length"],
                       win_length=pp["win_length"], n_mels=pp["n_mels"],
                       sample_rate=sr, preemph=pp.get("preemph", 0.97))
    x_t = torch.from_numpy(x).to(args.device)
    timed("wav2mel_tensor ×20", lambda: [wav2mel_tensor(x_t, attr=attr, **feat_kwargs) for _ in range(20)], rows)
    timed("speaker_encoder ×20", lambda: [model.speaker_encoder(
        wav2mel_tensor(x_t, attr=attr, **feat_kwargs)) for _ in range(20)], rows)
    timed("file2wav_mask", lambda: file2wav_mask(
        defend, sample_rate=sr, n_fft=pp["n_fft"], hop_length=pp["hop_length"],
        win_length=pp["win_length"], top_db=pp["top_db"]), rows)

    print("== 端到端 protect ==")
    out_dir = os.path.join("outputs", "profile_run")
    rep = timed("Pipeline.protect", lambda: pipe.protect(
        defend, args.speaker, adv, eps=args.eps,
        iters1=args.iters1, iters2=args.iters2, out_dir=out_dir,
        progress=False), rows)

    total = sum(v for _, v in rows if _ not in ("Pipeline 初始化（含模型加载）", "Pipeline.protect"))
    print("\n== 汇总 ==")
    for name, ms in rows:
        print(f"{name:<32}{ms:>10.1f} ms")
    print(f"\nprotect 内部：{rep.get('elapsed_s')} s（{args.iters1}+{args.iters2} 步 @ {args.device}）")
    print("建议：若 wav2mel/speaker_encoder 占比高 → 缓存 mel 滤波器与 STFT 窗；"
          "若 protect 内部远高 → 主要成本在逐样本扰动优化，可考虑 batch（L7）或减少 GL 迭代。")


if __name__ == "__main__":
    main()
