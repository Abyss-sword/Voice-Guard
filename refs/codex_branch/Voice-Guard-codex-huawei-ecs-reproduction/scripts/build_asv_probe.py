"""闭集 ASV：AdaIN-VC 说话人编码器 Es 嵌入 + 线性分类头（对应论文在 VCTK 上训练的 GE2E）。

用法：
    python scripts/build_asv_probe.py --model-dir checkpoints/adain-vc-official \
        --vctk data/VCTK22 --speakers p225,... --train-offset 30 --train-per 200 \
        --val-offset 230 --val-per 20 --out outputs/asv_probe.npz

产物 asv_probe.npz: names / probe_weight (20x128) / probe_bias (20)
用法（vguard/asv.py 的 EsProbeASV）：Es(wav) -> 归一化 -> probe -> argmax 判人。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vguard.vc.adainvc import load_adainvc, speaker_embed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--vctk", required=True)
    ap.add_argument("--speakers", required=True)
    ap.add_argument("--train-offset", type=int, default=30)
    ap.add_argument("--train-per", type=int, default=200)
    ap.add_argument("--val-offset", type=int, default=230)
    ap.add_argument("--val-per", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    names = [s.strip() for s in args.speakers.split(",") if s.strip()]
    model, config, attr = load_adainvc(args.model_dir, device=args.device)
    pp = config["preprocess"]

    X, y, Xv, yv = [], [], [], []
    for si, n in enumerate(names):
        d = os.path.join(args.vctk, n)
        files = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        for fi in range(args.train_offset, min(args.train_offset + args.train_per, len(files))):
            w, sr = sf.read(os.path.join(d, files[fi]))
            if sr != pp["sample_rate"]:
                import librosa
                w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=pp["sample_rate"])
            X.append(speaker_embed(model, config, attr, w.astype(np.float32), args.device))
            y.append(si)
        for fi in range(args.val_offset, min(args.val_offset + args.val_per, len(files))):
            w, sr = sf.read(os.path.join(d, files[fi]))
            if sr != pp["sample_rate"]:
                import librosa
                w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=pp["sample_rate"])
            Xv.append(speaker_embed(model, config, attr, w.astype(np.float32), args.device))
            yv.append(si)
        print(f"{n}: {len(files)} files, train@{args.train_offset}..+{args.train_per}, val@{args.val_offset}..+{args.val_per}")

    X = torch.from_numpy(np.stack(X)).to(args.device)
    y = torch.tensor(y).to(args.device)
    Xv = torch.from_numpy(np.stack(Xv)).to(args.device)
    yv = torch.tensor(yv).to(args.device)
    print("train:", X.shape, "val:", Xv.shape)

    # 归一化后训练线性分类头
    Xn = torch.nn.functional.normalize(X, dim=1)
    Xvn = torch.nn.functional.normalize(Xv, dim=1)
    probe = torch.nn.Linear(X.shape[1], len(names)).to(args.device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    ce = torch.nn.CrossEntropyLoss()
    for epoch in range(200):
        opt.zero_grad()
        loss = ce(probe(Xn), y)
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == 199:
            with torch.no_grad():
                acc = (probe(Xvn).argmax(1) == yv).float().mean().item()
            print(f"epoch {epoch}: loss {loss.item():.4f} val_acc {acc:.4f}")

    np.savez(args.out,
             names=np.asarray(names),
             probe_weight=probe.weight.detach().cpu().numpy().astype(np.float32),
             probe_bias=probe.bias.detach().cpu().numpy().astype(np.float32))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
