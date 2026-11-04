"""信道效应连续分析：对已信道化的 defended/conv 文件算 ECAPA 与 X 的相似度。

防御被压缩削弱的标志：conv_def_<ch> 的 sim_to_X 随压缩强度上升（转换开始"认出"X）。

用法：
    python scripts/channel_sim.py --exp-dir outputs/exp03_eps005 \
        --registry outputs/registry.npz --out outputs/exp03_eps005/channel_sim.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vguard.asv import embed_wav, load_ecapa  # noqa: E402

CHANNELS = ["none", "mp3-128k", "mp3-32k", "aac-64k", "g711", "resample8k"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reg = np.load(args.registry, allow_pickle=True)
    names = list(reg["names"])
    ecapa = load_ecapa(savedir="checkpoints/ecapa-local")

    def sim_to(wav, sr, target):
        e = embed_wav(ecapa, wav, sr)
        sims = np.asarray([float(np.dot(e, x) / (np.linalg.norm(e) * np.linalg.norm(x) + 1e-9))
                           for x in reg["embeddings"]])
        return float(sims[names.index(target)])

    rows = []
    for d in sorted(os.listdir(args.exp_dir)):
        sub = os.path.join(args.exp_dir, d)
        if not os.path.isdir(sub) or not os.path.exists(os.path.join(sub, "defended.wav")):
            continue
        target = d.split("_")[0]
        row = {"utterance": d}
        for ch in CHANNELS:
            fp_d = os.path.join(sub, f"defended_{ch}.wav")
            fp_c = os.path.join(sub, f"conv_def_{ch}.wav")
            if os.path.exists(fp_d):
                w, sr = sf.read(fp_d)
                row[f"simX_def_{ch}"] = round(sim_to(w.astype(np.float32), sr, target), 3)
            if os.path.exists(fp_c):
                w, sr = sf.read(fp_c)
                row[f"simX_conv_{ch}"] = round(sim_to(w.astype(np.float32), sr, target), 3)
        rows.append(row)
        print(row)

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(rows)
        print("saved:", args.out)
        # 汇总：各信道 simX_conv 均值
        print("=== mean sim_to_X of conv(defended) per channel ===")
        for ch in CHANNELS:
            vals = [r[f"simX_conv_{ch}"] for r in rows if f"simX_conv_{ch}" in r]
            if vals:
                print(f"{ch:12s} mean {np.mean(vals):.3f}")


if __name__ == "__main__":
    main()
