"""批量 MOS 评估（MOSNet，T2 严格指标）。

用法：
    python scripts/compute_mos.py --exp-dir outputs/exp02 \
        --out outputs/exp02/mos.csv --device cuda
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vguard.metrics.mos import load_mosnet, mos  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = load_mosnet(savedir="checkpoints/mosnet-hf", device=args.device)
    rows = []
    for d in sorted(os.listdir(args.exp_dir)):
        sub = os.path.join(args.exp_dir, d)
        if not os.path.isdir(sub) or not os.path.exists(os.path.join(sub, "defended.wav")):
            continue
        row = {"utterance": d}
        for key in ["original_trimmed", "defended", "conv_from_clean", "conv_from_defended"]:
            fp = os.path.join(sub, f"{key}.wav")
            if not os.path.exists(fp):
                continue
            w, sr = sf.read(fp)
            row[f"mos_{key}"] = round(mos(model, w, sr), 3)
        rows.append(row)
        print(row)

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(rows)
        print("saved:", args.out)


if __name__ == "__main__":
    main()
