"""WER/CER 评估：转换音频的转写内容 vs 内容源文本（Whisper base）。

口径：内容源 t̂ 的转写为 ground truth；对每句的 defended / conv_clean / conv_def
转写并算 CER/WER。论文 Eq.9-11 中 ASR_th=10% 即用于此处（内容是否被保留）。

用法：
    python scripts/compute_wer.py --exp-dir outputs/exp02 --vctk data/VCTK22 \
        --source-speaker p226 --source-utt 2 --out outputs/exp02/wer.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vguard.metrics.wer import cer, load_whisper, transcribe, wer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--vctk", required=True)
    ap.add_argument("--source-speaker", required=True)
    ap.add_argument("--source-utt", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    gt_path = os.path.join(args.vctk, args.source_speaker,
                           f"{args.source_speaker}_{args.source_utt:03d}.txt")
    assert os.path.exists(gt_path), gt_path
    gt = open(gt_path, encoding="utf-8").read().strip()

    model = load_whisper("base", device=args.device)

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
            hyp = transcribe(model, fp)
            row[f"cer_{key}"] = round(cer(hyp, gt), 4)
            row[f"wer_{key}"] = round(wer(hyp, gt), 4)
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
