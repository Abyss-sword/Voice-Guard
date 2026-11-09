"""WER/CER 评估：原始/保护音频对照目标文本，转换音频对照内容源文本。

默认使用 VCTK 人工转录；显式 --reference-mode whisper 可使用干净音频的伪参考。
这里报告 Whisper base 的编辑距离，不将论文 ASR_th 直接等同于 WER 阈值。

用法：
    python scripts/compute_wer.py --exp-dir outputs/exp02 --vctk data/VCTK22 \
        --source-speaker p226 --source-utt 2 --out outputs/exp02/wer.csv
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--reference-mode", choices=["transcript", "whisper"], default="transcript",
                    help="transcript uses VCTK ground truth; whisper uses clean-audio pseudo references")
    args = ap.parse_args()

    gt_path = os.path.join(args.vctk, args.source_speaker,
                           f"{args.source_speaker}_{args.source_utt:03d}.txt")
    model = load_whisper("base", device=args.device)
    def reference(text_path, wav_path):
        if args.reference_mode == "transcript":
            if not os.path.exists(text_path):
                raise FileNotFoundError(f"VCTK transcript missing: {text_path}; provide transcripts or explicitly use --reference-mode whisper")
            with open(text_path, encoding="utf-8") as f:
                return f.read().strip()
        return transcribe(model, wav_path)
    source_path = os.path.join(args.vctk, args.source_speaker,
                               f"{args.source_speaker}_{args.source_utt:03d}.wav")
    source_gt = reference(gt_path, source_path)

    rows = []
    for d in sorted(os.listdir(args.exp_dir)):
        sub = os.path.join(args.exp_dir, d)
        if not os.path.isdir(sub) or not os.path.exists(os.path.join(sub, "defended.wav")):
            continue
        speaker = d.split("_")[0]
        target_gt = reference(os.path.join(args.vctk, speaker, f"{d}.txt"),
                              os.path.join(sub, "original_trimmed.wav"))
        row = {"utterance": d, "reference_mode": args.reference_mode}
        for key in ["original_trimmed", "defended", "conv_from_clean", "conv_from_defended"]:
            fp = os.path.join(sub, f"{key}.wav")
            if not os.path.exists(fp):
                continue
            hyp = transcribe(model, fp)
            # Protected audio retains the TARGET sentence; converted audio retains SOURCE content.
            gt = source_gt if key.startswith("conv_") else target_gt
            row[f"cer_{key}"] = round(cer(hyp, gt), 4)
            row[f"wer_{key}"] = round(wer(hyp, gt), 4)
        rows.append(row)
        print(row)

    if rows:
        from vguard.run_state import write_csv
        write_csv(args.out, rows)
        print("saved:", args.out)


if __name__ == "__main__":
    main()
