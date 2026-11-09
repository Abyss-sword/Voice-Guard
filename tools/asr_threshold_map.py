"""ASR 阈值映射：把论文 Eq.10 的 ASR_th 语义映射到 Whisper WER 分布。

对已有实验产物统计四类音频的 WER/CER 分布（目标句文本 / 内容源文本），
并回答"若 ASR_th=10%，多少比例满足/不满足"，作为论文 F_str 内容维的可解释代理。

用法：
    python tools/asr_threshold_map.py --exp-dirs outputs/exp03_eps005,outputs/exp02 \
        --vctk data/VCTK22 --source-speaker p226 --source-utt 2 \
        --device cuda --out outputs/asr_threshold_map.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vguard.metrics.wer import cer, load_whisper, transcribe, wer  # noqa: E402

KEYS = ["original_trimmed", "defended", "conv_from_clean", "conv_from_defended"]
THRESHOLDS = [0.10, 0.20, 0.30]


def sample_std(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dirs", required=True)
    ap.add_argument("--vctk", default="data/VCTK22")
    ap.add_argument("--source-speaker", default="p226")
    ap.add_argument("--source-utt", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="每个实验目录最多取样句数（0=全部）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/asr_threshold_map.json")
    args = ap.parse_args()

    src_txt = os.path.join(args.vctk, args.source_speaker, f"{args.source_speaker}_{args.source_utt:03d}.txt")
    if not os.path.exists(src_txt):
        sys.exit(f"内容源文本缺失：{src_txt}")
    content_text = open(src_txt, encoding="utf-8").read().strip()

    dirs = [d.strip() for d in args.exp_dirs.split(",") if d.strip()]
    files = []
    for d in dirs:
        subs = sorted(s for s in glob.glob(os.path.join(d, "*")) if os.path.isdir(s))
        if args.limit:
            subs = subs[: args.limit]
        for sub in subs:
            for k in KEYS:
                fp = os.path.join(sub, f"{k}.wav")
                if os.path.exists(fp):
                    files.append((d, os.path.basename(sub), k, fp))
    if not files:
        sys.exit("没有找到实验产物（检查 --exp-dirs）")
    print(f"待转写文件：{len(files)} 个（Whisper + {args.device}）")

    model = load_whisper("base", device=args.device)
    recs = []
    for i, (d, utt, k, fp) in enumerate(files, 1):
        hyp = transcribe(model, fp)
        spk = utt.split("_")[0]
        if k.startswith("conv_from_"):
            ref = content_text
        else:
            t = os.path.join(args.vctk, spk, f"{utt}.txt")
            ref = open(t, encoding="utf-8").read().strip() if os.path.exists(t) else content_text
        recs.append({"exp_dir": d, "utterance": utt, "kind": k,
                     "wer": round(wer(hyp, ref), 4), "cer": round(cer(hyp, ref), 4)})
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    summary = {}
    for k in KEYS:
        vals = [r["wer"] for r in recs if r["kind"] == k]
        if not vals:
            continue
        n = len(vals)
        row = {"n": n, "mean": round(sum(vals) / n, 4),
               "median": round(sorted(vals)[n // 2], 4),
               "std": round(sample_std(vals), 4), "min": round(min(vals), 4),
               "max": round(max(vals), 4)}
        for th in THRESHOLDS:
            row[f"frac_le_{int(th*100)}pct"] = round(sum(1 for v in vals if v <= th) / n, 3)
        summary[k] = row

    by_exp = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by_exp[r["exp_dir"]][r["kind"]].append(r["wer"])

    payload = {"source_text": src_txt, "thresholds": THRESHOLDS,
               "summary_by_kind": summary,
               "by_exp": {d: {k: round(sum(v) / len(v), 4) for k, v in kv.items()}
                          for d, kv in by_exp.items()},
               "records": recs}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n=== WER 分布（各类别）===")
    print(f"{'类别':<22}{'n':>3}{'mean':>8}{'median':>8}{'std':>8}{'≤10%':>7}{'≤20%':>7}{'≤30%':>7}")
    for k, r in summary.items():
        print(f"{k:<22}{r['n']:>3}{r['mean']:>8.3f}{r['median']:>8.3f}{r['std']:>8.3f}"
              f"{r['frac_le_10pct']:>7.2f}{r['frac_le_20pct']:>7.2f}{r['frac_le_30pct']:>7.2f}")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
