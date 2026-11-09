"""把 exp 目录的结果（results.csv / wer.csv / reevaluate 输出）汇总成 markdown 表。

用法：
    python tools/build_tables.py --exp-dirs outputs/exp01,outputs/exp02 \
        --out docs/06_结果表.md [--probe outputs/asv_probe.npz]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vguard.run_state import summarize


def load_csv(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dirs", required=True, help="逗号分隔")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = ["# 实验结果汇总（自动生成）", ""]
    for exp in [e.strip() for e in args.exp_dirs.split(",") if e.strip()]:
        rows = load_csv(os.path.join(exp, "results.csv"))
        if not rows:
            continue
        lines.append(f"## {os.path.basename(exp)}")
        lines.append("")
        keys = rows[0].keys()
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "---|" * len(list(keys)))
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
        lines.append("")
        # 汇总
        stats = summarize(rows)
        def percent(rate):
            return "N/A" if rate is None else f"{100 * rate:.1f}%"
        lines.append(f"- 样本数 {stats['total']}；ASV 有效 {stats['evaluated']}；"
                     f"未筛选成功 {stats['unfiltered_successes']}/{stats['evaluated']}（{percent(stats['unfiltered_rate'])}）；"
                     f"clean-VC 筛选通过 {stats['screened']}/{stats['evaluated']}（{percent(stats['screen_pass_rate'])}）；"
                     f"筛选后成功 {stats['screened_successes']}/{stats['screened']}（{percent(stats['screened_rate'])}）。")
        lines.append("- 以上为身份维判定；未联合 MOS / 内容完整性阈值。")
        lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
