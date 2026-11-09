"""人工听测结果统计（② 主观 MOS）。

读取 tools/listening_app.py 写入的 results.jsonl，输出：
- 每个条件（original / epsilon=0.02 / epsilon=0.05 …）的 MOS 均值、中位数、样本量、标准差
- 每条的偏好份额（参与者选为"最愿意接受"的比例）
- 逐句明细表（Markdown）+ 机器可读 JSON

用法：
    python tools/summarize_listening.py --results outputs/listening/results.jsonl \
        --out outputs/listening/summary.json --md outputs/listening/summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="outputs/listening/results.jsonl")
    ap.add_argument("--out", default="outputs/listening/summary.json")
    ap.add_argument("--md", default="outputs/listening/summary.md")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        raise SystemExit(f"缺少 {args.results}（先收集评分）")
    recs = [json.loads(line) for line in open(args.results, encoding="utf-8") if line.strip()]
    if not recs:
        raise SystemExit("结果为空")

    by_kind = defaultdict(list)
    pref_count = defaultdict(int)
    per_item = defaultdict(lambda: defaultdict(list))
    n_sessions = len({r["session"] for r in recs})

    for r in recs:
        for it in r["items"]:
            for o in it["order"]:
                score = it["ratings"].get(o["letter"])
                if score is None:
                    continue
                by_kind[o["kind"]].append(float(score))
                per_item[it["id"]][o["kind"]].append(float(score))
        pref_count[r.get("preferred", "?")] += 1

    def stats(vals):
        if not vals:
            return None
        return {"n": len(vals), "mean": round(statistics.mean(vals), 3),
                "median": round(statistics.median(vals), 3),
                "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                "min": min(vals), "max": max(vals)}

    summary = {"sessions": n_sessions, "records": len(recs),
               "mos_by_condition": {k: stats(v) for k, v in sorted(by_kind.items())},
               "preference_counts": dict(pref_count),
               "preference_share": {k: round(v / max(len(recs), 1), 3) for k, v in pref_count.items()},
               "per_item": {ut: {k: round(statistics.mean(v), 3) for k, v in cond.items()}
                            for ut, cond in per_item.items()}}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = ["# 人工听测（主观 MOS）结果汇总", "",
             f"- 参与者会话数：**{n_sessions}**；记录数：**{len(recs)}**", "",
             "## 各条件 MOS（1–5）", "",
             "| 条件 | n | 均值 | 中位数 | 标准差 | 范围 |", "|---|---:|---:|---:|---:|---|"]
    for k, v in summary["mos_by_condition"].items():
        lines.append(f"| {k} | {v['n']} | **{v['mean']}** | {v['median']} | {v['std']} | {v['min']}–{v['max']} |")
    lines += ["", "## 整体偏好份额", "", "| 选项 | 次数 | 份额 |", "|---|---:|---:|"]
    for k, v in summary["preference_share"].items():
        lines.append(f"| {k} | {pref_count[k]} | {v} |")
    lines += ["", "## 逐句均值", ""]
    kinds = sorted({k for cond in per_item.values() for k in cond})
    lines.append("| 句子 | " + " | ".join(kinds) + " |")
    lines.append("|---|" + "---:|" * len(kinds))
    for ut, cond in sorted(per_item.items()):
        lines.append(f"| {ut} | " + " | ".join(f"{statistics.mean(cond[k]):.2f}" if k in cond else "—"
                                               for k in kinds) + " |")
    lines += ["", "> 说明：主观分只反映本批听众对这 15 段音频的偏好；样本量小，不能外推为总体 MOS；",
              "> 若与 MOSNet 预测差异大，以主观分为准并如实报告两者差异。"]
    os.makedirs(os.path.dirname(os.path.abspath(args.md)) or ".", exist_ok=True)
    open(args.md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nsaved:", args.out, "and", args.md)


if __name__ == "__main__":
    main()
