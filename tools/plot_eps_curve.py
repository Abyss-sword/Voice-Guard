"""ε 扫描结果绘图：防御率 vs ε、F_imp/F_str 分解、MCD 曲线（T3 图）。"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(exp_dir):
    p = os.path.join(exp_dir, "results.csv")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dirs", required=True, help="逗号分隔（各 ε 档目录）")
    ap.add_argument("--labels", default="", help="逗号分隔的图例标签，缺省用 eps 列")
    ap.add_argument("--out", default="docs/fig_eps_curve.png")
    args = ap.parse_args()

    dirs = [d.strip() for d in args.exp_dirs.split(",") if d.strip()]
    labels = [l.strip() for l in args.labels.split(",") if l.strip()] or [os.path.basename(d) for d in dirs]

    eps_vals, dacc, fimp, fstr, mcd_mean = [], [], [], [], []
    for d in dirs:
        rows = load(d)
        if not rows:
            continue
        # 论文筛选协议：剔除 clean-VC 失败（conv_clean 未达目标人）的句子
        rows = [r for r in rows if str(r.get("conv_clean_hits_X")).lower() == "true"]
        if not rows:
            continue
        eps = float(rows[0]["eps"])
        n = len(rows)
        ok = sum(1 for r in rows if str(r["defense_ok"]).lower() == "true")
        imp = sum(1 for r in rows if r["asv_defended"] == r["utterance"].split("_")[0])
        fstr_n = 0
        for r in rows:
            if r.get("asv_conv_def") is not None and r["asv_conv_def"] != r["utterance"].split("_")[0]:
                fstr_n += 1
        mcd = np.mean([float(r["mcd"]) for r in rows if r.get("mcd")])
        eps_vals.append(eps)
        dacc.append(100 * ok / max(n, 1))
        fimp.append(100 * imp / max(n, 1))
        fstr.append(100 * fstr_n / max(n, 1))
        mcd_mean.append(mcd)
        print(f"eps={eps}: DAcc={dacc[-1]:.0f}% F_imp={fimp[-1]:.0f}% F_str={fstr[-1]:.0f}% MCD={mcd:.1f}")

    fig, ax1 = plt.subplots(figsize=(8, 4.6))
    ax1.plot(eps_vals, dacc, "o-", lw=2, color="tab:blue", label="Defense success (F_imp ∧ F_str)")
    ax1.plot(eps_vals, fimp, "s--", color="tab:green", label="F_imp (defended still X)")
    ax1.plot(eps_vals, fstr, "^--", color="tab:red", label="F_str (converted not X)")
    ax1.set_xlabel("ε")
    ax1.set_ylabel("rate (%)")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="center left")
    ax2 = ax1.twinx()
    ax2.plot(eps_vals, mcd_mean, "D:", color="gray", label="MCD (mean)")
    ax2.set_ylabel("MCD")
    ax2.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
