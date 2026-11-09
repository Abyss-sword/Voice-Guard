"""C11：内容源扫描——同一被保护句 × 多个内容源，检验防御结论对内容源的敏感性。

用法（ECS / 本机）：
    python scripts/run_content_sweep.py --speakers p225,p226,p227 \
        --source-utts 0,1,2 --iters1 300 --iters2 150 --out-root outputs/content_sweep

产物：<out-root>/src{utt}_<method>/（每个内容源一个实验目录）+ content_summary.json
汇总：每个内容源的防御率（筛选后），以及跨内容源的均值/极差。
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speakers", default="p225,p226,p227")
    ap.add_argument("--source-speaker", default="p226")
    ap.add_argument("--source-utts", default="0,1,2")
    ap.add_argument("--method", default="vg")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--iters1", type=int, default=250)
    ap.add_argument("--iters2", type=int, default=120)
    ap.add_argument("--model-dir", default="checkpoints/adain-vc-official")
    ap.add_argument("--vctk", default="data/VCTK22")
    ap.add_argument("--registry", default="outputs/registry.npz")
    ap.add_argument("--attack-registry", default="outputs/registry_vc.npz")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-root", default="outputs/content_sweep")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    root = ROOT / args.out_root
    root.mkdir(parents=True, exist_ok=True)
    log = root / "run.log"
    summary = {}
    for su in [int(s) for s in args.source_utts.split(",") if s.strip()]:
        out_dir = root / f"src{su}_{args.method}"
        results = out_dir / "results.csv"
        if args.reuse and results.exists():
            print(f"[skip] {out_dir} 已有结果")
        else:
            cmd = [sys.executable, "scripts/run_experiment.py",
                   "--model-dir", args.model_dir, "--vctk", args.vctk,
                   "--speakers", args.speakers, "--per-speaker", "1",
                   "--utt-offset", "30", "--max-dur", "2.5",
                   "--registry", args.registry, "--attack-registry", args.attack_registry,
                   "--source-speaker", args.source_speaker, "--source-utt", str(su),
                   "--method", args.method, "--eps", str(args.eps),
                   "--iters1", str(args.iters1), "--iters2", str(args.iters2),
                   "--seed", str(args.seed), "--out-dir", str(out_dir),
                   "--device", args.device, "--no-progress"]
            print("+", " ".join(cmd), flush=True)
            with open(log, "a", encoding="utf-8") as f:
                r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                print(f"[warn] 内容源 {su} 失败（exit {r.returncode}）")
                continue
        rows = list(csv.DictReader(open(results, encoding="utf-8"))) if results.exists() else []
        screened = [r for r in rows if str(r.get("conv_clean_hits_X")) == "True"]
        ok = sum(1 for r in screened if str(r.get("defense_ok")) == "True")
        summary[f"src{su}"] = {"n": len(rows), "screened": len(screened), "ok": ok,
                               "rate": round(ok / len(screened), 3) if screened else None}
        print(f"src{su}: screened {len(screened)}/{len(rows)} ok={ok}")

    rates = [v["rate"] for v in summary.values() if v["rate"] is not None]
    payload = {"speakers": args.speakers, "method": args.method, "eps": args.eps,
               "iters": [args.iters1, args.iters2], "per_source": summary,
               "rate_mean": round(sum(rates) / len(rates), 3) if rates else None,
               "rate_range": [min(rates), max(rates)] if rates else None}
    (root / "content_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
