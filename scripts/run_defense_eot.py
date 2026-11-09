"""L1 防御侧 EOT 实验：对比基线 VG 与"信道后 F_imp 正则"版 VG 在真实信道下的防御率。

流程：
  1. 对 N 个说话人各 1 句跑基线 VG（`run_experiment.py --method vg`）
  2. 同参数跑防御侧 EOT 版（`--imp-eot --imp-eot-weight W`）
  3. 两个目录分别做 6 信道扫描（`channel_sweep.py`）
  4. 汇总每信道 ECAPA/probe 防御率并输出对比 JSON

用法：
    python scripts/run_defense_eot.py --model-dir checkpoints/adain-vc-official \
        --vctk data/VCTK22 --registry outputs/registry.npz \
        --attack-registry outputs/registry_vc.npz --speakers p225,p226,p227,p228,p229 \
        --iters1 800 --iters2 400 --imp-weight 0.5 --out-root outputs/l1_defense_eot --device cuda

产物：
    outputs/l1_defense_eot/base/, outputs/l1_defense_eot/imp/
    outputs/l1_defense_eot/channel_results_{base,imp}.csv
    outputs/l1_defense_eot/summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, log: Path, tag: str):
    print(f"+ [{tag}] {' '.join(map(str, cmd))}", flush=True)
    with open(log, "a", encoding="utf-8") as f:
        p = subprocess.run([str(c) for c in cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit(f"{tag} 失败（exit {p.returncode}），见 {log}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="checkpoints/adain-vc-official")
    ap.add_argument("--vctk", default="data/VCTK22")
    ap.add_argument("--registry", default="outputs/registry.npz")
    ap.add_argument("--attack-registry", default="outputs/registry_vc.npz")
    ap.add_argument("--probe", default="outputs/asv_probe.npz")
    ap.add_argument("--speakers", default="p225,p226,p227,p228,p229")
    ap.add_argument("--per-speaker", type=int, default=1)
    ap.add_argument("--source-speaker", default="p226")
    ap.add_argument("--source-utt", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--iters1", type=int, default=800)
    ap.add_argument("--iters2", type=int, default=400)
    ap.add_argument("--imp-weight", type=float, default=0.5)
    ap.add_argument("--channels", default="none,mp3-128k,mp3-32k,aac-64k,g711,resample8k")
    ap.add_argument("--out-root", default="outputs/l1_defense_eot")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse", action="store_true", help="已有 results.csv 的目录跳过")
    args = ap.parse_args()

    py = sys.executable
    root = ROOT / args.out_root
    root.mkdir(parents=True, exist_ok=True)
    log = root / "run.log"

    common = [py, "scripts/run_experiment.py", "--model-dir", args.model_dir,
              "--vctk", args.vctk, "--registry", args.registry,
              "--attack-registry", args.attack_registry,
              "--speakers", args.speakers, "--per-speaker", str(args.per_speaker),
              "--utt-offset", "30", "--max-dur", "2.5",
              "--source-speaker", args.source_speaker, "--source-utt", str(args.source_utt),
              "--eps", args.eps, "--iters1", args.iters1, "--iters2", args.iters2,
              "--method", "vg", "--device", args.device, "--no-progress"]

    dirs = {"base": root / "base", "imp": root / "imp"}
    for tag, d in dirs.items():
        if args.reuse and (d / "results.csv").exists():
            print(f"[skip] {d} 已有结果")
            continue
        cmd = common + ["--out-dir", d]
        if tag == "imp":
            cmd += ["--imp-eot", "--imp-eot-weight", str(args.imp_weight)]
        run(cmd, log, tag)

    # 信道扫描
    for tag, d in dirs.items():
        out_csv = root / f"channel_results_{tag}.csv"
        if args.reuse and out_csv.exists():
            print(f"[skip] {out_csv} 已存在")
            continue
        run([py, "scripts/channel_sweep.py", "--exp-dir", d,
             "--model-dir", args.model_dir, "--registry", args.registry,
             "--probe", args.probe, "--channels", args.channels,
             "--out", out_csv, "--device", args.device], log, f"channels:{tag}")

    # 汇总
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    summary = {}
    for tag in dirs:
        path = root / f"channel_results_{tag}.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        per = {}
        for ch in channels:
            n = len(rows)
            def _ok(prefix):
                return sum(1 for r in rows if str(r.get(f"{prefix}_{ch}")).strip() == "True")
            per[ch] = {"n": n,
                       "ecapa_rate": round(_ok("ec_ok") / n, 3) if n else None,
                       "probe_rate": round(_ok("pr_ok") / n, 3) if n else None}
        summary[tag] = per

    out_json = root / "summary.json"
    out_json.write_text(json.dumps({"speakers": args.speakers, "eps": args.eps,
                                    "iters": [args.iters1, args.iters2],
                                    "imp_weight": args.imp_weight,
                                    "per_channel": summary}, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n{'信道':<12}{'基线 ECAPA':>12}{'防御EOT ECAPA':>14}{'基线 probe':>12}{'EOT probe':>11}")
    for ch in channels:
        b = summary.get("base", {}).get(ch, {})
        i = summary.get("imp", {}).get(ch, {})
        print(f"{ch:<12}{str(b.get('ecapa_rate')):>12}{str(i.get('ecapa_rate')):>14}"
              f"{str(b.get('probe_rate')):>12}{str(i.get('probe_rate')):>11}")
    print(f"\nsaved: {out_json}")


if __name__ == "__main__":
    main()
