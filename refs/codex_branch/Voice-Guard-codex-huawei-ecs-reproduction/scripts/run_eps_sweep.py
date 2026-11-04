"""ε 扫描驱动器：同一批短句在不同 ε 下跑 VG 攻击（T3 实验）。

用法：
    python scripts/run_eps_sweep.py --model-dir checkpoints/adain-vc-official \
        --vctk data/VCTK22 --speakers p225,p226,p227 --per-speaker 2 --utt-offset 30 --max-dur 2.2 \
        --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz \
        --source-speaker p226 --source-utt 2 --epss 0.02,0.05,0.08,0.1 \
        --iters1 800 --iters2 400 --out-base outputs/exp03 --device cuda
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--vctk", required=True)
    ap.add_argument("--speakers", required=True)
    ap.add_argument("--per-speaker", type=int, default=2)
    ap.add_argument("--utt-offset", type=int, default=30)
    ap.add_argument("--max-dur", type=float, default=2.2)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--attack-registry", required=True)
    ap.add_argument("--source-speaker", required=True)
    ap.add_argument("--source-utt", type=int, required=True)
    ap.add_argument("--epss", default="0.02,0.05,0.08,0.1")
    ap.add_argument("--iters1", type=int, default=800)
    ap.add_argument("--iters2", type=int, default=400)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    for eps in [float(e) for e in args.epss.split(",") if e.strip()]:
        out_dir = f"{args.out_base}_eps{int(eps*100):03d}"
        if os.path.exists(os.path.join(out_dir, "results.csv")):
            # 已有结果则跳过已完成的句子由 run_experiment 内部续跑
            pass
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "run_experiment.py"),
               "--model-dir", args.model_dir, "--vctk", args.vctk,
               "--speakers", args.speakers, "--per-speaker", str(args.per_speaker),
               "--utt-offset", str(args.utt_offset), "--max-dur", str(args.max_dur),
               "--registry", args.registry, "--attack-registry", args.attack_registry,
               "--source-speaker", args.source_speaker, "--source-utt", str(args.source_utt),
               "--eps", str(eps), "--iters1", str(args.iters1), "--iters2", str(args.iters2),
               "--out-dir", out_dir, "--device", args.device, "--no-progress"]
        print("+", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)
    print("sweep done")


if __name__ == "__main__":
    main()
