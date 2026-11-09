"""L7：多进程并行实验（按说话人拆分给多个 worker，单卡并行提速）。

背景：单句攻击显存峰值仅 ~175MB（T4 实测），4060 8GB 也远未用满；
单进程攻击是"小算子串行延迟"型负载。把不同说话人分给 K 个独立进程
可并行占用 GPU 的不同执行队列，实测可缩短总时长（见 --report 输出）。

用法：
    python scripts/run_parallel.py --workers 3 --per-speaker 1 \
        --speakers p225,p226,p227,p228,p229,p230 \
        --model-dir checkpoints/adain-vc-official ... --out-root outputs/parallel_test
产物：<out-root>/workerN/（每个 worker 一个实验目录）+ <out-root>/parallel_summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--speakers", required=True)
    ap.add_argument("--per-speaker", type=int, default=1)
    ap.add_argument("--method", default="vg")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--iters1", type=int, default=300)
    ap.add_argument("--iters2", type=int, default=150)
    ap.add_argument("--model-dir", default="checkpoints/adain-vc-official")
    ap.add_argument("--vctk", default="data/VCTK22")
    ap.add_argument("--registry", default="outputs/registry.npz")
    ap.add_argument("--attack-registry", default="outputs/registry_vc.npz")
    ap.add_argument("--source-speaker", default="p226")
    ap.add_argument("--source-utt", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-root", default="outputs/parallel_test")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    if not speakers:
        sys.exit("需要 --speakers")
    root = ROOT / args.out_root
    root.mkdir(parents=True, exist_ok=True)

    # 轮流分配，均衡负载
    groups = [speakers[i::args.workers] for i in range(args.workers)]
    groups = [g for g in groups if g]
    workers = []
    t0 = time.perf_counter()
    for i, group in enumerate(groups):
        out_dir = root / f"worker{i}"
        cmd = [sys.executable, "scripts/run_experiment.py",
               "--model-dir", args.model_dir, "--vctk", args.vctk,
               "--speakers", ",".join(group), "--per-speaker", str(args.per_speaker),
               "--utt-offset", "30", "--max-dur", "2.5",
               "--registry", args.registry, "--attack-registry", args.attack_registry,
               "--source-speaker", args.source_speaker, "--source-utt", str(args.source_utt),
               "--method", args.method, "--eps", str(args.eps),
               "--iters1", str(args.iters1), "--iters2", str(args.iters2),
               "--seed", str(args.seed), "--out-dir", str(out_dir),
               "--device", args.device, "--no-progress"]
        log = open(root / f"worker{i}.log", "w", encoding="utf-8")
        print(f"+ worker{i} speakers={group}", flush=True)
        workers.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log))

    codes = [p.wait() for p, _ in workers]
    for _, log in workers:
        log.close()
    elapsed = time.perf_counter() - t0

    summary = {"workers": len(groups), "groups": groups, "elapsed_s": round(elapsed, 1),
               "return_codes": codes, "ok": all(c == 0 for c in codes)}
    (root / "parallel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if not summary["ok"]:
        sys.exit("部分 worker 失败，见 <out-root>/workerN.log")


if __name__ == "__main__":
    main()
