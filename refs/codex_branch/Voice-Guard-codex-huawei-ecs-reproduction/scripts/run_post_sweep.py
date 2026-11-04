"""后扫描自动执行器：等 ε 扫描完成后，串行跑加分项实验（T4）。

流程：
  1. 轮询等待 outputs/exp03_eps010/results.csv 达到 5 行（ε 扫描完成）
  2. channel_sweep.py 扫 exp03_eps005（VG@ε=0.05 的真实信道防御衰减）
  3. run_experiment.py --eot --eps 0.05 跑 exp04（VG+EOT，与 exp03 同 5 句）
  4. channel_sweep.py 扫 exp04
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def wait_for(path: str, rows: int, timeout_h: float = 6.0, poll_s: int = 120):
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    n = len(list(csv.DictReader(f)))
                if n >= rows:
                    print(f"[ok] {path} has {n} rows")
                    return
            except Exception:  # noqa: BLE001
                pass
        print(f"[wait] {path}: not ready ({int(time.time()-t0)}s)", flush=True)
        time.sleep(poll_s)
    raise TimeoutError(f"{path} not ready after {timeout_h}h")


def run(cmd, cwd=ROOT):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "checkpoints", "adain-vc-official"))
    ap.add_argument("--vctk", default=os.path.join(ROOT, "data", "VCTK22"))
    ap.add_argument("--speakers", default="p225,p226,p227,p228,p229")
    ap.add_argument("--registry", default=os.path.join(ROOT, "outputs", "registry.npz"))
    ap.add_argument("--attack-registry", default=os.path.join(ROOT, "outputs", "registry_vc.npz"))
    ap.add_argument("--probe", default=os.path.join(ROOT, "outputs", "asv_probe.npz"))
    ap.add_argument("--source-speaker", default="p226")
    ap.add_argument("--source-utt", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--iters1", type=int, default=800)
    ap.add_argument("--iters2", type=int, default=400)
    args = ap.parse_args()

    py = sys.executable
    eps010 = os.path.join(ROOT, "outputs", "exp03_eps010", "results.csv")

    # 1) 等 ε 扫描完成
    print("[1/4] waiting for eps sweep...", flush=True)
    wait_for(eps010, rows=5, timeout_h=6.0)

    # 2) 信道扫描 VG（已有结果则跳过，支持断点续跑）
    vg_chan = os.path.join(ROOT, "outputs", "exp03_eps005", "channel_results.csv")
    if os.path.exists(vg_chan):
        print("[2/4] VG channel sweep already done, skip", flush=True)
    else:
        print("[2/4] channel sweep on VG (exp03_eps005)", flush=True)
        run([py, os.path.join(ROOT, "scripts", "channel_sweep.py"),
             "--exp-dir", os.path.join(ROOT, "outputs", "exp03_eps005"),
             "--model-dir", args.model_dir, "--registry", args.registry, "--probe", args.probe,
             "--out", vg_chan, "--device", "cuda"])

    # 3) EOT 攻击（已有完整结果则跳过）
    eot_csv = os.path.join(ROOT, "outputs", "exp04", "results.csv")
    if os.path.exists(eot_csv):
        with open(eot_csv, encoding="utf-8") as f:
            n_eot = len(list(csv.DictReader(f)))
        if n_eot >= 5:
            print("[3/4] EOT batch already done, skip", flush=True)
        else:
            run([py, os.path.join(ROOT, "scripts", "run_experiment.py"),
                 "--model-dir", args.model_dir, "--vctk", args.vctk, "--speakers", args.speakers,
                 "--per-speaker", "1", "--utt-offset", "30", "--max-dur", "2.2",
                 "--registry", args.registry, "--attack-registry", args.attack_registry,
                 "--source-speaker", args.source_speaker, "--source-utt", str(args.source_utt),
                 "--eps", str(args.eps), "--iters1", str(args.iters1), "--iters2", str(args.iters2),
                 "--eot", "--out-dir", os.path.join(ROOT, "outputs", "exp04"),
                 "--device", "cuda", "--no-progress"])
    else:
        print("[3/4] VG+EOT batch (exp04)", flush=True)
        run([py, os.path.join(ROOT, "scripts", "run_experiment.py"),
             "--model-dir", args.model_dir, "--vctk", args.vctk, "--speakers", args.speakers,
             "--per-speaker", "1", "--utt-offset", "30", "--max-dur", "2.2",
             "--registry", args.registry, "--attack-registry", args.attack_registry,
             "--source-speaker", args.source_speaker, "--source-utt", str(args.source_utt),
             "--eps", str(args.eps), "--iters1", str(args.iters1), "--iters2", str(args.iters2),
             "--eot", "--out-dir", os.path.join(ROOT, "outputs", "exp04"),
             "--device", "cuda", "--no-progress"])

    # 4) 信道扫描 EOT
    print("[4/4] channel sweep on VG+EOT (exp04)", flush=True)
    run([py, os.path.join(ROOT, "scripts", "channel_sweep.py"),
         "--exp-dir", os.path.join(ROOT, "outputs", "exp04"),
         "--model-dir", args.model_dir, "--registry", args.registry, "--probe", args.probe,
         "--out", os.path.join(ROOT, "outputs", "exp04", "channel_results.csv"),
         "--device", "cuda"])
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
