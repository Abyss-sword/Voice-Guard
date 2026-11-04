"""Staged cloud reproduction. --dry-run prints commands without loading models."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.preflight import SPEAKERS
from vguard.run_state import write_json


def run_bounded(command, seconds):
    process = subprocess.Popen(command, start_new_session=(os.name != "nt"))
    try:
        code = process.wait(timeout=seconds)
        if code:
            raise subprocess.CalledProcessError(code, command)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # Also stop grandchildren such as curl or a model evaluator.
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise


def commands(stage, device, seed):
    py = sys.executable
    registry = ["--vctk", "data/VCTK22", "--speakers", ",".join(SPEAKERS), "--per-speaker", "30", "--device", device]
    if stage == "models":
        return [[py, "tools/download_models.py", "models"]]
    if stage == "data":
        return [[py, "tools/download_models.py", "vctk", "--extract"]]
    if stage == "data-subset":
        return [[py, "tools/download_vctk_subset.py"]]
    if stage == "registry":
        return [[py, "scripts/build_registry.py", "--mode", "vc", "--model-dir", "checkpoints/adain-vc-official",
                 "--out", "outputs/registry_vc.npz", *registry],
                [py, "scripts/build_registry.py", "--mode", "ecapa", "--out", "outputs/registry.npz", *registry]]
    common = ["--model-dir", "checkpoints/adain-vc-official", "--vctk", "data/VCTK22",
              "--registry", "outputs/registry.npz", "--attack-registry", "outputs/registry_vc.npz",
              "--per-speaker", "1", "--utt-offset", "30", "--max-dur", "2.5",
              "--source-speaker", "p226", "--source-utt", "2", "--device", device,
              "--seed", str(seed), "--no-progress"]
    specs = []
    if stage in ("smoke", "benchmark"):
        it1, it2 = (60, 30) if stage == "smoke" else (3000, 1500)
        specs = [(stage, "vg", ".05", "p225", it1, it2)]
    elif stage == "baseline-smoke":
        specs = [(f"smoke_{method}", method, ".05", "p225", 2, 1)
                 for method in ("emb", "fb", "e2e")]
    elif stage == "main":
        specs = [(f"main_{m}", m, ".05", "p225,p226,p227,p228,p229", 3000, 1500)
                 for m in ("vg", "vg-nop", "emb", "fb", "e2e")]
    elif stage == "eps":
        specs = [(f"eps_{eps}", "vg", eps, "p225,p226,p227,p228,p229", 3000, 1500)
                 for eps in (".02", ".05", ".08", ".10")]
    return [[py, "scripts/run_experiment.py", *common, "--speakers", speakers,
             "--method", method, "--eps", eps, "--iters1", str(it1), "--iters2", str(it2),
             "--out-dir", f"outputs/cloud/{name}"] for name, method, eps, speakers, it1, it2 in specs]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["models", "data", "data-subset", "registry", "smoke", "baseline-smoke", "benchmark", "main", "eps"])
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-hours", type=float, default=2, help="stop child processes after this wall-time budget")
    args = ap.parse_args()
    if args.max_hours <= 0:
        ap.error("max-hours must be positive")
    os.chdir(ROOT)
    plan = commands(args.stage, args.device, args.seed)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    started = time.monotonic()
    run = ROOT / "outputs/cloud" / f"run_{args.stage}_{time.time_ns()}.json"
    record = {"stage": args.stage, "max_hours": args.max_hours, "commands": plan, "state": "running"}
    write_json(run, record)
    try:
        for command in plan:
            remaining = args.max_hours * 3600 - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, args.max_hours * 3600)
            print("+", subprocess.list2cmdline(command), flush=True)
            run_bounded(command, remaining)
        record["state"] = "complete"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        record["state"] = "interrupted_or_failed"
        record["error"] = str(exc)
        raise
    finally:
        record["elapsed_s"] = round(time.monotonic() - started, 1)
        write_json(run, record)


if __name__ == "__main__":
    main()
