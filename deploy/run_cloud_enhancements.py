"""云端增强实验一键驱动（在 ECS 的 /root/voiceguard 下运行）。

阶段：
  seed      多种子方差：seed 2027/2028 × 5 方法（每方法 5 句，3000/1500 步）
  expand    扩展样本：每说话人 3 句 × 3 方法（vg / vg-nop / emb）
  asr       ASR 阈值映射：对已有云端结果跑 Whisper WER 分布
  channels  信道扫描：对 main_vg 与 seed2027_main_vg 跑 6 信道
  staged    两阶段恢复验证（调用既有 deploy/run_staged_recovery.sh，若存在）

用法（ECS）：
    .venv/bin/python deploy/run_cloud_enhancements.py --dry-run
    .venv/bin/python deploy/run_cloud_enhancements.py --stages seed,expand
    .venv/bin/python deploy/run_cloud_enhancements.py --max-hours 6

说明：--max-hours 只限制本脚本运行时间，**不会停止 ECS 计费**；结束后请关机。
断点续跑：已有 `results.csv` 的目录自动跳过。
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs" / "enhancements"
SPEAKERS = "p225,p226,p227,p228,p229"
COMMON = [
    "--model-dir", "checkpoints/adain-vc-official", "--vctk", "data/VCTK22",
    "--registry", "outputs/registry.npz", "--attack-registry", "outputs/registry_vc.npz",
    "--utt-offset", "30", "--max-dur", "2.5",
    "--source-speaker", "p226", "--source-utt", "2",
    "--device", "cuda", "--no-progress",
    "--iters1", "3000", "--iters2", "1500",
]


def say(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    return line


def has_results(d: Path) -> bool:
    return (d / "results.csv").exists() and (d / "results.csv").stat().st_size > 0


class Runner:
    def __init__(self, max_hours: float, dry: bool):
        self.deadline = time.monotonic() + max_hours * 3600
        self.dry = dry
        self.log_path = OUT / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
        if not dry:
            OUT.mkdir(parents=True, exist_ok=True)
        self.log_lines: list[str] = []

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def run(self, name: str, cmd: list[str], stage_hours: float = 3.0):
        budget = min(self.remaining(), stage_hours * 3600)
        line = f"+ [{name}] {' '.join(cmd)}"
        print(line, flush=True)
        self.log_lines.append(line)
        if self.dry:
            return
        if budget <= 60:
            msg = say("达到 --max-hours 上限，停止")
            self.log_lines.append(msg)
            raise SystemExit(0)
        t0 = time.monotonic()
        try:
            subprocess.run(cmd, cwd=ROOT, timeout=budget, check=True)
            msg = say(f"{name} 完成（{time.monotonic()-t0:.0f}s）")
        except subprocess.TimeoutExpired:
            msg = say(f"{name} 超时（>{budget/3600:.1f}h），继续下一项")
        except subprocess.CalledProcessError as e:
            msg = say(f"{name} 失败（exit {e.returncode}），继续下一项")
        self.log_lines.append(msg)

    def flush(self):
        if not self.dry:
            self.log_path.write_text("\n".join(self.log_lines), encoding="utf-8")


def experiment_cmd(out: Path, method: str, per_speaker: int, seed: int) -> list[str]:
    return [PY, "scripts/run_experiment.py", *COMMON,
            "--speakers", SPEAKERS, "--per-speaker", str(per_speaker),
            "--method", method, "--eps", ".05", "--seed", str(seed),
            "--out-dir", str(out.relative_to(ROOT))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="seed,expand,asr,channels")
    ap.add_argument("--seeds", default="2027 2028")
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stage-hours", type=float, default=3.0, help="单个阶段的时间上限（小时）")
    args = ap.parse_args()
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    seeds = [int(s) for s in args.seeds.split()]

    r = Runner(args.max_hours, args.dry_run)
    try:
        # ---------------- seed：多种子 ----------------
        if "seed" in stages:
            for seed in seeds:
                for m in ("vg", "vg-nop", "emb", "fb", "e2e"):
                    out = OUT / f"seed{seed}_main_{m}"
                    if has_results(out):
                        say(f"[skip] {out.relative_to(ROOT)} 已有结果")
                        continue
                    r.run(f"seed{seed}/{m}", experiment_cmd(out, m, 1, seed), args.stage_hours)

        # ---------------- expand：扩展样本 ----------------
        if "expand" in stages:
            for m in ("vg", "vg-nop", "emb"):
                out = OUT / f"expand3_main_{m}"
                if has_results(out):
                    say(f"[skip] {out.relative_to(ROOT)} 已有结果")
                    continue
                r.run(f"expand/{m}", experiment_cmd(out, m, 3, 2026), args.stage_hours)

        # ---------------- asr：ASR 阈值映射 ----------------
        if "asr" in stages:
            dirs = ["outputs/cloud/main_vg", "outputs/cloud/main_vg-nop",
                    "outputs/cloud/main_emb", "outputs/enhancements/seed2027_main_vg"]
            existing = [d for d in dirs if (ROOT / d).is_dir()]
            if existing:
                r.run("asr-map", [PY, "tools/asr_threshold_map.py",
                                  "--exp-dirs", ",".join(existing),
                                  "--vctk", "data/VCTK22",
                                  "--source-speaker", "p226", "--source-utt", "2",
                                  "--device", "cuda",
                                  "--out", str((OUT / "asr_threshold_map.json").relative_to(ROOT))],
                      1.0)
            else:
                say("[skip] asr：没有可用的云端实验目录")

        # ---------------- channels：信道扫描 ----------------
        if "channels" in stages:
            for d in ("outputs/cloud/main_vg", "outputs/enhancements/seed2027_main_vg"):
                if not (ROOT / d).is_dir():
                    say(f"[skip] channels：{d} 不存在")
                    continue
                r.run(f"channels:{Path(d).name}",
                      [PY, "scripts/channel_sweep.py",
                       "--exp-dir", d,
                       "--model-dir", "checkpoints/adain-vc-official",
                       "--registry", "outputs/registry.npz",
                       "--probe", "outputs/asr_probe.npz",
                       "--out", f"{d}/channel_results.csv"],
                      1.0)

        # ---------------- staged：两阶段恢复 ----------------
        if "staged" in stages:
            script = ROOT / "deploy" / "run_staged_recovery.sh"
            if script.exists():
                r.run("staged", ["bash", str(script)], 4.0)
            else:
                say("[skip] staged：deploy/run_staged_recovery.sh 不存在")
    finally:
        r.flush()

    say("全部阶段结束")
    say(f"日志：{r.log_path.relative_to(ROOT)}")
    say("⚠️ ECS 仍在计费，请到控制台关机或释放实例")
    say("回传：打包 outputs/cloud 与 outputs/enhancements 下载，提交到 docs/evidence/ 或发给负责人合并")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
