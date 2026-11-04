"""Summarize the bounded ECS campaign without treating partial runs as complete."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import completed_row, summarize, write_json

EXPECTED = [f"main_{method}" for method in ("vg", "vg-nop", "emb", "fb", "e2e")]
EXPECTED += [f"eps_{eps}" for eps in (".02", ".05", ".08", ".10")]


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def average(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    values = [value for value in values if math.isfinite(value)]
    return round(sum(values) / len(values), 4) if values else None


def finite_fields(rows, prefix):
    try:
        return all(math.isfinite(float(row[f"{prefix}_{key}"])) for row in rows
                   for key in ("original_trimmed", "defended", "conv_from_clean", "conv_from_defended"))
    except (KeyError, ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/cloud")
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    groups = []
    for name in EXPECTED:
        directory = args.root / name
        rows = read_csv(directory / "results.csv")
        mos = read_csv(directory / "mos.csv")
        wer = read_csv(directory / "wer.csv")
        utterances = {row["utterance"] for row in rows}
        stats = summarize(rows)
        complete = (len(rows) == len(utterances) == stats["evaluated"] == 5 and
                    all(completed_row(directory / utterance) is not None for utterance in utterances) and
                    len(mos) == len(wer) == 5 and finite_fields(mos, "mos") and finite_fields(wer, "wer") and
                    {row["utterance"] for row in mos} == utterances and
                    {row["utterance"] for row in wer} == utterances and
                    all(row.get("reference_mode") == "transcript" for row in wer))
        groups.append({"name": name, "complete": complete, **stats,
                       "mean_mcd": average(rows, "mcd"),
                       "mean_mos_defended": average(mos, "mos_defended"),
                       "mean_wer_defended": average(wer, "wer_defended"),
                       "total_sample_seconds": round(sum(float(row.get("elapsed_s", 0)) for row in rows), 2)})
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "exit_code": args.exit_code,
              "complete": all(group["complete"] for group in groups) and args.exit_code in (None, 0),
              "groups": groups, "protocol": "5 speakers x 1 utterance; 20-speaker enrollment; ECAPA identity screening",
              "limitations": ["Not an exact replication of the paper's models, dataset or metrics",
                              "MOS is a third-party predictor; WER uses VCTK transcripts",
                              "Identity success is not joint quality success",
                              "main and eps run concurrently; per-sample times include contention"]}
    args.root.mkdir(parents=True, exist_ok=True)
    write_json(args.root / "campaign-summary.json", report)
    lines = ["# Huawei ECS campaign", "", f"Generated UTC: {report['generated_utc']}", "",
             f"Complete: {report['complete']}", "", report["protocol"], "",
             "| Group | Evaluated | Clean-VC pass | Screened success | MCD | MOS protected | WER protected |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        lines.append(f"| {group['name']} | {group['evaluated']}/5 | {group['screened']}/{group['evaluated']} | "
                     f"{group['screened_successes']}/{group['screened']} | {group['mean_mcd']} | "
                     f"{group['mean_mos_defended']} | {group['mean_wer_defended']} |")
    lines += ["", *[f"- {item}" for item in report["limitations"]], ""]
    (args.root / "campaign-summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Campaign complete: {report['complete']}")


if __name__ == "__main__":
    main()
