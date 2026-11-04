"""Rebuild the quality comparison from archived text evidence (no models needed)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def load_csv(path):
    with path.open(encoding="utf-8") as stream:
        return {row["utterance"]: row for row in csv.DictReader(stream)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/evidence/2026-11-04/quality")
    args = parser.parse_args()
    root = args.evidence
    baseline_mos = load_csv(root / "baseline_main_vg/mos.csv")
    baseline_wer = load_csv(root / "baseline_main_vg/wer.csv")
    rows = []
    groups = {}
    for name, split in (("refine_dev_pcm16_eval", "development"),
                        ("refine_validation_pcm16_eval", "fixed_parameter_validation")):
        results = json.loads((root / name / "summary.json").read_text(encoding="utf-8"))["rows"]
        controls = {row["utterance"]: row for row in results if row["scale"] == 0}
        selected = [row for row in results if row["scale"] == 1]
        groups[split] = {"total": len(selected), "screened": 0, "joint_passes": 0}
        for row in selected:
            utterance = row["utterance"]
            speaker = utterance.split("_")[0]
            control = controls[utterance]
            clean_ok = control["asv_converted"] == speaker
            quality_ok = (row["mos_defended"] >= 3 and row["wer_defended"] <= .1
                          and row["asv_defended"] == speaker)
            joint = clean_ok and quality_ok and row["asv_converted"] != speaker
            if joint != row["screened_quality_identity_ok"]:
                raise ValueError(f"Stored quality gate disagrees with evidence: {utterance}")
            scores = row["converted_similarities"]
            wrong_best = max(value for name, value in scores.items() if name != speaker)
            rows.append({"utterance": utterance, "split": split,
                         "baseline_mos": float(baseline_mos[utterance]["mos_defended"]),
                         "refined_mos": row["mos_defended"],
                         "baseline_wer": float(baseline_wer[utterance]["wer_defended"]),
                         "refined_wer": row["wer_defended"],
                         "clean_vc_identity_ok": clean_ok,
                         "protected_identity": row["asv_defended"],
                         "converted_identity": row["asv_converted"],
                         "converted_wrong_minus_target_cosine": wrong_best - scores[speaker],
                         "mask_mean_excess_db": row["mask_excess_db_mean"],
                         "mask_exceed_fraction": row["mask_exceed_fraction"],
                         "quality_ok": quality_ok, "screened_joint_pass": joint})
            groups[split]["screened"] += int(clean_ok)
            groups[split]["joint_passes"] += int(joint)
    means = {key: sum(row[key] for row in rows) / len(rows) for key in
             ("baseline_mos", "refined_mos", "baseline_wer", "refined_wer")}
    result = {"protocol": "MOSNet>=3, Whisper target WER<=0.10, ECAPA preserved protected identity and changed converted identity; clean-VC screened",
              "paper_equivalent": False, "rows": rows, "means": means, "groups": groups,
              "inputs_sha256": {path.relative_to(root).as_posix(): file_sha256(path) for path in
                                 [root / "baseline_main_vg/mos.csv", root / "baseline_main_vg/wer.csv",
                                  root / "refine_dev_pcm16_eval/summary.json",
                                  root / "refine_validation_pcm16_eval/summary.json"]}}
    write_json(root / "comparison.json", result)
    lines = ["# Quality experiment: PCM16 export verification", "",
             "Small-sample engineering gates; not paper-equivalent scores.", "",
             "| Utterance | Baseline MOS | Refined MOS | Baseline WER | Refined WER | Converted identity | Screened joint pass |",
             "|---|---:|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append(f"| {row['utterance']} | {row['baseline_mos']:.3f} | {row['refined_mos']:.3f} | "
                     f"{row['baseline_wer']:.1%} | {row['refined_wer']:.1%} | {row['converted_identity']} | "
                     f"{row['screened_joint_pass']} |")
    lines += ["", f"Means across all five fixed utterances: {means}", "",
              f"Development: {groups['development']}; fixed-parameter validation: {groups['fixed_parameter_validation']}",
              "", "p229 is excluded because clean conversion already fails identity. All failures remain in quality averages.",
              "Full-context follow-up uses a different input protocol; see its separate results and controls.", ""]
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"means": means, "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
