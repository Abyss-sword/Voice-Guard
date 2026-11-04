"""Atomic experiment records and summaries, usable without torch/model weights."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


ARTIFACTS = ("defended.wav", "original_trimmed.wav", "source.wav",
             "conv_from_clean.wav", "conv_from_defended.wav")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utterance_seed(seed, utterance):
    return int.from_bytes(hashlib.sha256(f"{seed}:{utterance}".encode()).digest()[:4], "little")


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def completed_row(directory):
    directory = Path(directory)
    record = directory / "result.json"
    if record.is_file() and all((directory / f).is_file() for f in ARTIFACTS):
        row = json.loads(record.read_text(encoding="utf-8"))
        if not row.get("asv_error"):
            return row
    return None


def as_bool(value):
    return str(value).lower() in ("true", "1")


def summarize(rows):
    # Missing ASV results do not count as evaluated samples or successes.
    evaluated = [r for r in rows if r.get("defense_ok") not in (None, "") and not r.get("asv_error")]
    screened = [r for r in evaluated if as_bool(r.get("conv_clean_hits_X"))]
    ok = sum(as_bool(r.get("defense_ok")) for r in evaluated)
    screened_ok = sum(as_bool(r.get("defense_ok")) for r in screened)
    return {
        "total": len(rows), "evaluated": len(evaluated), "asv_missing": len(rows) - len(evaluated),
        "unfiltered_successes": ok,
        "unfiltered_rate": ok / len(evaluated) if evaluated else None,
        "screened": len(screened),
        "screen_pass_rate": len(screened) / len(evaluated) if evaluated else None,
        "screened_successes": screened_ok,
        "screened_rate": screened_ok / len(screened) if screened else None,
        "protocol": "ECAPA identity only; quality thresholds are not included",
    }
