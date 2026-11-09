"""Shared closed-set identity criteria and small-sample uncertainty reporting."""
from __future__ import annotations

import math


def identity_verdict(asv, target):
    """Missing predictions never count as defense; clean VC is a separate screen."""
    predicted = {name: asv.get(name, {}).get("pred") for name in
                 ("original", "defended", "conv_clean", "conv_def")}
    complete = not asv.get("error") and all(predicted[name] for name in
                                          ("defended", "conv_clean", "conv_def"))
    screened = bool(complete and predicted["conv_clean"] == target)
    identity_ok = bool(complete and predicted["defended"] == target
                       and predicted["conv_def"] != target)
    return {"asv_complete": bool(complete), "clean_vc_identity_ok": screened,
            "identity_ok": identity_ok, "screened_identity_ok": screened and identity_ok,
            "protocol": "ECAPA closed-set identity only; not a quality or imperceptibility claim"}


def quality_pass(mos_score, word_error, original_identity_preserved):
    return bool(original_identity_preserved and math.isfinite(mos_score)
                and math.isfinite(word_error) and mos_score >= 3 and word_error <= .10)


def wilson_interval(successes, total, z=1.959963984540054):
    """95% binomial interval; descriptive only when utterances share speakers."""
    if not 0 <= successes <= total:
        raise ValueError("Need 0 <= successes <= total")
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return [max(0., center-half), min(1., center+half)]


def summarize_pairs(rows):
    """Rows represent paired clean/protected trials, each with its own clean control."""
    eligible = [row for row in rows if row["clean_vc_identity_ok"]]
    successes = sum(bool(row["screened_quality_identity_ok"]) for row in eligible)
    return {"total": len(rows), "screened": len(eligible), "joint_successes": successes,
            "joint_rate": successes/len(eligible) if eligible else None,
            "wilson95_descriptive": wilson_interval(successes, len(eligible)),
            "mean_mos": sum(row["mos_defended"] for row in rows)/len(rows) if rows else None,
            "macro_wer": sum(row["wer_defended"] for row in rows)/len(rows) if rows else None,
            "uncertainty_note": "Repeated utterances/sources within a speaker are correlated; interval is not a speaker-generalization confidence claim"}
