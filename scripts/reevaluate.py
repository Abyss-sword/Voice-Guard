"""离线重评：对已跑完的 utterance 目录统一计算 MCD + ASV，输出 results.csv。

与 run_experiment.py 解耦：攻击只跑一次，评估模型（ECAPA / Es 线性探头）可反复换。

用法：
    python scripts/reevaluate.py --exp-dir outputs/exp01 \
        --model-dir checkpoints/adain-vc-official \
        --registry outputs/registry.npz \
        [--probe outputs/asv_probe.npz] \
        --out outputs/exp01/results_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vguard.asv import embed_wav, load_ecapa  # noqa: E402
from vguard.metrics.mcd import mcd_mel  # noqa: E402
from vguard.speaker import ProbeASV  # noqa: E402
from vguard.vc.adainvc import load_adainvc, speaker_embed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--registry", required=True, help="ECAPA 注册库 npz")
    ap.add_argument("--probe", default="", help="Es 线性探头 npz（可选）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, config, attr = load_adainvc(args.model_dir, device=args.device)
    pp = config["preprocess"]
    fk = {k: pp[k] for k in ("sample_rate", "preemph", "n_fft", "hop_length",
                             "win_length", "n_mels", "ref_db", "max_db", "top_db")}

    reg = np.load(args.registry, allow_pickle=True)
    names = list(reg["names"])
    ecapa = load_ecapa(savedir="checkpoints/ecapa-local")

    probe = None
    if args.probe and os.path.exists(args.probe):
        p = np.load(args.probe, allow_pickle=True)

        def es_embed(wav, sr):
            w = wav.astype(np.float32)
            if sr != pp["sample_rate"]:
                import librosa
                w = librosa.resample(w, orig_sr=sr, target_sr=pp["sample_rate"])
            return speaker_embed(model, config, attr, w, args.device)

        probe = ProbeASV(list(p["names"]), p["probe_weight"], p["probe_bias"], es_embed)
        probe.to(args.device)

    rows = []
    for d in sorted(os.listdir(args.exp_dir)):
        sub = os.path.join(args.exp_dir, d)
        if not os.path.isdir(sub) or not os.path.exists(os.path.join(sub, "defended.wav")):
            continue
        target = d.split("_")[0]
        row = {"utterance": d}

        w_orig, sr = sf.read(os.path.join(sub, "original_trimmed.wav"))
        w_def, _ = sf.read(os.path.join(sub, "defended.wav"))
        row["mcd"] = round(mcd_mel(w_orig.astype(np.float32), w_def.astype(np.float32),
                                   fk, attr=None, device=args.device), 3)

        # ECAPA
        for key, f in [("defended", "defended.wav"), ("conv_clean", "conv_from_clean.wav"),
                       ("conv_def", "conv_from_defended.wav")]:
            fp = os.path.join(sub, f)
            if not os.path.exists(fp):
                continue
            w, sr = sf.read(fp)
            e = embed_wav(ecapa, w.astype(np.float32), sr)
            sims = np.asarray([float(np.dot(e, x) / (np.linalg.norm(e) * np.linalg.norm(x) + 1e-9))
                               for x in reg["embeddings"]])
            pred = names[int(np.argmax(sims))]
            row[f"ec_{key}"] = pred
            row[f"ec_simX_{key}"] = round(float(sims[names.index(target)]), 3)
        row["ec_screen_ok"] = row.get("ec_conv_clean") == target
        row["ec_defense_ok"] = (row.get("ec_defended") == target) and (row.get("ec_conv_def") != target)

        # 探头
        if probe is not None:
            for key, f in [("defended", "defended.wav"), ("conv_clean", "conv_from_clean.wav"),
                           ("conv_def", "conv_from_defended.wav")]:
                fp = os.path.join(sub, f)
                if not os.path.exists(fp):
                    continue
                w, sr = sf.read(fp)
                pred, score = probe.predict(w, sr)
                row[f"pr_{key}"] = pred
                row[f"pr_score_{key}"] = round(score, 2)
            row["pr_screen_ok"] = row.get("pr_conv_clean") == target
            row["pr_defense_ok"] = (row.get("pr_defended") == target) and (row.get("pr_conv_def") != target)

        rows.append(row)
        print(row)

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(rows)
        print(f"saved {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
