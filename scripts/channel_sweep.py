"""信道鲁棒性扫描：对已跑好的攻击产物施加真实信道 → VC 转换 → ASV，看防御衰减。

用法：
    python scripts/channel_sweep.py --exp-dir outputs/exp02 \
        --model-dir checkpoints/adain-vc-official \
        --registry outputs/registry.npz --probe outputs/asv_probe.npz \
        --channels mp3-128k,mp3-32k,aac-64k,g711,resample8k,none \
        --out outputs/exp02/channel_results.csv

对每个 utterance 目录的 defended.wav：
    channel(defended) -> 存 channeled_def.wav -> VC 转换（声纹=channeled_def）-> ASV(探头/ECAPA)
    防御成功率 = ASV(defended_after_channel)==X 且 ASV(conv)!=X 的比例（按信道分列）。
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
from vguard.channel import codec_channel  # noqa: E402
from vguard.speaker import ProbeASV  # noqa: E402
from vguard.vc.adainvc import convert as vc_convert  # noqa: E402
from vguard.vc.adainvc import load_adainvc, speaker_embed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--probe", default="")
    ap.add_argument("--channels", default="none,mp3-128k,mp3-32k,aac-64k,g711,resample8k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, config, attr = load_adainvc(args.model_dir, device=args.device)
    pp = config["preprocess"]
    sr = pp["sample_rate"]

    reg = np.load(args.registry, allow_pickle=True)
    names = list(reg["names"])
    ecapa = load_ecapa(savedir="checkpoints/ecapa-local")

    probe = None
    if args.probe and os.path.exists(args.probe):
        p = np.load(args.probe, allow_pickle=True)

        def es_embed(wav, sr_):
            w = wav.astype(np.float32)
            if sr_ != sr:
                import librosa
                w = librosa.resample(w, orig_sr=sr_, target_sr=sr)
            return speaker_embed(model, config, attr, w, args.device)

        probe = ProbeASV(list(p["names"]), p["probe_weight"], p["probe_bias"], es_embed)
        probe.to(args.device)

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    rows = []

    for d in sorted(os.listdir(args.exp_dir)):
        sub = os.path.join(args.exp_dir, d)
        def_path = os.path.join(sub, "defended.wav")
        if not os.path.isdir(sub) or not os.path.exists(def_path):
            continue
        target = d.split("_")[0]
        w_def, _ = sf.read(def_path)
        w_def = w_def.astype(np.float32)
        row = {"utterance": d}

        for ch in channels:
            if ch == "none":
                w_ch = w_def
            else:
                w_ch = codec_channel(w_def, sr, ch)
            ch_path = os.path.join(sub, f"defended_{ch}.wav")
            sf.write(ch_path, w_ch, sr)
            conv_path = os.path.join(sub, f"conv_def_{ch}.wav")
            content_src = os.path.join(sub, "source.wav")
            vc_convert(model, config, attr, content_src, ch_path, conv_path, args.device)
            if probe is not None:
                pd_, s1 = probe.predict(w_ch, sr)
                wc, _ = sf.read(conv_path)
                pc_, s2 = probe.predict(wc, sr)
                row[f"pr_def_{ch}"] = pd_
                row[f"pr_conv_{ch}"] = pc_
                row[f"pr_ok_{ch}"] = (pd_ == target) and (pc_ != target)
            e = embed_wav(ecapa, w_ch, sr)
            sims = np.asarray([float(np.dot(e, x) / (np.linalg.norm(e) * np.linalg.norm(x) + 1e-9))
                               for x in reg["embeddings"]])
            pd2 = names[int(np.argmax(sims))]
            wc2, _ = sf.read(conv_path)
            e2 = embed_wav(ecapa, wc2.astype(np.float32), sr)
            sims2 = np.asarray([float(np.dot(e2, x) / (np.linalg.norm(e2) * np.linalg.norm(x) + 1e-9))
                                for x in reg["embeddings"]])
            pc2 = names[int(np.argmax(sims2))]
            row[f"ec_ok_{ch}"] = (pd2 == target) and (pc2 != target)

        rows.append(row)
        print(row)

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(rows)
        print("saved:", args.out)
        # 汇总：各信道防御率
        print("=== per-channel defense rate ===")
        for ch in channels:
            n = len(rows)
            ok = sum(1 for r in rows if r.get(f"pr_ok_{ch}") is True)
            ok_ec = sum(1 for r in rows if r.get(f"ec_ok_{ch}") is True)
            print(f"{ch:12s} probe {100*ok/max(n,1):5.1f}%   ecapa {100*ok_ec/max(n,1):5.1f}%")


if __name__ == "__main__":
    main()
