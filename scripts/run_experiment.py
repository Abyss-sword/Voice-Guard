"""Batch reproduction: protect -> convert -> metrics over a VCTK sample, with resume.

Usage:
    python scripts/run_experiment.py --model-dir checkpoints/adain-vc-official \
        --vctk data/VCTK22 --speakers p225,p226,...,p234 --per-speaker 5 --first-utt 1 \
        --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz \
        --source-speaker p226 --source-utt 2 \
        --eps 0.1 --out-dir outputs/exp01 --device cuda

For each (speaker X, utterance i):
    x_hat  = X's utterance i (audio to protect)
    y_hat  = least-likely speaker Y's first utterance (perturbation direction)
    t_hat  = content source (--source-speaker / --source-utt)
    outputs: defended; conversions F(t,x) and F(t,x'); ASV + MCD; defense flag.
Resume: utterance dirs that already contain defended.wav are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vguard.attack import voice_guard_attack
from vguard.asv import embed_wav, load_ecapa
from vguard.masking import file2wav_mask
from vguard.metrics.mcd import mcd_mel
from vguard.speaker import SpeakerRegistry
from vguard.vc.adainvc import convert as vc_convert
from vguard.vc.adainvc import load_adainvc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--vctk", required=True)
    ap.add_argument("--speakers", required=True)
    ap.add_argument("--per-speaker", type=int, default=5)
    ap.add_argument("--utt-offset", type=int, default=30,
                    help="test utterances start at this sorted-file index (enrollment uses the first 30)")
    ap.add_argument("--registry", required=True, help="eval registry (ECAPA)")
    ap.add_argument("--attack-registry", required=True, help="attack registry (VC Es embeddings)")
    ap.add_argument("--asv-probe", default="", help="optional Es linear probe npz (used as attack flag)")
    ap.add_argument("--max-dur", type=float, default=0.0,
                    help="only use test utterances shorter than this (s); 0 = no filter")
    ap.add_argument("--source-speaker", required=True)
    ap.add_argument("--source-utt", type=int, required=True)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--method", choices=["vg", "vg-nop", "emb", "fb", "e2e"], default="vg")
    ap.add_argument("--eot", action="store_true", help="VG+EOT：阶段2用可微信道做稳健训练")
    ap.add_argument("--iters1", type=int, default=3000)
    ap.add_argument("--iters2", type=int, default=1500)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    model, config, attr = load_adainvc(args.model_dir, device=device)
    pp = config["preprocess"]
    feature_kwargs = {k: pp[k] for k in
                      ("sample_rate", "preemph", "n_fft", "hop_length",
                       "win_length", "n_mels", "ref_db", "max_db", "top_db")}
    sr = pp["sample_rate"]

    reg_npz = np.load(args.registry, allow_pickle=True)
    eval_reg = SpeakerRegistry(reg_npz["embeddings"], list(reg_npz["names"])).to(device)
    atk_npz = np.load(args.attack_registry, allow_pickle=True)
    atk_reg = SpeakerRegistry(atk_npz["embeddings"], list(atk_npz["names"])).to(device)

    probe_asv = None
    if args.asv_probe and os.path.exists(args.asv_probe):
        from vguard.speaker import ProbeASV
        from vguard.vc.adainvc import speaker_embed
        p = np.load(args.asv_probe, allow_pickle=True)

        def es_embed_of(wav, sr_):
            w = wav.astype(np.float32)
            if sr_ != sr:
                import librosa
                w = librosa.resample(w, orig_sr=sr_, target_sr=sr)
            return speaker_embed(model, config, attr, w, device)

        probe_asv = ProbeASV(list(p["names"]), p["probe_weight"], p["probe_bias"],
                             es_embed_of).to(device)

        def make_probe_flag(target_name):
            import torch.nn.functional as F

            def fn(emb):
                e = F.normalize(emb.reshape(-1), dim=0)
                logits = probe_asv.w @ e + probe_asv.b
                pred = probe_asv.names[int(torch.argmax(logits))]
                return pred != target_name
            return fn

        print(f"[probe] loaded {args.asv_probe} for attack flag")

    src_path = os.path.join(args.vctk, args.source_speaker,
                            f"{args.source_speaker}_{args.source_utt:03d}.wav")
    assert os.path.exists(src_path), src_path
    src_wav, _ = sf.read(src_path)
    src_wav = src_wav.astype(np.float32)

    def first_wav(d):
        files = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        assert files, d
        return os.path.join(d, files[0])

    ll_map = {}
    for x in speakers:
        e = atk_reg.embeddings[atk_reg.names.index(x)]
        y, _ = atk_reg.least_likely(x, e)
        ll_map[x] = y
        print(f"least-likely: {x} -> {y}")

    ecapa = None
    rows = []
    for x in speakers:
        y = ll_map[x]
        adv_path = first_wav(os.path.join(args.vctk, y))
        x_files = sorted(f for f in os.listdir(os.path.join(args.vctk, x)) if f.endswith(".wav"))
        test_files = []
        for f in x_files[args.utt_offset:]:
            if args.max_dur > 0:
                info = sf.info(os.path.join(args.vctk, x, f))
                if info.duration > args.max_dur:
                    continue
            test_files.append(f)
            if len(test_files) >= args.per_speaker:
                break
        for fn in test_files:
            utt = os.path.splitext(fn)[0]
            out_sub = os.path.join(args.out_dir, utt)
            defend_path = os.path.join(args.vctk, x, fn)
            if not os.path.exists(defend_path):
                print(f"skip {utt}: missing")
                continue
            if os.path.exists(os.path.join(out_sub, "defended.wav")):
                print(f"skip {utt}: done")
                continue
            os.makedirs(out_sub, exist_ok=True)
            # 保存内容源，供后续信道扫描复用
            sf.write(os.path.join(out_sub, "source.wav"), src_wav, sr)

            wav, theta, psd_max = file2wav_mask(
                defend_path, sample_rate=sr, n_fft=pp["n_fft"], hop_length=pp["hop_length"],
                win_length=pp["win_length"], top_db=pp["top_db"])
            adv_wav, _ = sf.read(adv_path)
            adv_wav = adv_wav.astype(np.float32)

            if args.method == "vg":
                flag_fn = make_probe_flag(x) if probe_asv is not None else None
                channel_fn = None
                if args.eot:
                    from vguard.channel import differentiable_channel
                    channel_fn = lambda adv: differentiable_channel(adv, sr)
                res = voice_guard_attack(
                    model, torch.from_numpy(wav), vc_tgt_mask=theta,
                    vc_tgt_psd_max=float(psd_max), adv_tgt=torch.from_numpy(adv_wav),
                    registry=atk_reg, vc_tgt_speaker=x, eps=args.eps,
                    n_iters1=args.iters1, n_iters2=args.iters2, c_init=1.0, attr=attr,
                    progress=not args.no_progress, flag_fn=flag_fn,
                    channel_fn=channel_fn, **feature_kwargs)
                adv_np = res["adv_wav"].numpy().astype(np.float32)
            elif args.method == "vg-nop":
                from vguard.baselines import vg_wo_p
                adv_np, flag = vg_wo_p(
                    model, config, attr, wav, adv_wav, atk_reg, x, args.eps,
                    args.iters1, args.iters2, device, not args.no_progress)
                res = {"flag": flag, "c": None}
            else:
                from vguard.baselines import freq_domain_attack
                adv_np, _ = freq_domain_attack(
                    args.method, model, config, attr, wav, adv_wav, src_wav,
                    args.eps, args.iters1, device)
                res = {"flag": None, "c": None}

            sf.write(os.path.join(out_sub, "defended.wav"), adv_np, sr)
            sf.write(os.path.join(out_sub, "original_trimmed.wav"), wav.astype(np.float32), sr)

            conv_clean = os.path.join(out_sub, "conv_from_clean.wav")
            conv_def = os.path.join(out_sub, "conv_from_defended.wav")
            vc_convert(model, config, attr, src_path,
                       os.path.join(out_sub, "original_trimmed.wav"), conv_clean, device)
            vc_convert(model, config, attr, src_path,
                       os.path.join(out_sub, "defended.wav"), conv_def, device)

            row = {"utterance": utt, "eps": args.eps, "flag": res["flag"], "c": res["c"]}
            row["mcd"] = round(mcd_mel(wav.astype(np.float32), adv_np, feature_kwargs,
                                       attr=None, device=device), 3)

            try:
                if ecapa is None:
                    ecapa = load_ecapa(savedir="checkpoints/ecapa-local")
                for key, p in [("defended", os.path.join(out_sub, "defended.wav")),
                               ("conv_clean", conv_clean), ("conv_def", conv_def)]:
                    a, _ = sf.read(p)
                    e = embed_wav(ecapa, a.astype(np.float32), sr)
                    sims = np.asarray([
                        float(np.dot(e, re_) / (np.linalg.norm(e) * np.linalg.norm(re_) + 1e-9))
                        for re_ in reg_npz["embeddings"]])
                    pred = eval_reg.names[int(np.argmax(sims))]
                    row[f"asv_{key}"] = pred
                    row[f"simX_{key}"] = round(float(sims[eval_reg.names.index(x)]), 3)
                row["conv_clean_hits_X"] = row["asv_conv_clean"] == x
                row["defense_ok"] = (row["asv_defended"] == x) and (row["asv_conv_def"] != x)
            except Exception as e:  # noqa: BLE001
                row["asv_error"] = str(e)[:200]

            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

            with open(os.path.join(args.out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

    if rows:
        n = len(rows)
        ok = sum(1 for r in rows if r.get("defense_ok"))
        print(f"SUMMARY: {n} utterances, defense_ok {ok} ({100 * ok / max(n, 1):.1f}%)")
        print("csv:", os.path.join(args.out_dir, "results.csv"))


if __name__ == "__main__":
    main()
