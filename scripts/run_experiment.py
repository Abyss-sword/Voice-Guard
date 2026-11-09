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
Resume: only complete result.json records with all five WAV artifacts are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vguard.attack import voice_guard_attack
from vguard.audio import load_audio
from vguard.asv import embed_wav, load_ecapa
from vguard.masking import file2wav_mask
from vguard.metrics.mcd import mcd_mel
from vguard.speaker import SpeakerRegistry
from vguard.vc.adainvc import convert as vc_convert
from vguard.vc.adainvc import load_adainvc
from vguard.run_state import (completed_row, file_sha256, summarize, utterance_seed,
                              write_csv, write_json)


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
    ap.add_argument("--imp-eot", action="store_true",
                    help="防御侧 EOT：阶段2 增加「信道后身份仍保持 (F_imp)」正则项")
    ap.add_argument("--imp-eot-weight", type=float, default=0.5,
                    help="防御侧 EOT 正则权重（默认 0.5）")
    ap.add_argument("--band-alloc", action="store_true",
                    help="L3：按掩蔽阈 θx 分配频带扰动能量（阈值高的频带更多）")
    ap.add_argument("--iters1", type=int, default=3000)
    ap.add_argument("--iters2", type=int, default=1500)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device
    if args.per_speaker < 1 or args.utt_offset < 30 or min(args.iters1, args.iters2) < 1:
        ap.error("per-speaker/iterations must be positive and utt-offset >= 30 (no enrollment leakage)")
    if not 0 < args.eps <= 0.5:
        ap.error("eps must be in (0, 0.5]")
    if device.startswith("cuda") and not torch.cuda.is_available():
        ap.error("CUDA requested but unavailable; use a GPU host or explicitly select --device cpu")
    checkpoint = Path(args.model_dir) / "vctk_model.ckpt"
    if not checkpoint.exists():
        checkpoint = Path(args.model_dir) / "model.ckpt"
    signature = {k: v for k, v in vars(args).items() if k not in ("no_progress", "out_dir")}
    signature["input_sha256"] = {str(p): file_sha256(p) for p in
        (checkpoint, Path(args.model_dir) / "config.yaml", Path(args.model_dir) / "attr.pkl",
         Path(args.registry), Path(args.attack_registry))}
    manifest_path = Path(args.out_dir) / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["parameters"] != signature:
            ap.error("Existing experiment parameters/assets differ; choose a new --out-dir")
    else:
        if list(Path(args.out_dir).glob("*/defended.wav")):
            ap.error("Legacy artifacts have no manifest; use a new --out-dir or reevaluate them separately")
        write_json(manifest_path, {"parameters": signature, "python": sys.version,
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None})

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
    src_wav = load_audio(src_path, sr)
    source_sha = file_sha256(src_path)

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
    result_csv = os.path.join(args.out_dir, "results.csv")
    rows = []
    if os.path.exists(result_csv):
        with open(result_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    def save_row(row):
        nonlocal rows
        rows = [r for r in rows if r["utterance"] != row["utterance"]] + [row]
        rows.sort(key=lambda r: r["utterance"])
        write_csv(result_csv, rows)
        write_json(Path(args.out_dir) / "summary.json", summarize(rows))
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
        if len(test_files) < args.per_speaker:
            raise ValueError(f"{x}: requested {args.per_speaker} test files but found {len(test_files)}")
        for fn in test_files:
            utt = os.path.splitext(fn)[0]
            out_sub = os.path.join(args.out_dir, utt)
            defend_path = os.path.join(args.vctk, x, fn)
            if not os.path.exists(defend_path):
                print(f"skip {utt}: missing")
                continue
            previous = completed_row(out_sub)
            input_hashes = {"source_sha256": source_sha, "target_sha256": file_sha256(defend_path),
                            "adv_sha256": file_sha256(adv_path)}
            if previous is not None:
                if any(previous.get(k) != v for k, v in input_hashes.items()):
                    raise ValueError(f"{utt}: input audio changed; choose a new --out-dir")
                save_row(previous)
                print(f"skip {utt}: done")
                continue
            seed = utterance_seed(args.seed, utt)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            sample_started = time.monotonic()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            os.makedirs(out_sub, exist_ok=True)
            # 保存内容源，供后续信道扫描复用
            sf.write(os.path.join(out_sub, "source.wav"), src_wav, sr)

            wav, theta, psd_max = file2wav_mask(
                defend_path, sample_rate=sr, n_fft=pp["n_fft"], hop_length=pp["hop_length"],
                win_length=pp["win_length"], top_db=pp["top_db"])
            adv_wav = load_audio(adv_path, sr)

            if args.method == "vg":
                flag_fn = make_probe_flag(x) if probe_asv is not None else None
                channel_fn = None
                if args.eot:
                    from vguard.channel import differentiable_channel
                    channel_fn = lambda adv: differentiable_channel(adv, sr)
                imp_channel_fn = None
                if getattr(args, "imp_eot", False):
                    from vguard.channel import differentiable_channel
                    imp_channel_fn = lambda adv: differentiable_channel(adv, sr)
                res = voice_guard_attack(
                    model, torch.from_numpy(wav), vc_tgt_mask=theta,
                    vc_tgt_psd_max=float(psd_max), adv_tgt=torch.from_numpy(adv_wav),
                    registry=atk_reg, vc_tgt_speaker=x, eps=args.eps,
                    n_iters1=args.iters1, n_iters2=args.iters2, c_init=1.0, attr=attr,
                    progress=not args.no_progress, flag_fn=flag_fn,
                    channel_fn=channel_fn,
                    imp_channel_fn=imp_channel_fn,
                    imp_weight=float(getattr(args, "imp_eot_weight", 0.0)),
                    band_alloc=bool(getattr(args, "band_alloc", False)),
                    **feature_kwargs)
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

            row = {"utterance": utt, "method": args.method, "seed": seed,
                   "eps": args.eps, "flag": res["flag"], "c": res["c"]}
            if "stage" in res:
                row["stage"] = res["stage"]
            row.update(input_hashes)
            row["mcd"] = round(mcd_mel(wav.astype(np.float32), adv_np, feature_kwargs,
                                       attr=None, device=device), 3)

            try:
                if ecapa is None:
                    ecapa = load_ecapa(device=device, savedir="checkpoints/ecapa-local")
                for key, p in [("original", os.path.join(out_sub, "original_trimmed.wav")),
                               ("defended", os.path.join(out_sub, "defended.wav")),
                               ("conv_clean", conv_clean), ("conv_def", conv_def)]:
                    a, audio_sr = sf.read(p)
                    e = embed_wav(ecapa, a.astype(np.float32), audio_sr)
                    sims = np.asarray([
                        float(np.dot(e, re_) / (np.linalg.norm(e) * np.linalg.norm(re_) + 1e-9))
                        for re_ in reg_npz["embeddings"]])
                    pred = eval_reg.names[int(np.argmax(sims))]
                    row[f"asv_{key}"] = pred
                    row[f"simX_{key}"] = round(float(sims[eval_reg.names.index(x)]), 3)
                row["conv_clean_hits_X"] = row["asv_conv_clean"] == x
                row["original_hits_X"] = row["asv_original"] == x
                row["defense_ok"] = (row["asv_defended"] == x) and (row["asv_conv_def"] != x)
            except Exception as e:  # noqa: BLE001
                row["asv_error"] = str(e)[:200]

            if device.startswith("cuda"):
                torch.cuda.synchronize()
                row["peak_cuda_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
                row["peak_cuda_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1024**2, 1)
            row["elapsed_s"] = round(time.monotonic() - sample_started, 2)
            write_json(Path(out_sub) / "result.json", row)
            save_row(row)
            print(json.dumps(row, ensure_ascii=False))

    if rows:
        n = len(rows)
        print("SUMMARY:", json.dumps(summarize(rows), ensure_ascii=False))
        print("csv:", os.path.join(args.out_dir, "results.csv"))
        if summarize(rows)["asv_missing"]:
            raise SystemExit("ASV evaluation incomplete; artifacts saved for retry. This run is not complete.")


if __name__ == "__main__":
    main()
