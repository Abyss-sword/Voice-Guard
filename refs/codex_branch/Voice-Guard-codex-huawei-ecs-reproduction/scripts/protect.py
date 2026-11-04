"""Voice Guard end-to-end protection CLI.

Usage:
    python scripts/protect.py --model-dir checkpoints/adain-vc-official \
        --defend data/VCTK22/p225/p225_001.wav \
        --adv-target data/VCTK22/p226/p226_001.wav \
        --registry outputs/registry.npz --target-speaker p225 \
        --eps 0.1 --out-dir outputs/protect_p225_001

Outputs: defended.wav / original_trimmed.wav / delta.wav / report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vguard.attack import voice_guard_attack
from vguard.masking import file2wav_mask
from vguard.speaker import SpeakerRegistry
from vguard.vc.adainvc import load_adainvc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--defend", required=True, help="audio of target speaker X to protect")
    ap.add_argument("--adv-target", required=True, help="audio of least-likely speaker Y")
    ap.add_argument("--registry", required=True, help="eval registry npz (ECAPA, names+embeddings)")
    ap.add_argument("--attack-registry", required=True,
                    help="attack registry npz (VC speaker-encoder embeddings, same speaker names)")
    ap.add_argument("--target-speaker", required=True, help="X, must be in registry")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--iters1", type=int, default=3000)
    ap.add_argument("--iters2", type=int, default=1500)
    ap.add_argument("--c-init", type=float, default=1.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    t0 = time.time()
    model, config, attr = load_adainvc(args.model_dir, device=device)
    pp = config["preprocess"]
    print(f"[1/5] model loaded, preprocess sample_rate={pp['sample_rate']}")

    wav, theta_xs, psd_max = file2wav_mask(
        args.defend, sample_rate=pp["sample_rate"], n_fft=pp["n_fft"],
        hop_length=pp["hop_length"], win_length=pp["win_length"], top_db=pp["top_db"],
    )
    print(f"[2/5] defend wav: {len(wav)/pp['sample_rate']:.2f}s, theta {theta_xs.shape}")

    reg_npz = np.load(args.registry, allow_pickle=True)
    eval_registry = SpeakerRegistry(reg_npz["embeddings"], list(reg_npz["names"])).to(device)

    atk_npz = np.load(args.attack_registry, allow_pickle=True)
    attack_registry = SpeakerRegistry(atk_npz["embeddings"], list(atk_npz["names"])).to(device)
    assert args.target_speaker in attack_registry.names, (args.target_speaker, attack_registry.names)

    adv_wav, _ = sf.read(args.adv_target)
    adv_wav = adv_wav.astype(np.float32)

    feature_kwargs = {k: pp[k] for k in
                      ("sample_rate", "preemph", "n_fft", "hop_length",
                       "win_length", "n_mels", "ref_db", "max_db", "top_db")}
    res = voice_guard_attack(
        model,
        torch.from_numpy(wav),
        vc_tgt_mask=theta_xs,
        vc_tgt_psd_max=float(psd_max),
        adv_tgt=torch.from_numpy(adv_wav),
        registry=attack_registry,
        vc_tgt_speaker=args.target_speaker,
        eps=args.eps,
        n_iters1=args.iters1,
        n_iters2=args.iters2,
        c_init=args.c_init,
        attr=attr,
        progress=not args.no_progress,
        **feature_kwargs,
    )
    print(f"[4/5] attack done: flag={res['flag']}, c={res['c']:.4g}, stage={res['stage']}")

    adv_np = res["adv_wav"].numpy().astype(np.float32)
    sr = pp["sample_rate"]
    sf.write(os.path.join(args.out_dir, "defended.wav"), adv_np, sr)
    sf.write(os.path.join(args.out_dir, "original_trimmed.wav"), wav.astype(np.float32), sr)
    sf.write(os.path.join(args.out_dir, "delta.wav"), (adv_np - wav).astype(np.float32), sr)

    asv_note = "ASV check skipped"
    try:
        from vguard.asv import embed_wav, load_ecapa
        cls = load_ecapa(savedir="checkpoints/ecapa-local")
        emb_def = embed_wav(cls, adv_np, sr)
        sims = np.asarray([
            np.dot(emb_def, e) / (np.linalg.norm(emb_def) * np.linalg.norm(e) + 1e-9)
            for e in reg_npz["embeddings"]
        ])
        best_i = int(np.argmax(sims))
        asv_note = {"pred": eval_registry.names[best_i],
                    "sim_to_target": float(sims[eval_registry.names.index(args.target_speaker)]),
                    "defense_ok": eval_registry.names[best_i] != args.target_speaker}
    except Exception as e:  # noqa: BLE001
        asv_note = f"ASV check failed: {e}"

    report = {
        "defend": os.path.abspath(args.defend),
        "adv_target": os.path.abspath(args.adv_target),
        "target_speaker": args.target_speaker,
        "eps": args.eps, "iters1": args.iters1, "iters2": args.iters2,
        "c_final": res["c"], "defense_flag": res["flag"], "stage": res["stage"],
        "duration_s": len(wav) / sr,
        "elapsed_s": round(time.time() - t0, 1),
        "asv": asv_note,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[5/5] saved to {args.out_dir}")


if __name__ == "__main__":
    main()
