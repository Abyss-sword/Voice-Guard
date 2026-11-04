"""Build speaker registries for the attack (Es embeddings, 128-dim) and eval (ECAPA, 192-dim).

Usage:
    python scripts/build_registry.py --mode vc --model-dir checkpoints/adain-vc-official \
        --vctk data/VCTK22 --speakers p225,... --per-speaker 30 --out outputs/registry_vc.npz
    python scripts/build_registry.py --mode ecapa --vctk data/VCTK22 --speakers p225,... \
        --per-speaker 30 --out outputs/registry_ecapa.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vguard.asv import build_registry, load_ecapa  # noqa: E402
from vguard.vc.adainvc import load_adainvc, speaker_embed  # noqa: E402
from vguard.run_state import file_sha256, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vc", "ecapa"], required=True)
    ap.add_argument("--model-dir", help="required for mode=vc")
    ap.add_argument("--vctk", required=True)
    ap.add_argument("--speakers", required=True, help="comma-separated speaker ids")
    ap.add_argument("--per-speaker", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    names = [s.strip() for s in args.speakers.split(",") if s.strip()]
    if len(set(names)) != len(names) or len(names) < 2 or args.per_speaker < 1:
        ap.error("need >=2 unique speakers and positive per-speaker")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    enrollment = {}
    for name in names:
        files = sorted(Path(args.vctk, name).glob("*.wav"))
        if len(files) < args.per_speaker:
            ap.error(f"{name}: need {args.per_speaker} enrollment WAVs, found {len(files)}")
        enrollment[name] = [{"path": str(path), "sha256": file_sha256(path)}
                            for path in files[:args.per_speaker]]

    if args.mode == "vc":
        model, config, attr = load_adainvc(args.model_dir, device=args.device)
        pp = config["preprocess"]
        embs = []
        for n in names:
            d = os.path.join(args.vctk, n)
            wav_files = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
            vs = []
            for f in wav_files[: args.per_speaker]:
                w, sr = sf.read(os.path.join(d, f))
                if sr != pp["sample_rate"]:
                    import librosa
                    w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=pp["sample_rate"])
                vs.append(speaker_embed(model, config, attr, w.astype(np.float32), args.device))
            embs.append(np.mean(vs, axis=0))
            print(f"{n}: {len(vs)} utterances -> emb {embs[-1].shape}")
        embs_np = np.stack(embs).astype(np.float32)
        embs = embs_np
        np.savez(args.out, names=np.asarray(names), embeddings=embs_np)
        print(f"saved {args.out}: {names} {embs_np.shape}")
    else:
        cls = load_ecapa(device=args.device, savedir="checkpoints/ecapa-local")
        speaker_wavs = {}
        for n in names:
            d = os.path.join(args.vctk, n)
            wav_files = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
            wavs = []
            for f in wav_files[: args.per_speaker]:
                w, sr = sf.read(os.path.join(d, f))
                if sr != 16000:
                    import torch
                    import torchaudio.functional as F
                    w = F.resample(torch.from_numpy(w).float().unsqueeze(0), sr, 16000).squeeze(0).numpy()
                wavs.append(w.astype(np.float32))
            speaker_wavs[n] = wavs
            print(f"{n}: {len(wavs)} utterances")
        names_out, embs = build_registry(cls, speaker_wavs)
        names = names_out
        np.savez(args.out, names=np.asarray(names), embeddings=embs.astype(np.float32))

    write_json(str(args.out) + ".manifest.json", {
        "mode": args.mode, "device": args.device, "per_speaker": args.per_speaker,
        "enrollment": enrollment, "registry_sha256": file_sha256(args.out),
        "shape": list(embs.shape),
    })
    print(f"saved {args.out}: {names} {embs.shape}")


if __name__ == "__main__":
    main()
