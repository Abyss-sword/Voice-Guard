"""Synthetic CPU check with real AdaIN weights; never a defense-effectiveness result."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frontend", action="store_true", help="also construct the UI and check local HTTP")
    args = ap.parse_args()
    os.chdir(ROOT)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    import numpy as np
    import soundfile as sf
    import torch
    from vguard.attack import voice_guard_attack
    from vguard.masking import generate_th
    from vguard.speaker import SpeakerRegistry
    from vguard.vc.adainvc import load_adainvc, speaker_embed, convert
    from vguard.run_state import file_sha256, write_json

    torch.set_num_threads(2)
    torch.manual_seed(2026)
    out = ROOT / "outputs/local-verification"
    out.mkdir(parents=True, exist_ok=True)
    model, config, attr = load_adainvc("checkpoints/adain-vc-official", "cpu")
    pp = config["preprocess"]
    sr = pp["sample_rate"]
    t = np.arange(sr, dtype=np.float32) / sr
    x = (.1 * np.sin(2 * np.pi * (220 * t + 100 * t**2))).astype("float32")
    y = (.1 * np.sin(2 * np.pi * (510 * t + 50 * t**2))).astype("float32")
    ex = speaker_embed(model, config, attr, x, "cpu")
    ey = speaker_embed(model, config, attr, y, "cpu")
    registry = SpeakerRegistry(np.stack([ex, ey]), ["synthetic_a", "synthetic_b"])
    theta, maximum = generate_th(x, sample_rate=sr, n_fft=pp["n_fft"],
                                  hop_length=pp["hop_length"], win_length=pp["win_length"])
    feature_kwargs = {k: pp[k] for k in ("sample_rate", "preemph", "n_fft", "hop_length",
                                        "win_length", "n_mels", "ref_db", "max_db", "top_db")}
    # A forced flag exercises stage 2 without claiming speaker verification on synthetic tones.
    result = voice_guard_attack(model, torch.from_numpy(x), theta, float(maximum), torch.from_numpy(y),
        registry, "synthetic_a", .02, n_iters1=2, n_iters2=1, attr=attr,
        progress=False, flag_fn=lambda embedding: True, **feature_kwargs)
    adv = result["adv_wav"].numpy()
    assert adv.shape == x.shape and np.isfinite(adv).all()
    sf.write(out / "source.wav", y, sr)
    sf.write(out / "defended.wav", adv, sr)
    converted = convert(model, config, attr, str(out / "source.wav"), str(out / "defended.wav"),
                        str(out / "converted.wav"), "cpu")
    assert converted.size and np.isfinite(converted).all()
    report = {"scope": "synthetic CPU execution only; no VCTK, no ASV accuracy or defense success measured",
              "parameters": sum(p.numel() for p in model.parameters()), "embedding_dim": int(ex.size),
              "sample_rate": sr, "two_stage_execution": True, "conversion_finite": True,
              "checkpoint_sha256": file_sha256("checkpoints/adain-vc-official/vctk_model.ckpt")}
    if args.frontend:
        from frontend.app import _patch_gradio_client_schema, build_app
        _patch_gradio_client_schema()
        app = build_app(SimpleNamespace(vctk="", out_dir=str(out), iters1=2, iters2=1),
                        SimpleNamespace(speakers=["synthetic_a", "synthetic_b"]))
        try:
            app.queue(max_size=8, default_concurrency_limit=1)
            app.launch(server_name="127.0.0.1", server_port=17860, prevent_thread_lock=True, quiet=True)
            with urllib.request.urlopen("http://127.0.0.1:17860/", timeout=10) as response:
                report["frontend_http_status"] = response.status
            report["frontend_scope"] = "UI construction and HTTP only; no protect/convert event invoked"
        finally:
            app.close()
    write_json(out / "verification.json", report)
    print(report)


if __name__ == "__main__":
    main()
