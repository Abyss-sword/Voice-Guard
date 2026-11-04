"""Audit the quality/identity frontier of existing VG perturbations, including zero noise.

This is an explicit post-processing experiment, not a change to the paper baseline.
MOS>=3 and target WER<=0.10 are engineering gates; ASR_th in the paper is ambiguous.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.asv import embed_wav, load_ecapa
from vguard.audio import load_audio, restore_protected_crop
from vguard.masking import generate_th, perturbation_psd, project_masked_delta
from vguard.metrics.mos import load_mosnet, mos
from vguard.metrics.wer import load_whisper, transcribe, wer, cer
from vguard.run_state import file_sha256, utterance_seed, write_csv, write_json
from vguard.vc.adainvc import convert, load_adainvc


def quality_gate(mos_score, target_wer, identity_preserved):
    return bool(identity_preserved and np.isfinite(mos_score) and np.isfinite(target_wer)
                and mos_score >= 3.0 and target_wer <= 0.10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--speakers", default="p225,p226", help="development speakers; freeze settings before validation")
    parser.add_argument("--scales", default="0,0.05,0.1,0.2,0.4,0.7,1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-project-iterations", type=int, default=0)
    parser.add_argument("--mask-margin-db", type=float, default=3.0)
    parser.add_argument("--restore-context", action="store_true", help="restore unmodified samples outside the attack crop")
    parser.add_argument("--wav-subtype", choices=["FLOAT", "PCM_16"], default="FLOAT",
                        help="evaluate the decoded exported waveform, including quantization")
    args = parser.parse_args()
    scales = [float(value) for value in args.scales.split(",")]
    if not scales or len(set(scales)) != len(scales) or any(not np.isfinite(value) or not 0 <= value <= 1 for value in scales):
        parser.error("scales must be finite, unique and in [0,1]")
    if 0 not in scales:
        parser.error("include scale=0 to screen clean conversion identity")
    if args.mask_project_iterations < 0 or not np.isfinite(args.mask_margin_db) or args.mask_margin_db < 0:
        parser.error("projection iterations/margin must be nonnegative and finite")
    source_manifest = args.exp_dir / "manifest.json"
    params = json.loads(source_manifest.read_text(encoding="utf-8"))["parameters"]
    for path, expected in params["input_sha256"].items():
        if file_sha256(path) != expected:
            raise ValueError(f"Source experiment asset changed: {path}")
    speakers = set(args.speakers.split(","))
    jobs = [directory for directory in sorted(args.exp_dir.iterdir())
            if directory.is_dir() and directory.name.split("_")[0] in speakers
            and ((directory / "result.json").is_file() or (directory / "optimization.json").is_file())]
    if {directory.name.split("_")[0] for directory in jobs} != speakers:
        parser.error("all requested source speakers must have a completed result")
    signature = {"source_manifest_sha256": file_sha256(source_manifest), "scales": scales,
                 "mask_project_iterations": args.mask_project_iterations, "mask_margin_db": args.mask_margin_db,
                 "restore_context": args.restore_context,
                 "wav_subtype": args.wav_subtype,
                 "code_sha256": file_sha256(Path(__file__)),
                 "masking_sha256": file_sha256(ROOT / "vguard/masking.py"),
                 "evaluation_code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in
                    [ROOT / name for name in ("vguard/audio.py", "vguard/asv.py", "vguard/vc/adainvc.py",
                                             "vguard/metrics/mos.py", "vguard/metrics/wer.py")]},
                 "quality_model_sha256": {str(path): file_sha256(path) for path in
                     [ROOT / "checkpoints/mosnet-hf" / name for name in
                      ("pytorch_model.bin", "config.json", "modeling_mosnet.py", "configuration_mosnet.py")]
                     + [Path.home() / ".cache/whisper/base.pt"]},
                 "speakers": sorted(speakers), "device": args.device,
                 "gate": "MOS>=3, target WER<=0.10, ECAPA protected identity; not the exact paper ASR metric",
                 "inputs": {str(path): file_sha256(path) for directory in jobs for path in
                            (directory / "original_trimmed.wav", directory / "defended.wav")},
                 "reference_inputs": {str(path): file_sha256(path) for path in
                    [Path(params['vctk']) / directory.name.split('_')[0] / f'{directory.name}.{suffix}'
                     for directory in jobs for suffix in ('wav','txt')] +
                    [Path(params['vctk']) / params['source_speaker'] / f"{params['source_speaker']}_{params['source_utt']:03d}.{suffix}"
                     for suffix in ('wav','txt')]}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != signature:
        raise ValueError("Sweep inputs/parameters changed; use another output directory")
    write_json(manifest, signature)
    model, config, attr = load_adainvc(params["model_dir"], device=args.device)
    classifier = load_ecapa(device=args.device)
    mos_model = load_mosnet(device=args.device)
    recognizer = load_whisper("base", device=args.device)
    with np.load(params["registry"], allow_pickle=False) as registry:
        names = list(registry["names"])
        embeddings = np.array(registry["embeddings"], copy=True)
    def predict(wave, sr):
        embedding = embed_wav(classifier, wave, sr)
        similarities = embeddings @ embedding / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embedding) + 1e-9)
        order = np.argsort(similarities)[::-1]
        return str(names[int(order[0])]), {str(names[int(i)]): float(similarities[i]) for i in order}
    pp = config["preprocess"]
    source = Path(params["vctk"]) / params["source_speaker"] / f"{params['source_speaker']}_{params['source_utt']:03d}.wav"
    rows = []
    for directory in jobs:
        utterance, target = directory.name, directory.name.split("_")[0]
        reference = (Path(params["vctk"]) / target / f"{utterance}.txt").read_text().strip()
        source_reference = source.with_suffix(".txt").read_text().strip()
        original, sr = sf.read(directory / "original_trimmed.wav", dtype="float32")
        protected, sr2 = sf.read(directory / "defended.wav", dtype="float32")
        if original.shape != protected.shape or sr != sr2:
            raise ValueError("Only waveform-aligned perturbations are supported")
        if args.restore_context:
            import librosa
            full = load_audio(str(Path(params['vctk']) / target / f'{utterance}.wav'), sr)
            _, interval = librosa.effects.trim(full, top_db=pp['top_db'])
            protected = restore_protected_crop(full, original, protected, *map(int, interval))
            original = full
        theta, psd_max = generate_th(original, sr, pp["n_fft"], pp["hop_length"], pp["win_length"])
        for scale in scales:
            output = args.out_dir / utterance / f"scale_{scale:g}"
            record = output / "result.json"
            if record.exists() and (output / "defended.wav").is_file() and (output / "converted.wav").is_file():
                saved = json.loads(record.read_text(encoding="utf-8"))
                if (saved["audio_sha256"] != file_sha256(output / "defended.wav") or
                        saved["converted_sha256"] != file_sha256(output / "converted.wav")):
                    raise ValueError(f"Completed audio changed: {output}")
                rows.append(saved)
                write_csv(args.out_dir / "results.csv", rows)
                continue
            started = time.monotonic()
            seed = utterance_seed(2026, f"{utterance}:{scale:g}")
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            output.mkdir(parents=True, exist_ok=True)
            delta = scale * (protected - original)
            if args.mask_project_iterations:
                delta = project_masked_delta(delta, theta, psd_max, pp["n_fft"], pp["hop_length"],
                                             pp["win_length"], args.mask_project_iterations, args.mask_margin_db)
            wave = original + delta
            if np.max(np.abs(wave)) > 1:
                raise ValueError("Candidate would clip; refusing silent amplitude normalization")
            path = output / "defended.wav"
            sf.write(path, wave, sr, subtype=args.wav_subtype)
            wave, exported_sr = sf.read(path, dtype="float32")
            if exported_sr != sr or not np.isfinite(wave).all():
                raise ValueError("Invalid exported waveform")
            hypothesis = transcribe(recognizer, str(path))
            converted = output / "converted.wav"
            convert(model, config, attr, str(source), str(path), str(converted), args.device)
            conversion, conv_sr = sf.read(converted, dtype="float32")
            conv_hypothesis = transcribe(recognizer, str(converted))
            delta = wave - original
            psd = perturbation_psd(delta, pp["n_fft"], pp["hop_length"], pp["win_length"], psd_max).T
            excess = np.maximum(psd - theta, 0)
            prediction, protected_similarities = predict(wave, sr)
            conv_prediction, converted_similarities = predict(conversion, conv_sr)
            row = {"utterance": utterance, "scale": scale, "seed": seed,
                   "restored_context": args.restore_context,
                   "wav_subtype": args.wav_subtype,
                   "mos_defended": mos(mos_model, wave, sr), "wer_defended": wer(hypothesis, reference),
                   "cer_defended": cer(hypothesis, reference), "hypothesis": hypothesis, "reference": reference,
                   "asv_defended": prediction, "asv_converted": conv_prediction,
                   "protected_similarities": protected_similarities,
                   "converted_similarities": converted_similarities,
                   "mos_converted": mos(mos_model, conversion, conv_sr),
                   "wer_converted": wer(conv_hypothesis, source_reference),
                   "converted_hypothesis": conv_hypothesis, "source_reference": source_reference,
                   "linf": float(np.max(np.abs(delta))),
                   "snr_db": float(10*np.log10(np.mean(original**2)/np.mean(delta**2))) if np.any(delta) else None,
                   "mask_excess_db_mean": float(np.mean(excess)), "mask_exceed_fraction": float(np.mean(excess > 0)),
                   "identity_ok": prediction == target and conv_prediction != target,
                   "elapsed_s": round(time.monotonic() - started, 2),
                   "audio_sha256": file_sha256(path), "converted_sha256": file_sha256(converted)}
            row["quality_ok"] = quality_gate(row["mos_defended"], row["wer_defended"], prediction == target)
            # This needs the independently measured scale=0 control to be interpretable.
            row["quality_and_identity_ok"] = row["quality_ok"] and row["identity_ok"]
            write_json(record, row)
            rows.append(row)
            write_csv(args.out_dir / "results.csv", rows)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    controls = {row["utterance"]: row for row in rows if row["scale"] == 0}
    for row in rows:
        control = controls.get(row["utterance"])
        row["clean_vc_identity_ok"] = None if control is None else control["asv_converted"] == row["utterance"].split("_")[0]
        row["screened_quality_identity_ok"] = row["quality_and_identity_ok"] and row["clean_vc_identity_ok"] is True
    write_csv(args.out_dir / "results.csv", rows)
    write_json(args.out_dir / "summary.json", {"candidates": len(rows), "rows": rows,
        "protocol": signature["gate"], "note": "A sweep is not held-out validation unless parameters were frozen before evaluation"})


if __name__ == "__main__":
    main()
