"""Paired clean/protected real-codec experiments with quality and identity controls.

This is a laboratory codec simulation, not a test of WeChat or another live service.
Each source/channel gets a fresh clean-VC identity control. All failures are retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.asv import embed_wav, load_ecapa
from vguard.channel import codec_channel
from vguard.evaluation import quality_pass, summarize_pairs
from vguard.metrics.mos import load_mosnet, mos
from vguard.metrics.wer import load_whisper, transcribe, wer
from vguard.run_state import file_sha256, utterance_seed, write_csv, write_json
from vguard.vc.adainvc import convert, load_adainvc

CHANNELS = {"none", "pcm16", "mp3-128k", "mp3-32k", "aac-64k", "opus-24k", "g711", "resample8k"}


def difference_diagnostics(clean, protected, sr):
    """Common-prefix differences can contain codec time shifts; not isolated noise."""
    length = min(len(clean), len(protected))
    delta = protected[:length] - clean[:length]
    spectrum = np.fft.rfft(delta)
    energy = np.abs(spectrum)**2
    frequencies = np.fft.rfftfreq(length, 1/sr)
    total = float(energy.sum())
    return {"common_prefix_difference_rms": float(np.sqrt(np.mean(delta**2))),
            "difference_high_band_fraction_ge4khz": float(energy[frequencies >= 4000].sum()/total) if total else None,
            "clean_samples": len(clean), "protected_samples": len(protected),
            "difference_note": "Common-prefix diagnostic; codec delay/nonlinearity may contribute"}


def summarize(rows):
    grouped = {}
    controls = {(row["utterance"], row["content"]): row for row in rows if row["channel"] == "none"}
    for channel in sorted({row["channel"] for row in rows}):
        selected = [row for row in rows if row["channel"] == channel]
        stats = summarize_pairs(selected)
        initially_successful = [row for row in selected if controls.get(
            (row["utterance"],row["content"]),{}).get("screened_quality_identity_ok",False)]
        stats["initial_joint_successes"] = len(initially_successful)
        stats["retained_initial_joint_successes"] = sum(row["screened_quality_identity_ok"] for row in initially_successful)
        stats["initial_successes_losing_clean_screen"] = sum(not row["clean_vc_identity_ok"] for row in initially_successful)
        grouped[channel] = stats
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--speakers", default="p225,p226,p227,p228,p229")
    parser.add_argument("--contents", default="p226_002", help="comma-separated VCTK source utterances")
    parser.add_argument("--channels", default="none,pcm16,mp3-128k,mp3-32k,aac-64k,opus-24k,g711,resample8k")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    channels = args.channels.split(",")
    if "none" not in channels or len(set(channels)) != len(channels) or not set(channels) <= CHANNELS:
        parser.error("Use unique supported channels, including the unprocessed 'none' control")
    manifest = args.exp_dir/"manifest.json"
    params = json.loads(manifest.read_text(encoding="utf-8"))["parameters"]
    speakers = set(args.speakers.split(","))
    jobs = [directory for directory in sorted(args.exp_dir.iterdir()) if directory.is_dir()
            and directory.name.split("_")[0] in speakers and (directory/"defended.wav").is_file()
            and ((directory/"result.json").is_file() or (directory/"optimization.json").is_file())]
    if {directory.name.split("_")[0] for directory in jobs} != speakers:
        parser.error("Each requested speaker needs at least one completed input")
    contents = args.contents.split(",")
    if len(contents) != len(set(contents)):
        parser.error("Duplicate content sources")
    sources = {content: Path(params["vctk"])/content.split("_")[0]/f"{content}.wav" for content in contents}
    assets = {path: file_sha256(path) for path in params["input_sha256"]}
    if assets != params["input_sha256"]:
        raise ValueError("Input weights or registries differ from source manifest")
    ffmpeg = subprocess.run(["ffmpeg","-version"],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    input_files = [manifest, *sources.values(), *[path.with_suffix(".txt") for path in sources.values()]]
    input_files += [path for directory in jobs for path in
                    (directory/"original_trimmed.wav",directory/"defended.wav",
                     Path(params["vctk"])/directory.name.split("_")[0]/f"{directory.name}.txt")]
    scoring_files = list((ROOT/"checkpoints/mosnet-hf").glob("*"))
    scoring_files += [Path.home()/".cache/whisper/base.pt"]
    scoring_files += list((ROOT/"checkpoints/ecapa-local").glob("*.ckpt"))
    signature = {"source_manifest": str(manifest), "channels": channels, "contents":contents,
                 "speakers":sorted(speakers), "device":args.device, "ffmpeg":ffmpeg,
                 "input_sha256":{str(path):file_sha256(path) for path in input_files},
                 "model_sha256":{str(path):file_sha256(path) for path in scoring_files if path.is_file()},
                 "code_sha256":{name:file_sha256(ROOT/name) for name in
                    ("scripts/audit_channels.py","vguard/channel.py","vguard/evaluation.py","vguard/asv.py",
                     "vguard/vc/adainvc.py","vguard/metrics/mos.py","vguard/metrics/wer.py")},
                 "gate":"MOS>=3, target transcript WER<=.10, protected identity retained, converted identity changed, paired clean-VC identity correct",
                 "limitations":["Small correlated sample", "Closed-set ECAPA and third-party MOSNet", "Laboratory codecs; no live platform or human listening test"]}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    output_manifest=args.out_dir/"manifest.json"
    if output_manifest.exists() and json.loads(output_manifest.read_text(encoding="utf-8")) != signature:
        raise ValueError("Inputs/code/parameters changed; choose a new audit output directory")
    write_json(output_manifest,signature)
    model,config,attr=load_adainvc(params["model_dir"],device=args.device)
    classifier=load_ecapa(device=args.device)
    mos_model=load_mosnet(device=args.device)
    recognizer=load_whisper("base",device=args.device)
    with np.load(params["registry"],allow_pickle=False) as registry:
        names=list(registry["names"])
        embeddings=np.array(registry["embeddings"],copy=True)
    def predict(wave,sr):
        embedding=embed_wav(classifier,wave,sr)
        scores=embeddings@embedding/(np.linalg.norm(embeddings,axis=1)*np.linalg.norm(embedding)+1e-9)
        return str(names[int(np.argmax(scores))]),{str(name):float(score) for name,score in zip(names,scores)}
    rows=[]
    for directory in jobs:
        utterance=directory.name
        target=utterance.split("_")[0]
        clean,sr=sf.read(directory/"original_trimmed.wav",dtype="float32")
        protected,protected_sr=sf.read(directory/"defended.wav",dtype="float32")
        if sr != protected_sr or clean.shape != protected.shape:
            raise ValueError(f"Input waveforms are not aligned: {utterance}")
        reference=(Path(params["vctk"])/target/f"{utterance}.txt").read_text().strip()
        for channel in channels:
            case=args.out_dir/utterance/channel
            case.mkdir(parents=True,exist_ok=True)
            clean_path,protected_path=case/"clean.wav",case/"defended.wav"
            if not all((case/content/"result.json").is_file() for content in contents):
                sf.write(clean_path,codec_channel(clean,sr,channel),sr,subtype="FLOAT")
                sf.write(protected_path,codec_channel(protected,sr,channel),sr,subtype="FLOAT")
            clean_ch,_=sf.read(clean_path,dtype="float32")
            protected_ch,_=sf.read(protected_path,dtype="float32")
            shared=None
            for content,source in sources.items():
                output=case/content
                record=output/"result.json"
                if record.exists():
                    saved=json.loads(record.read_text(encoding="utf-8"))
                    if any(file_sha256(args.out_dir/path) != digest for path,digest in saved["artifacts_sha256"].items()):
                        raise ValueError(f"Completed artifacts changed: {output}")
                    rows.append(saved)
                    continue
                started=time.monotonic()
                output.mkdir(parents=True,exist_ok=True)
                if shared is None:
                    clean_pred,clean_scores=predict(clean_ch,sr)
                    protected_pred,protected_scores=predict(protected_ch,sr)
                    clean_hyp=transcribe(recognizer,str(clean_path))
                    protected_hyp=transcribe(recognizer,str(protected_path))
                    shared={"asv_clean":clean_pred,"asv_defended":protected_pred,
                            "clean_scores":clean_scores,"protected_scores":protected_scores,
                            "mos_clean":mos(mos_model,clean_ch,sr),"mos_defended":mos(mos_model,protected_ch,sr),
                            "wer_clean":wer(clean_hyp,reference),"wer_defended":wer(protected_hyp,reference),
                            "clean_hypothesis":clean_hyp,"protected_hypothesis":protected_hyp,"reference":reference,
                            **difference_diagnostics(clean_ch,protected_ch,sr)}
                conversion_info={}
                source_text=source.with_suffix(".txt").read_text().strip()
                seed=utterance_seed(2026,f"{utterance}:{content}")
                for label,path in (("clean",clean_path),("defended",protected_path)):
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                    destination=output/f"converted_{label}.wav"
                    convert(model,config,attr,str(source),str(path),str(destination),args.device)
                    wave,conv_sr=sf.read(destination,dtype="float32")
                    pred,scores=predict(wave,conv_sr)
                    hypothesis=transcribe(recognizer,str(destination))
                    conversion_info.update({f"asv_converted_{label}":pred,f"converted_{label}_scores":scores,
                                            f"mos_converted_{label}":mos(mos_model,wave,conv_sr),
                                            f"wer_converted_{label}":wer(hypothesis,source_text),
                                            f"converted_{label}_hypothesis":hypothesis})
                clean_ok=conversion_info["asv_converted_clean"]==target
                quality_ok=quality_pass(shared["mos_defended"],shared["wer_defended"],shared["asv_defended"]==target)
                identity_ok=shared["asv_defended"]==target and conversion_info["asv_converted_defended"]!=target
                artifacts=[clean_path,protected_path,output/"converted_clean.wav",output/"converted_defended.wav"]
                row={"utterance":utterance,"channel":channel,"content":content,"seed":seed,
                     **shared,**conversion_info,"source_reference":source_text,
                     "clean_vc_identity_ok":clean_ok,"quality_ok":quality_ok,"identity_ok":identity_ok,
                     "screened_quality_identity_ok":bool(clean_ok and quality_ok and identity_ok),
                     "elapsed_s":round(time.monotonic()-started,2),
                     "artifacts_sha256":{path.relative_to(args.out_dir).as_posix():file_sha256(path) for path in artifacts}}
                write_json(record,row)
                rows.append(row)
                write_csv(args.out_dir/"results.csv",rows)
                print(json.dumps({key:row[key] for key in ("utterance","channel","content","mos_defended","wer_defended","clean_vc_identity_ok","screened_quality_identity_ok")}),flush=True)
    write_csv(args.out_dir/"results.csv",rows)
    write_json(args.out_dir/"summary.json",{"complete":True,"expected_rows":len(jobs)*len(channels)*len(contents),
               "rows":rows,"channels":summarize(rows),"protocol":signature})


if __name__=="__main__":main()
