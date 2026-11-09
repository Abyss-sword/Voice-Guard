"""Official AdaIN-VC, pinned ECAPA and VCTK22 downloads with curl resume."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json

ECAPA_REV = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
MOSNET_REV = "4162a3e63ac725ecf7dce8dee3aa71e7afd9c65d"
VCTK_SHA = "a758d8e3e7ccf4e744508d431420a6f434c9369b1311c4fa0673c1839e000411"
PREPROCESS = dict(sample_rate=24000, preemph=0.97, n_fft=2048, hop_length=300,
                  win_length=1200, n_mels=512, ref_db=20, max_db=100, top_db=15, n_iter=100)
CURL = os.getenv("VG_CURL", "curl")


def download(url, destination, expected_sha=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_suffix(destination.suffix + ".download.json")
    if destination.exists() and receipt.exists():
        record = json.loads(receipt.read_text(encoding="utf-8"))
        actual = file_sha256(destination)
        if record.get("url") == url and actual == record.get("sha256") and (not expected_sha or actual == expected_sha):
            print("verified:", destination)
            return
    partial = destination.with_suffix(destination.suffix + ".part")
    subprocess.run([CURL, "--fail", "--location", "--retry", "4", "--connect-timeout", "20",
                    "--speed-time", "120", "--speed-limit", "1024", "--continue-at", "-",
                    "--output", str(partial), url], check=True)
    actual = file_sha256(partial)
    if expected_sha and actual != expected_sha:
        raise ValueError(f"SHA256 mismatch: {partial}; remove the partial file before retrying")
    if partial.stat().st_size == 0:
        raise ValueError(f"Empty download: {url}")
    os.replace(partial, destination)
    write_json(receipt, {"url": url, "sha256": actual, "bytes": destination.stat().st_size,
                        "verification": "publisher digest" if expected_sha else "local provenance digest"})


def adainvc():
    import yaml
    out = ROOT / "checkpoints/adain-vc-official"
    for filename in ("vctk_model.ckpt", "attr.pkl"):
        download(f"https://speech.ee.ntu.edu.tw/~jjery2243542/resource/model/is19/{filename}", out / filename)
    config = yaml.safe_load((ROOT / "third_party/ada_in_vc/config.yaml").read_text(encoding="utf-8"))
    config["preprocess"] = PREPROCESS
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def ecapa(endpoint):
    out = ROOT / "checkpoints/ecapa-local"
    for filename in ("hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt",
                     "classifier.ckpt", "label_encoder.txt"):
        target = out / ("hyperparams.upstream.yaml" if filename == "hyperparams.yaml" else filename)
        download(f"{endpoint}/speechbrain/spkrec-ecapa-voxceleb/resolve/{ECAPA_REV}/{filename}", target)
    pristine = out / "hyperparams.upstream.yaml"
    text = pristine.read_text(encoding="utf-8").replace(
        "pretrained_path: speechbrain/spkrec-ecapa-voxceleb", "pretrained_path: checkpoints/ecapa-local")
    local = out / "hyperparams.yaml"
    local.write_text(text, encoding="utf-8")
    write_json(out / "local_config.json", {"revision": ECAPA_REV,
        "upstream_sha256": file_sha256(pristine), "local_sha256": file_sha256(local)})


def mosnet(endpoint):
    out = ROOT / "checkpoints/mosnet-hf"
    for filename in ("config.json", "configuration_mosnet.py", "modeling_mosnet.py", "pytorch_model.bin"):
        target = out / ("config.upstream.json" if filename == "config.json" else filename)
        download(f"{endpoint}/1NEYRON1/mosnet/resolve/{MOSNET_REV}/{filename}", target)
    pristine = out / "config.upstream.json"
    config = json.loads(pristine.read_text(encoding="utf-8"))
    config["auto_map"] = {key: value.split("--")[-1] for key, value in config.get("auto_map", {}).items()}
    write_json(out / "config.json", config)
    write_json(out / "local_config.json", {"revision": MOSNET_REV,
        "upstream_sha256": file_sha256(pristine), "local_sha256": file_sha256(out / "config.json")})


def original_mosnet():
    revision='3ea20beaa3c1e886464fe0b1b9807fa62c732029'
    out=ROOT/'checkpoints/mosnet-original'
    files={'pre_trained/cnn_blstm.h5':'78b75e7d76ee6074ea7d57dcffa56d0c90be9d3d8dedc2217e25e259423cb756',
           'model.py':'c1c0525c8bb012ae5319ff59ebe27b53a91783f28014c62ccfccee5afb0eceb5',
           'utils.py':'61a5e2fc918cc6e8685814e4747f9c0f8b4c2bcd94a09b2dc9c2ed1085c9c78f',
           'LICENSE':'b3445db00a86dc8a53b506bf2d3010278f6ab26631fe0f06a451a1637df3891c'}
    for name,digest in files.items():
        download(f'https://raw.githubusercontent.com/lochenchou/MOSNet/{revision}/{name}',out/Path(name).name,digest)
    write_json(out/'source.json',{'repository':'lochenchou/MOSNet','revision':revision,
        'files':{Path(name).name:digest for name,digest in files.items()}})


def extract_vctk(archive, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename.replace("\\", "/")
            if member.is_dir() or not name.lower().endswith((".wav", ".txt")):
                continue
            parts = Path(name).parts
            speaker = next((p for p in parts[:-1] if p.startswith("p") and p[1:].isdigit()), None)
            if speaker is None:
                continue
            target = output / speaker / Path(name).name
            if not target.resolve().is_relative_to(output):
                raise ValueError(f"Invalid archive member: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == member.file_size:
                continue
            partial = target.with_suffix(target.suffix + ".part")
            with z.open(member) as source, partial.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            os.replace(partial, target)
    if not any(output.glob("p*/*.wav")):
        raise ValueError("No speaker WAVs found in archive")


def main():
    global CURL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["adainvc", "ecapa", "mosnet", "mosnet-original", "vctk", "models"])
    ap.add_argument("--endpoint", default=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))
    ap.add_argument("--extract", action="store_true", help="extract VCTK to data/VCTK22/pXXX")
    ap.add_argument("--curl", default=CURL, help="curl executable; Windows may use Git's OpenSSL curl")
    args = ap.parse_args()
    CURL = args.curl
    if args.what in ("adainvc", "models"):
        adainvc()
    if args.what in ("ecapa", "models"):
        ecapa(args.endpoint.rstrip("/"))
    if args.what == "mosnet":
        mosnet(args.endpoint.rstrip("/"))
    if args.what == 'mosnet-original':
        original_mosnet()
    if args.what == "vctk":
        dst = ROOT / "data/VCTK_22kHz_raw.zip"
        download(f"{args.endpoint.rstrip('/')}/datasets/arampacha/vctk_22kHz/resolve/main/raw_data.zip", dst, VCTK_SHA)
        if args.extract:
            extract_vctk(dst, ROOT / "data/VCTK22")


if __name__ == "__main__":
    main()
