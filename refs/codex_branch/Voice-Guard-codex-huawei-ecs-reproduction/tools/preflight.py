"""Readiness report. Missing resources cause exit 1 rather than a false deployment pass."""
from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json

SPEAKERS = [f"p{i}" for i in list(range(225, 235)) + list(range(236, 242)) + list(range(243, 247))]


def check(scope="runtime", device="cuda"):
    checks = []
    def add(name, ok, detail):
        checks.append(dict(name=name, ok=bool(ok), detail=detail))
    add("python", (3, 10) <= sys.version_info[:2] <= (3, 12), sys.version.split()[0])
    for filename in ("third_party/ada_in_vc/model.py", "third_party/ada_in_vc/config.yaml"):
        add(filename, (ROOT / filename).is_file(), "required upstream source")
    packages = {"torch": "torch", "torchaudio": "torchaudio", "librosa": "librosa", "numpy": "numpy",
                "soundfile": "soundfile", "yaml": "PyYAML", "speechbrain": "speechbrain", "gradio": "gradio"}
    versions = {}
    for module, package in packages.items():
        found = importlib.util.find_spec(module) is not None
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
        add(package, found, versions[package])
    gpu = None
    if versions.get("torch"):
        try:
            import torch
            gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
            add("device", device == "cpu" or gpu is not None, gpu or "CPU only")
            import torchaudio
            add("torch/torchaudio import", True, torchaudio.__version__)
        except Exception as exc:
            add("torch/torchaudio import", False, str(exc))
    if scope in ("assets", "experiment"):
        assets = ["checkpoints/adain-vc-official/config.yaml", "checkpoints/adain-vc-official/attr.pkl",
                  "checkpoints/adain-vc-official/vctk_model.ckpt"]
        assets += ["checkpoints/ecapa-local/" + f for f in ("hyperparams.yaml", "embedding_model.ckpt",
                   "mean_var_norm_emb.ckpt", "classifier.ckpt", "label_encoder.txt")]
        for filename in assets:
            path = ROOT / filename
            add(filename, path.is_file() and path.stat().st_size > 0, file_sha256(path) if path.is_file() else "missing")
    if scope == "experiment":
        for speaker in SPEAKERS:
            count = len(list((ROOT / "data/VCTK22" / speaker).glob("*.wav")))
            add(speaker, count >= 31, f"{count} WAVs; enrollment=30, test>=1")
        for filename, dim in (("registry.npz", 192), ("registry_vc.npz", 128)):
            path = ROOT / "outputs" / filename
            try:
                import numpy as np
                with np.load(path, allow_pickle=False) as reg:
                    names, embeddings = list(reg["names"]), reg["embeddings"]
                    ok = set(names) == set(SPEAKERS) and len(names) == len(SPEAKERS)
                    ok = ok and embeddings.shape == (len(names), dim) and np.isfinite(embeddings).all()
                    add(filename, ok, f"{embeddings.shape}; sha256={file_sha256(path)}")
            except Exception as exc:
                add(filename, False, str(exc))
    disk = shutil.disk_usage(ROOT)
    return {"ready": all(c["ok"] for c in checks), "scope": scope, "device": device,
            "gpu": gpu, "free_gb": round(disk.free / 1024**3, 2), "versions": versions, "checks": checks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope", choices=["runtime", "assets", "experiment"], default="runtime")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--out")
    args = ap.parse_args()
    report = check(args.scope, args.device)
    if args.out:
        write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
