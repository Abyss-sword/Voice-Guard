"""Run the service or an explicit experiment command in the same image."""
import os
import sys

if len(sys.argv) > 1:
    os.execvp(sys.argv[1], sys.argv[1:])
device = os.getenv("VG_DEVICE", "cuda")
if device.startswith("cuda"):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Configure NVIDIA Container Toolkit and --gpus all; VG_DEVICE=cpu selects CPU mode.")
args = [sys.executable, "frontend/app.py", "--model-dir", "checkpoints/adain-vc-official",
        "--registry", "outputs/registry.npz", "--attack-registry", "outputs/registry_vc.npz",
        "--ecapa-dir", "checkpoints/ecapa-local", "--vctk", "data/VCTK22", "--device", device,
        "--server-name", "0.0.0.0", "--server-port", "7860"]
os.execv(sys.executable, args)
