"""组装华为云部署精简包（避免上传 12 GB 全量 VCTK）。

产物（默认 deploy/cloud_bundle/）：
    vguard/ scripts/ tools/ frontend/ tests/ requirements.txt
    deploy/Dockerfile
    checkpoints/                 模型权重（约 129 MB）
    outputs/registry*.npz        注册库（约 30 KB）
    data/VCTK22/<spk>/<首句>.wav   每个注册说话人 1 句（约 4 MB，供 least-likely 自动选取 + 样例按钮）
    CLOUD_README.md              包内启动说明

用法：
    python tools/make_cloud_bundle.py                    # 生成目录
    python tools/make_cloud_bundle.py --zip              # 额外打 zip（便于 scp 上传）
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DIRS = ["vguard", "scripts", "tools", "frontend", "tests"]
FILES = ["requirements.txt", os.path.join("deploy", "Dockerfile")]
WEIGHTS = os.path.join("checkpoints")
REGISTRY = ["registry.npz", "registry_vc.npz"]

CLOUD_README = """# 云端部署包（Voice Guard）

本包为部署到华为云用的**精简副本**：权重 + 注册库 + 每个说话人 1 句演示音频（总计约 135 MB）。

## 方式 1：直接运行（无需 Docker，最快）

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -i https://repo.huaweicloud.com/repository/pypi/simple/ \\
  torch==2.5.1 torchaudio==2.5.1 numpy scipy librosa soundfile pyyaml tqdm matplotlib \\
  "gradio==4.44.1" "gradio-client==1.3.0" "fastapi==0.112.4" "starlette==0.37.2" "jinja2==3.1.2" \\
  uvicorn python-multipart speechbrain "huggingface-hub<0.30"
MPLCONFIGDIR=/tmp/mpl nohup python frontend/app.py \\
  --model-dir checkpoints/adain-vc-official \\
  --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz \\
  --vctk data/VCTK22 --device cuda \\
  --server-name 0.0.0.0 --server-port 7860 > app.log 2>&1 &
```

（无 GPU 时把 `--device cuda` 改为 `--device cpu`；页面里保持「演示快速模式」）

## 方式 2：Docker（构建后推 SWR 给 ModelArts 用）

```bash
docker build -f deploy/Dockerfile -t voiceguard:latest .
docker run -d --gpus all -p 7860:7860 voiceguard:latest   # 本地验证
```

## 健康检查

```bash
curl -I http://127.0.0.1:7860/    # 期望 HTTP/1.1 200
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "deploy", "cloud_bundle"))
    ap.add_argument("--zip", action="store_true", help="额外打包 zip")
    args = ap.parse_args()
    out = os.path.abspath(args.out)

    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    for d in DIRS:
        src = os.path.join(ROOT, d)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(out, d),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in FILES:
        src = os.path.join(ROOT, f)
        if os.path.isfile(src):
            dst = os.path.join(out, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    # 权重
    shutil.copytree(os.path.join(ROOT, WEIGHTS), os.path.join(out, "checkpoints"))
    os.makedirs(os.path.join(out, "outputs"), exist_ok=True)
    for r in REGISTRY:
        p = os.path.join(ROOT, "outputs", r)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(out, "outputs", r))

    # 精简 VCTK：每个注册说话人取 1 句
    import numpy as np
    reg = np.load(os.path.join(ROOT, "outputs", "registry.npz"), allow_pickle=True)
    speakers = list(reg["names"])
    kept = 0
    for spk in speakers:
        d = os.path.join(ROOT, "data", "VCTK22", spk)
        if not os.path.isdir(d):
            continue
        wavs = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        if not wavs:
            continue
        dst_dir = os.path.join(out, "data", "VCTK22", spk)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(os.path.join(d, wavs[0]), os.path.join(dst_dir, wavs[0]))
        kept += 1

    with open(os.path.join(out, "CLOUD_README.md"), "w", encoding="utf-8") as f:
        f.write(CLOUD_README)

    total = sum(os.path.getsize(os.path.join(dp, fn))
                for dp, _, fns in os.walk(out) for fn in fns)
    print(f"bundle: {out}")
    print(f"  speakers: {kept}/{len(speakers)} 首句")
    print(f"  size: {total / 1024 / 1024:.1f} MB")

    if args.zip:
        zpath = shutil.make_archive(out, "zip", root_dir=out)
        print(f"  zip: {zpath}  ({os.path.getsize(zpath) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
