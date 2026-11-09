"""打包"权重与注册库"分发包（上传云盘用）。

产物：release/voiceguard_weights_<version>.zip
    内含 weights/（checkpoints 全量）+ registry/（outputs/*.npz）+ samples/（平台样例）
    + README-weights.txt（说明如何放置）
用法：
    python tools/make_weights_package.py                # 默认 v1.0
    python tools/make_weights_package.py --version v1.1
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEXT = """# Voice Guard 权重与注册库（分发包）

解压后把内容放回源码仓库对应位置即可运行：

    weights/  -> 仓库根目录 checkpoints/
    registry/ -> 仓库根目录 outputs/（registry.npz / registry_vc.npz / asv_probe.npz）
    samples/  -> 可选，公开样例音频（platform/samples/）

## 校验

本包 SHA256 见发布页说明。解压后核对：

    certutil -hashfile voiceguard_weights_<version>.zip SHA256     # Windows
    sha256sum voiceguard_weights_<version>.zip                     # Linux

## 内容说明

- checkpoints/adain-vc-official/：官方 AdaIN-VC（NTU 直链版本），24kHz / 512 mel
- checkpoints/ecapa-local/：SpeechBrain ECAPA-TDNN（VoxCeleb），独立评估 ASV
- checkpoints/mosnet-hf/：MOSNet（HF 移植，safetensors 版）
- outputs/registry*.npz：20 位 VCTK 说话人注册库（Es 128 维 / ECAPA 192 维）
- 数据（VCTK22 全量 10.69GB）不入此包；用 `tools/download_models.py vctk --extract`
  下载，或使用仓库内演示音频子集。

## 参考

论文：Li et al., Voice Guard, IJCAI 2023, pp.4812-4820
代码：https://github.com/Abyss-sword/Voice-Guard
"""


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "release"))
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, f"voiceguard_weights_{args.version}.zip")

    ckpt = os.path.join(ROOT, "checkpoints")
    if not os.path.isdir(ckpt):
        sys.exit(f"checkpoints/ 不存在：{ckpt}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 权重
        for dp, _, fns in os.walk(ckpt):
            for fn in fns:
                src = os.path.join(dp, fn)
                rel = os.path.relpath(src, ckpt)
                zf.write(src, os.path.join("weights", rel))
        # 注册库与探头
        for name in ("registry.npz", "registry_vc.npz", "asv_probe.npz"):
            p = os.path.join(ROOT, "outputs", name)
            if os.path.isfile(p):
                zf.write(p, os.path.join("registry", name))
        # 样例
        smp = os.path.join(ROOT, "platform", "samples")
        if os.path.isdir(smp):
            for dp, _, fns in os.walk(smp):
                for fn in fns:
                    src = os.path.join(dp, fn)
                    zf.write(src, os.path.join("samples", os.path.relpath(src, smp)))
        zf.writestr("README-weights.txt", TEXT.replace("<version>", args.version))

    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    digest = sha256_of(zip_path)
    print(f"package : {zip_path}")
    print(f"size    : {size_mb:.1f} MB")
    print(f"sha256  : {digest}")


if __name__ == "__main__":
    main()
