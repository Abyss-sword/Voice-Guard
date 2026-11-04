"""AdaIN-VC 转换 CLI（干净/被保护音频 -> 转换音频），复现与案例演示用。

用法：
    python scripts/convert.py --model-dir checkpoints/adain-vc-official \
        --source <内容音频 t> --target <声纹音频 x 或 defended> --out <输出 wav>
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vguard.vc.adainvc import convert, load_adainvc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, config, attr = load_adainvc(args.model_dir, device=args.device)
    convert(model, config, attr, args.source, args.target, args.out, device=args.device)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
