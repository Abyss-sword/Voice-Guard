"""C9 独立评估器：用 ModelScope 中文声纹模型（3D-Speaker ERes2Net）做交叉验证。

动机：主评估用 SpeechBrain ECAPA（VoxCeleb）与 Resemblyzer GE2E；再引入一个
**中文语料训练**的声纹模型，可检验"防御率"结论是否依赖单一评估器。

前置：
    pip install modelscope          # 国内下载较快；首次运行会拉取模型权重
    # 也可用 huggingface 镜像（见 --source）

用法：
    # 单文件打分（打印 192 维嵌入的范数与跨文件相似度矩阵）
    python tools/asv_modelscope.py --audios a.wav b.wav --device cuda
    # 指定模型 / 输出 JSON
    python tools/asv_modelscope.py --audios a.wav b.wav --out outputs/asv_ms.json

说明：本脚本只做"两两相似度"与嵌入，不内置注册库评估；若要与 ECAPA 口径对接，
用 --registry 传入 registry.npz（names/embeddings），即可输出每个音频最近邻说话人。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

DEFAULT_MODEL = "iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audios", nargs="+", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--registry", default="", help="可选 registry.npz（names/embeddings）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
    except Exception as e:  # noqa: BLE001
        sys.exit(f"需要先安装 modelscope：pip install modelscope（{e}）")

    sv = pipeline(task=Tasks.speaker_verification, model=args.model, device=args.device)
    embs = {}
    for p in args.audios:
        if not os.path.exists(p):
            sys.exit(f"音频不存在：{p}")
        out = sv(p)  # 不同版本返回 (emb, sample_rate) 或 dict
        emb = out[0] if isinstance(out, tuple) else out.get("embeddings", out)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        embs[p] = emb
        print(f"{os.path.basename(p):<28} emb_dim={emb.shape[0]} l2={np.linalg.norm(emb):.3f}")

    paths = list(embs)
    sim = np.zeros((len(paths), len(paths)))
    for i, a in enumerate(paths):
        for j, b in enumerate(paths):
            va, vb = embs[a], embs[b]
            sim[i, j] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
    print("\n相似度矩阵：")
    print("            " + " ".join(f"{os.path.basename(p)[:10]:>12}" for p in paths))
    for i, a in enumerate(paths):
        print(f"{os.path.basename(a)[:10]:<12}" + " ".join(f"{sim[i, j]:>12.3f}" for j in range(len(paths))))

    nearest = {}
    if args.registry and os.path.exists(args.registry):
        reg = np.load(args.registry, allow_pickle=True)
        names, E = list(reg["names"]), np.asarray(reg["embeddings"], dtype=np.float32)
        # 维度不同（ECAPA 192 vs ERes2Net 192 可能不同）时跳过
        for p, e in embs.items():
            if e.shape[0] != E.shape[1]:
                print(f"[warn] {p} 嵌入维度 {e.shape[0]} 与注册库 {E.shape[1]} 不一致，跳过最近邻")
                continue
            s = E @ e / (np.linalg.norm(E, axis=1) * np.linalg.norm(e) + 1e-9)
            nearest[os.path.basename(p)] = {"pred": names[int(np.argmax(s))],
                                           "sim": round(float(np.max(s)), 4)}
            print(f"最近邻: {os.path.basename(p)} -> {nearest[os.path.basename(p)]['pred']} "
                  f"(sim {nearest[os.path.basename(p)]['sim']})")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        json.dump({"model": args.model, "audios": paths,
                   "emb_dim": {os.path.basename(p): int(e.shape[0]) for p, e in embs.items()},
                   "similarity": sim.tolist(), "nearest": nearest},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("saved:", args.out)


if __name__ == "__main__":
    main()
