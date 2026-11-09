"""L2 硬掩蔽投影评估：把扰动严格压到 θx 之下，量化"不可感知 ↔ 防御"代价。

对实验目录中每句：
  1. 读 original_trimmed / defended / delta，计算当前超阈率（soft 结果）
  2. 用 project_masked_delta_hard 迭代投影到不超阈（margin=0）
  3. 写出 defended_hard.wav；再算超阈率
  4. 可选 --evaluate：对 defended_hard 做 ECAPA 身份判定 + Whisper WER（MOSNet 需 --mos）
产物：<exp-dir>/hard_masking.json + 每句 defended_hard.wav

用法：
    python scripts/hard_masking_eval.py --exp-dir outputs/exp03_eps005 \
        --model-dir checkpoints/adain-vc-official --registry outputs/registry.npz \
        --max-iters 100 [--evaluate] [--device cuda]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from vguard.masking import (file2wav_mask, masking_exceedance,  # noqa: E402
                            project_masked_delta_hard, project_to_mask_exact)
from vguard.vc.adainvc import load_adainvc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--model-dir", default="checkpoints/adain-vc-official")
    ap.add_argument("--registry", default="outputs/registry.npz")
    ap.add_argument("--max-iters", type=int, default=100)
    ap.add_argument("--evaluate", action="store_true", help="对硬投影结果做 ECAPA 身份判定")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, config, attr = load_adainvc(args.model_dir, device=args.device)
    pp = config["preprocess"]
    sr = pp["sample_rate"]
    reg = np.load(args.registry, allow_pickle=True)
    names = list(reg["names"])
    ecapa = None
    if args.evaluate:
        from vguard.asv import embed_wav, load_ecapa
        ecapa = load_ecapa(savedir="checkpoints/ecapa-local")

    rows = []
    for d in sorted(os.listdir(args.exp_dir)):
        sub = os.path.join(args.exp_dir, d)
        if not os.path.isdir(sub):
            continue
        orig_p = os.path.join(sub, "original_trimmed.wav")
        def_p = os.path.join(sub, "defended.wav")
        if not (os.path.exists(orig_p) and os.path.exists(def_p)):
            continue
        target = d.split("_")[0]
        defended, _ = sf.read(def_p, dtype="float32")
        # 以 file2wav_mask 的 trim 结果为准（theta 与它同源，帧数必须一致）
        orig_t, theta, psd_max = file2wav_mask(orig_p, sample_rate=sr, n_fft=pp["n_fft"],
                                               hop_length=pp["hop_length"], win_length=pp["win_length"],
                                               top_db=pp["top_db"])
        L = len(orig_t)
        if len(defended) > L:
            defended = defended[:L]
        elif len(defended) < L:
            defended = np.pad(defended, (0, L - len(defended)))
        orig = orig_t.astype(np.float32)
        delta = (defended - orig).astype(np.float32)
        before = masking_exceedance(delta, theta, float(psd_max), pp["n_fft"],
                                    pp["hop_length"], pp["win_length"], margin_db=0.0)
        hard, diag = project_masked_delta_hard(delta, theta, float(psd_max), pp["n_fft"],
                                               pp["hop_length"], pp["win_length"],
                                               max_iters=args.max_iters, margin_db=0.0)
        after = masking_exceedance(hard, theta, float(psd_max), pp["n_fft"],
                                   pp["hop_length"], pp["win_length"], margin_db=0.0)
        snr_db = float(10 * np.log10(np.mean(orig ** 2) / (np.mean((defended - orig) ** 2) + 1e-19)))
        row = {"utterance": d, "target": target,
               "exceed_before": before["exceed_frac"], "exceed_after": after["exceed_frac"],
               "mean_excess_before": before["mean_excess_db"],
               "mean_excess_after": after["mean_excess_db"],
               "iters": diag["iters"], "snr_db": round(snr_db, 2)}
        sf.write(os.path.join(sub, "defended_hard.wav"), (orig + hard).astype(np.float32), sr)
        # 严格版：一次性缩放到每格都不超阈（代价 = 扰动变弱）
        exact, ex_diag = project_to_mask_exact(delta, theta, float(psd_max), pp["n_fft"],
                                               pp["hop_length"], pp["win_length"], margin_db=0.0)
        sf.write(os.path.join(sub, "defended_exact.wav"), (orig + exact).astype(np.float32), sr)
        snr_exact = float(10 * np.log10(np.mean(orig ** 2) / (np.mean(exact ** 2) + 1e-19)))
        row["scale_exact"] = ex_diag.get("scale")
        row["snr_db_exact"] = round(snr_exact, 2)
        row["exceed_exact"] = ex_diag.get("exceed_frac")
        if ecapa is not None:
            from vguard.asv import embed_wav
            from vguard.vc.adainvc import convert as vc_convert
            src = os.path.join(sub, "source.wav")
            for tag, w in (("hard", (orig + hard)), ("exact", (orig + exact))):
                e = embed_wav(ecapa, w.astype(np.float32), sr)
                sims = np.asarray([float(np.dot(e, x) / (np.linalg.norm(e) * np.linalg.norm(x) + 1e-9))
                                   for x in reg["embeddings"]])
                pred = names[int(np.argmax(sims))]
                row[f"asv_{tag}"] = pred
                row[f"fimp_{tag}"] = bool(pred == target)
                # F_str：转换后的身份必须不再是目标说话人
                if os.path.exists(src):
                    tmp_in = os.path.join(sub, f"defended_{tag}_tmp.wav")
                    tmp_out = os.path.join(sub, f"conv_def_{tag}.wav")
                    sf.write(tmp_in, w.astype(np.float32), sr)
                    vc_convert(model, config, attr, src, tmp_in, tmp_out, args.device)
                    wc, _ = sf.read(tmp_out, dtype="float32")
                    ec = embed_wav(ecapa, wc.astype(np.float32), sr)
                    sims2 = np.asarray([float(np.dot(ec, x) / (np.linalg.norm(ec) * np.linalg.norm(x) + 1e-9))
                                        for x in reg["embeddings"]])
                    pred2 = names[int(np.argmax(sims2))]
                    row[f"conv_{tag}"] = pred2
                    row[f"fstr_{tag}"] = bool(pred2 != target)
                    row[f"defense_{tag}"] = bool(row[f"fimp_{tag}"] and row[f"fstr_{tag}"])
        rows.append(row)
        print(row, flush=True)

    out = os.path.join(args.exp_dir, "hard_masking.json")
    Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    n = len(rows)
    if n:
        print(f"\n=== 汇总（{n} 句）===")
        print(f"超阈率均值：before {np.mean([r['exceed_before'] for r in rows]):.4f} "
              f"-> after {np.mean([r['exceed_after'] for r in rows]):.4f}")
        if ecapa is not None:
            print(f"硬投影后 F_imp 通过：{sum(1 for r in rows if r.get('fimp_hard'))}/{n}")
    print("saved:", out)


if __name__ == "__main__":
    main()
