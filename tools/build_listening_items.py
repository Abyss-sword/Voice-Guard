"""构建人工听测材料（② 主观 MOS 听测）。

从本地实验目录生成 items.json 与答案键（answer key），供 tools/listening_app.py 使用。

默认设计：5 句 × 3 条件（原始裁剪 / VG ε=0.02 / VG ε=0.05）= 15 段音频。
参与者页面把每句的 3 段随机映射为 A/B/C，打 1–5 分并选择"最愿意接受"的一段；
答案键（每个 item 的 kind→文件）只保存给研究者，不进入参与者包。

用法：
    python tools/build_listening_items.py \
        --exp-dirs "outputs/exp03_eps002:epsilon=0.02,outputs/exp03_eps005:epsilon=0.05" \
        --out outputs/listening/items.json --key outputs/listening/answer-key.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_exp_dirs(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        path, _, label = part.partition(":")
        out.append((path.strip(), label.strip() or os.path.basename(path.strip())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-dirs", default="outputs/exp03_eps002:epsilon=0.02,outputs/exp03_eps005:epsilon=0.05")
    ap.add_argument("--include-original", action="store_true", default=True)
    ap.add_argument("--out", default="outputs/listening/items.json")
    ap.add_argument("--key", default="outputs/listening/answer-key.json")
    ap.add_argument("--copy-audio", action="store_true",
                    help="把音频复制到 outputs/listening/audio（便于整体打包分发）")
    args = ap.parse_args()

    exps = [(ROOT / p, label) for p, label in parse_exp_dirs(args.exp_dirs)]
    for d, _ in exps:
        if not d.is_dir():
            sys.exit(f"实验目录不存在：{d}")

    # 以第一个实验目录的句子列表为基准
    utts = sorted(d.name for d in exps[0][0].iterdir() if d.is_dir())
    if not utts:
        sys.exit(f"{exps[0][0]} 下没有句子目录")

    audio_dir = ROOT / "outputs" / "listening" / "audio"
    if args.copy_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    items, key = [], {"exps": [{"dir": str(d.relative_to(ROOT)), "label": l} for d, l in exps],
                      "items": {}}
    for utt in utts:
        clips = []
        orig = exps[0][0] / utt / "original_trimmed.wav"
        if args.include_original and orig.exists():
            src = orig
            if args.copy_audio:
                dst = audio_dir / f"{utt}_original.wav"
                import shutil
                shutil.copy2(src, dst)
                src = dst
            clips.append({"kind": "original", "path": str(src)})
        for d, label in exps:
            f = d / utt / "defended.wav"
            if not f.exists():
                print(f"[warn] 缺少 {f}，跳过")
                continue
            src = f
            if args.copy_audio:
                dst = audio_dir / f"{utt}_{label.replace('=','')}.wav"
                import shutil
                shutil.copy2(src, dst)
                src = dst
            clips.append({"kind": label, "path": str(src)})
        if len(clips) < 2:
            continue
        items.append({"id": utt, "clips": clips})
        key["items"][utt] = [{"kind": c["kind"], "path": str(Path(c["path"]).relative_to(ROOT))
                              if Path(c["path"]).is_absolute() and str(Path(c["path"])).startswith(str(ROOT))
                              else c["path"]} for c in clips]

    for p, payload in ((args.out, {"design": "per-item randomized A/B/C; 1-5 MOS; preference",
                                   "items": items}), (args.key, key)):
        dst = ROOT / p if not os.path.isabs(p) else Path(p)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("saved:", dst)

    print(f"句子数: {len(items)}；条件数/句: {len(items[0]['clips']) if items else 0}；"
          f"总音频: {sum(len(i['clips']) for i in items)}")


if __name__ == "__main__":
    main()
