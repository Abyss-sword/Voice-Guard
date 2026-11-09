"""C17 监听网页：把匿名试听材料部署成网页，收集同学评分（人工听测落地）。

用法（在 ECS 或本机）：
    python tools/listening_app.py --items outputs/listening_items.json --out outputs/listening/results.jsonl

items JSON 格式（无则先在 --items 指向的路径生成示例）：
    [{"id": "p225_040", "original": "a.wav", "defended": "b.wav",
      "conv_clean": "c.wav", "conv_def": "d.wav"}, ...]

页面：每个条目随机化 A/B/C/D 顺序（避免位置偏差），用户对每段打 1–5 分并给整体偏好，
提交后追加写入 JSONL（含匿名会话 id 与随机化映射，便于事后还原）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid

import gradio as gr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_items(path: str):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        example = os.path.join(ROOT, "platform", "samples", "group_xx_voice_guard")
        items = [{
            "id": "sample",
            "original": os.path.join(example, "input.wav"),
            "defended": os.path.join(example, "output_defended.wav"),
            "conv_clean": os.path.join(example, "output_converted.wav"),
            "conv_def": os.path.join(example, "output_defended.wav"),
        }]
        json.dump(items, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[listening] 未找到 {path}，已生成示例（请替换为真实试听材料）")
        return items
    return json.load(open(path, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default="outputs/listening_items.json")
    ap.add_argument("--out", default="outputs/listening/results.jsonl")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()

    items = load_items(args.items)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def build():
        rnd = random.Random()
        slots = {}
        for it in items:
            order = [("original", it["original"]), ("defended", it["defended"]),
                     ("conv_clean", it["conv_clean"]), ("conv_def", it["conv_def"])]
            rnd.shuffle(order)
            slots[it["id"]] = order
        return slots

    def make_ui(slots):
        comps = []
        with gr.Blocks(title="Voice Guard 匿名试听") as demo:
            gr.Markdown("# Voice Guard 匿名试听\n\n每段音频 1–5 分（5=音质最好），"
                        "最后选择你更愿意接受的一段。顺序已随机化，请勿猜测。")
            sid = gr.State(uuid.uuid4().hex[:10])
            ratings = {}
            for it in items:
                slots_ = slots[it["id"]]
                with gr.Group():
                    gr.Markdown(f"**条目 {it['id']}**")
                    for idx, (kind, path) in enumerate(slots_):
                        letter = "ABCD"[idx]
                        gr.Audio(value=path, interactive=False, label=f"{letter}")
                        ratings[it["id"], letter] = gr.Slider(1, 5, value=3, step=1,
                                                             label=f"{letter} 评分")
            pref = gr.Radio(choices=["A", "B", "C", "D"], value="A", label="整体更愿意接受哪一段")

            def submit(state, pref_val, *vals):
                rec = {"ts": time.time(), "session": state, "preferred": pref_val,
                       "items": [{"id": it["id"],
                                  "order": [{"letter": "ABCD"[i], "kind": k}
                                            for i, (k, _) in enumerate(slots[it["id"]])],
                                  "ratings": {f"ABCD"[i]: int(vals[4 * j + i])
                                              for i in range(4)}}
                                 for j, it in enumerate(items)]}
                with open(args.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                return "已记录，感谢！"
            out = gr.Markdown()
            gr.Button("提交").click(submit, [sid, pref, *ratings.values()], out)
        return demo

    demo = make_ui(build())
    demo.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
