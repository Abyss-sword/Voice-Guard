"""人工听测页面（② 主观 MOS 听测）。

数据驱动：读取 tools/build_listening_items.py 生成的 items.json，
每句的音频在页面上随机映射为 A/B/C…（避免位置/标签偏差），
听众对每段打 1–5 分并选择"最愿意接受"的一段；提交追加写入 JSONL。

用法：
    python tools/listening_app.py --items outputs/listening/items.json \
        --out outputs/listening/results.jsonl --port 7861
    # 局域网分享：--host 0.0.0.0（需自行确保网络/权限）

提交记录字段：ts / session / preferred / items[{id, order[{letter,kind}], ratings{letter:score}}]
统计：python tools/summarize_listening.py --results outputs/listening/results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_order(clips, rng):
    order = list(clips)
    rng.shuffle(order)
    letters = ["ABCDEFGH"[:i + 1][-1] for i in range(len(order))]
    return [{"letter": letters[i], "kind": c["kind"], "path": c["path"]} for i, c in enumerate(order)]


def record_submission(path: str, session: str, preferred: str, items: list, orders: dict, ratings: dict):
    rec = {"ts": time.time(), "session": session, "preferred": preferred,
           "items": [{"id": it_id,
                      "order": [{"letter": o["letter"], "kind": o["kind"]} for o in orders[it_id]],
                      "ratings": {o["letter"]: int(ratings[(it_id, o["letter"])]) for o in orders[it_id]}}
                     for it_id in orders]}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default="outputs/listening/items.json")
    ap.add_argument("--out", default="outputs/listening/results.jsonl")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--title", default="Voice Guard 匿名听测")
    args = ap.parse_args()

    items_path = ROOT / args.items if not os.path.isabs(args.items) else Path(args.items)
    if not items_path.exists():
        raise SystemExit(f"缺少 {items_path}，先运行 tools/build_listening_items.py")
    items = json.loads(items_path.read_text(encoding="utf-8"))["items"]

    import gradio as gr

    rng = random.Random()
    orders = {it["id"]: make_order(it["clips"], rng) for it in items}

    with gr.Blocks(title=args.title) as demo:
        gr.Markdown(f"# {args.title}\n\n每段音频请按 **1–5 分**评分（5 = 听起来最自然、最像真人原声）。"
                    "顺序与标签已随机化，请勿猜测。全部打分后选择你**最愿意接受**的一段并提交。")
        sid = gr.State(uuid.uuid4().hex[:10])
        ratings = {}
        with gr.Column():
            for it in items:
                gr.Markdown(f"**句子 {it['id']}**")
                for o in orders[it["id"]]:
                    gr.Audio(value=o["path"], interactive=False,
                             label=f"{o['letter']}（句子 {it['id']}）")
                    ratings[(it["id"], o["letter"])] = gr.Slider(
                        1, 5, value=3, step=1, label=f"{o['letter']} 评分")
        pref = gr.Radio(choices=["A", "B", "C", "D"], value="A",
                        label="最愿意接受哪一段（跨所有句子给你的整体偏好）")
        out = gr.Markdown()

        def submit(state, pref_val, *vals):
            flat = list(vals)
            idx = 0
            got = {}
            for it in items:
                for o in orders[it["id"]]:
                    got[(it["id"], o["letter"])] = flat[idx]
                    idx += 1
            record_submission(args.out, state, pref_val, items, orders, got)
            return "✅ 已记录，感谢参与！"

        gr.Button("提交评分", variant="primary").click(submit, [sid, pref, *ratings.values()], out)

    demo.launch(server_name=args.host, server_port=args.port, show_api=False)


if __name__ == "__main__":
    main()
