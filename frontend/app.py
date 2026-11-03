"""Voice Guard Gradio 前端（一键保护 + 转换演示 + 指标展示）。

用法：
    python frontend/app.py --model-dir checkpoints/adain-vc-official \
        --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz \
        --vctk data/VCTK22   # 用于自动选取 least-likely 说话人的音频
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from vguard.pipeline import Pipeline  # noqa: E402


def _plot_waveforms(paths, out_png):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    labels = {"original": "original", "defended": "defended", "delta": "perturbation (x5)"}
    for key in ("original", "defended", "delta"):
        if key not in paths:
            continue
        w, sr = sf.read(paths[key])
        t = np.arange(len(w)) / sr
        if key == "delta":
            ax.plot(t, w * 5, lw=0.8, alpha=0.7, label=labels[key])
        else:
            ax.plot(t, w, lw=0.6, label=labels[key])
    ax.legend()
    ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def build_app(args):
    pipe = Pipeline(args.model_dir, args.registry, args.attack_registry,
                    ecapa_dir=args.ecapa_dir, device=args.device)

    # speaker -> first wav path (for auto least-likely audio)
    speaker_first = {}
    if args.vctk and os.path.isdir(args.vctk):
        for s in pipe.speakers:
            d = os.path.join(args.vctk, s)
            if os.path.isdir(d):
                wavs = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
                if wavs:
                    speaker_first[s] = os.path.join(d, wavs[0])

    with gr.Blocks(title="Voice Guard 语音隐私主动防御") as demo:
        gr.Markdown("# Voice Guard：语音隐私主动防御\n"
                    "复现 IJCAI 2023 *Voice Guard*（时域对抗扰动 + 心理声学掩蔽）。"
                    "上传音频 → 一键保护 → 试听/下载，并查看被保护后 VC 转换是否还能伪造声纹。")

        with gr.Tab("保护"):
            with gr.Row():
                defend_in = gr.Audio(label="要保护的声音（说话人 X）", type="filepath")
                speaker_dd = gr.Dropdown(choices=pipe.speakers, value=pipe.speakers[0],
                                         label="被保护说话人")
            adv_in = gr.Audio(label="扰动方向音频（least-likely 说话人 Y；留空则自动选）",
                              type="filepath")
            with gr.Row():
                eps_sl = gr.Slider(0.02, 0.3, value=0.1, step=0.01, label="ε（扰动强度）")
                demo_cb = gr.Checkbox(value=True, label="演示快速模式（300/200 步）")
            run_btn = gr.Button("一键保护", variant="primary")

            with gr.Row():
                aud_orig = gr.Audio(label="原始（trim 后）")
                aud_def = gr.Audio(label="被保护音频")
                aud_delta = gr.Audio(label="扰动本身")
            with gr.Row():
                aud_conv_clean = gr.Audio(label="干净 → VC 转换")
                aud_conv_def = gr.Audio(label="被保护 → VC 转换")
            plot_out = gr.Image(label="波形对比")
            report_out = gr.JSON(label="报告")

            def on_protect(defend, speaker, adv, eps, demo_mode):
                if not defend:
                    raise gr.Error("请先上传要保护的声音")
                if not adv:
                    y = pipe.least_likely(speaker)
                    if y in speaker_first:
                        adv = speaker_first[y]
                    else:
                        raise gr.Error("请提供 least-likely 目标音频")
                it1, it2 = (300, 200) if demo_mode else (args.iters1, args.iters2)
                out_dir = os.path.join(args.out_dir, f"{speaker}_{int(time.time())}")
                rep = pipe.protect(defend, speaker, adv, eps=eps, iters1=it1, iters2=it2,
                                   out_dir=out_dir, progress=True)
                png = _plot_waveforms(rep["files"], os.path.join(out_dir, "waves.png"))
                return (rep["files"]["original"], rep["files"]["defended"],
                        rep["files"]["delta"], rep["files"]["conv_clean"],
                        rep["files"]["conv_def"], png, rep)

            run_btn.click(on_protect,
                          [defend_in, speaker_dd, adv_in, eps_sl, demo_cb],
                          [aud_orig, aud_def, aud_delta, aud_conv_clean, aud_conv_def,
                           plot_out, report_out])

        with gr.Tab("转换演示"):
            src_in = gr.Audio(label="内容音频 t̂", type="filepath")
            tgt_in = gr.Audio(label="声纹音频 x̂", type="filepath")
            conv_btn = gr.Button("转换")
            conv_out = gr.Audio(label="F(t̂, x̂)")

            def on_convert(src, tgt):
                if not src or not tgt:
                    raise gr.Error("请上传两段音频")
                out = os.path.join(args.out_dir, f"conv_{int(time.time())}.wav")
                from vguard.vc.adainvc import convert as vc_convert
                vc_convert(pipe.model, pipe.config, pipe.attr, src, tgt, out, pipe.device)
                return out

            conv_btn.click(on_convert, [src_in, tgt_in], [conv_out])

        with gr.Tab("说明"):
            gr.Markdown(
                "**方法**：给 X 的音频注入时域扰动 δ=ε·tanh(w)，使零样本 VC 无法再伪造其声纹；"
                "扰动被约束在心理声学掩蔽阈值 θx 之下（人耳不可闻）。\n\n"
                "**判定**（Eq.9-11）：被保护音频仍被识别为 X 且转换后不再被识别为 X，视为防御成功。\n\n"
                "论文：Li et al., *Voice Guard*, IJCAI 2023, pp.4812-4820。")

    return demo


def _patch_gradio_client_schema():
    """gradio-client 1.3.0 对 JSON Schema 的 `additionalProperties: true` 未处理
    （bool schema 会进入 get_type 抛 TypeError / APIInfoParseError），导致 /info
    路由 500、launch 时 localhost 可达性检查失败。补丁：bool schema → "Any"。"""
    try:
        import gradio_client.utils as gcu

        if not getattr(gcu._json_schema_to_python_type, "_vg_patched", False):
            _orig_js2py = gcu._json_schema_to_python_type

            def safe_js2py(schema, defs):
                if isinstance(schema, bool) or schema == {}:
                    return "Any"
                return _orig_js2py(schema, defs)

            safe_js2py._vg_patched = True
            gcu._json_schema_to_python_type = safe_js2py

            _orig_get_type = gcu.get_type

            def safe_get_type(schema):
                if isinstance(schema, bool) or schema == {}:
                    return "Any"
                return _orig_get_type(schema)

            gcu.get_type = safe_get_type
    except Exception:  # noqa: BLE001
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--attack-registry", required=True)
    ap.add_argument("--vctk", default="")
    ap.add_argument("--ecapa-dir", default="checkpoints/ecapa-local")
    ap.add_argument("--iters1", type=int, default=3000)
    ap.add_argument("--iters2", type=int, default=1500)
    ap.add_argument("--out-dir", default="outputs/web")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--server-name", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=7860)
    args = ap.parse_args()

    _patch_gradio_client_schema()
    os.makedirs(args.out_dir, exist_ok=True)
    demo = build_app(args)
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
