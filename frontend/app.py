"""Voice Guard Gradio 前端（一键保护 + 转换演示 + 说明）。

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

CSS = """
.vg-hero {
  background: linear-gradient(135deg, #4f46e5 0%, #0891b2 100%);
  border-radius: 16px; padding: 28px 32px; color: #fff; margin-bottom: 18px;
}
.vg-hero h2 { color: #fff; margin: 0 0 8px 0; font-size: 26px; }
.vg-hero p { margin: 4px 0; opacity: .92; font-size: 14.5px; }
.vg-badges { margin-top: 12px; }
.vg-badge {
  display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
  border-radius: 999px; padding: 4px 14px; margin-right: 10px; font-size: 13px;
}
.vg-steps { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
.vg-step {
  flex: 1; min-width: 150px; background: #eef2ff; border: 1px solid #c7d2fe;
  border-radius: 10px; padding: 10px 14px; font-size: 13.5px; color: #3730a3;
}
.vg-step b { display: block; font-size: 15px; margin-bottom: 2px; }
.vg-status-ok {
  background: #ecfdf5; border: 1.5px solid #34d399; border-radius: 12px;
  padding: 16px 20px; font-size: 16px; color: #065f46;
}
.vg-status-fail {
  background: #fef2f2; border: 1.5px solid #f87171; border-radius: 12px;
  padding: 16px 20px; font-size: 16px; color: #991b1b;
}
.vg-metrics { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
.vg-metric {
  flex: 1; min-width: 110px; background: #f8fafc; border: 1px solid #e2e8f0;
  border-radius: 10px; padding: 10px 12px; text-align: center;
}
.vg-metric .k { font-size: 12px; color: #64748b; }
.vg-metric .v { font-size: 18px; font-weight: 600; color: #0f172a; margin-top: 2px; }
.vg-note { font-size: 13px; color: #64748b; margin-top: 6px; }
footer { display: none !important; }
"""


def _fmt_sim(report: dict, speaker: str) -> str:
    asv = report.get("asv", {})
    return asv.get(speaker, {}).get("sim", {})


def _status_html(report: dict) -> str:
    tgt = report["target_speaker"]
    asv = report.get("asv", {})
    sim_def = asv.get("defended", {}).get("sim", {}).get(tgt, None)
    sim_conv = asv.get("conv_def", {}).get("sim", {}).get(tgt, None)
    pred_def = asv.get("defended", {}).get("pred", "-")
    pred_conv = asv.get("conv_def", {}).get("pred", "-")

    imp = pred_def == tgt
    str_ = pred_conv != tgt
    ok = imp and str_

    if ok:
        head = "✅ 防御成功"
    elif imp:
        head = "⚠️ 部分成功：转换后仍可能被识别为 " + pred_conv
    else:
        head = "❌ 防御未达标：被保护音频被判为 " + pred_def

    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in [
            ("被保护音频判定", pred_def + (f"（相似度 {sim_def:.3f}）" if sim_def is not None else "")),
            ("被保护→VC 转换判定", pred_conv + (f"（相似度 {sim_conv:.3f}）" if sim_conv is not None else "")),
            ("判定式", "F_imp ∧ F_str（Eq.9-11，ECAPA 口径）"),
        ]
    )
    cls = "vg-status-ok" if ok else "vg-status-fail"
    return (f'<div class="{cls}"><b>{head}</b>'
            f'<table style="margin-top:10px;font-size:14px;border-collapse:collapse;">{rows}</table></div>')


def _metrics_html(report: dict, mcd: float | None) -> str:
    items = [
        ("ε", f"{report['eps']:.2f}"),
        ("c（掩蔽权重）", f"{report['c_final']}"),
        ("MCD", f"{mcd:.1f}" if mcd is not None else "—"),
        ("音频时长", f"{report['duration_s']:.1f}s"),
        ("耗时", f"{report['elapsed_s']:.0f}s"),
        ("迭代", f"{report['iters1']}+{report['iters2']}"),
    ]
    cells = "".join(f'<div class="vg-metric"><div class="k">{k}</div><div class="v">{v}</div></div>'
                    for k, v in items)
    return f'<div class="vg-metrics">{cells}</div>'


def _report_table_html(report: dict) -> str:
    asv = report.get("asv", {})
    sims = asv.get("defended", {}).get("sim", {})
    top = sorted(sims.items(), key=lambda kv: -kv[1])[:5] if sims else []
    top_rows = "".join(f"<tr><td>{n}</td><td>{s:.3f}</td></tr>" for n, s in top)
    return (f'<div style="font-size:13.5px;"><b>defended 音频的 top-5 相似说话人（ECAPA）</b>'
            f'<table style="margin-top:8px;border-collapse:collapse;">'
            f'<tr><th style="padding:2px 12px 2px 0;">说话人</th><th>相似度</th></tr>{top_rows}</table></div>')


def _plot_waveforms(paths, out_png):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_facecolor("#f8fafc")
    for key, label, style in (("original", "original", dict(lw=0.7, color="#0f172a")),
                              ("defended", "defended", dict(lw=0.7, color="#4f46e5", alpha=0.9)),
                              ("delta", "perturbation (×5)", dict(lw=0.8, color="#dc2626", alpha=0.65))):
        if key not in paths:
            continue
        w, sr = sf.read(paths[key])
        t = np.arange(len(w)) / sr
        ax.plot(t, w * (5 if key == "delta" else 1), label=label, **style)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_title("波形对比", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def build_app(args):
    pipe = Pipeline(args.model_dir, args.registry, args.attack_registry,
                    ecapa_dir=args.ecapa_dir, device=args.device)

    speaker_first = {}
    if args.vctk and os.path.isdir(args.vctk):
        for s in pipe.speakers:
            d = os.path.join(args.vctk, s)
            if os.path.isdir(d):
                wavs = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
                if wavs:
                    speaker_first[s] = os.path.join(d, wavs[0])

    theme = gr.themes.Soft(primary_hue=gr.themes.colors.indigo,
                           secondary_hue=gr.themes.colors.cyan)
    curve_img = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                             "docs", "fig_eps_curve.png"))

    with gr.Blocks(title="Voice Guard 语音隐私主动防御", theme=theme, css=CSS) as demo:
        gr.HTML(
            '<div class="vg-hero">'
            "<h2>🛡️ Voice Guard · 语音隐私主动防御</h2>"
            "<p>复现 IJCAI 2023《Voice Guard》：在发布前给声音注入<b>人耳不可感知</b>的时域对抗扰动，"
            "使零样本语音转换（AdaIN-VC）无法再伪造你的声纹。</p>"
            '<div class="vg-badges">'
            '<span class="vg-badge">⏱ 时域扰动</span>'
            '<span class="vg-badge">👂 心理声学掩蔽</span>'
            '<span class="vg-badge">🧪 身份+质量+内容 三维判定</span>'
            "</div></div>")

        with gr.Tab("🛡️ 一键保护"):
            gr.HTML(
                '<div class="vg-steps">'
                '<div class="vg-step"><b>① 输入</b>上传要保护的声音，选择说话人</div>'
                '<div class="vg-step"><b>② 参数</b>扰动强度 ε（推荐 0.05），演示模式更快</div>'
                '<div class="vg-step"><b>③ 执行</b>点击「一键保护」，GPU 优化 2–5 分钟</div>'
                '<div class="vg-step"><b>④ 验证</b>试听并查看 VC 转换是否失效</div>'
                "</div>")

            with gr.Group():
                gr.Markdown("### ① 输入")
                with gr.Row():
                    defend_in = gr.Audio(label="🎙 要保护的声音（说话人 X）", type="filepath")
                    adv_in = gr.Audio(label="🎯 扰动方向音频（least-likely 说话人 Y；留空自动选取）",
                                      type="filepath")
                speaker_dd = gr.Dropdown(choices=pipe.speakers, value=pipe.speakers[0],
                                         label="被保护说话人（注册库内）")

            with gr.Group():
                gr.Markdown("### ② 参数")
                with gr.Row():
                    eps_sl = gr.Slider(0.02, 0.3, value=0.05, step=0.01,
                                       label="ε（扰动强度）",
                                       info="0.05 为本实验最优工作点；越小越不可闻，越大防御越强")
                    demo_cb = gr.Checkbox(value=True, label="⚡ 演示快速模式（300/200 步）",
                                          info="关闭则用完整 3000/1500 步（更慢更稳）")

            with gr.Group():
                gr.Markdown("### ③ 执行")
                run_btn = gr.Button("🛡️ 一键保护", variant="primary", size="lg")

            with gr.Group():
                gr.Markdown("### ④ 验证结果")
                status_html = gr.HTML(value='<div class="vg-note">运行后这里显示防御判定。</div>')
                metrics_html = gr.HTML()
                with gr.Tabs():
                    with gr.Tab("🎧 试听"):
                        with gr.Row():
                            aud_orig = gr.Audio(label="原始（trim 后）")
                            aud_def = gr.Audio(label="被保护音频")
                        with gr.Row():
                            aud_delta = gr.Audio(label="扰动本身")
                            aud_conv_clean = gr.Audio(label="干净 → VC 转换")
                            aud_conv_def = gr.Audio(label="被保护 → VC 转换")
                    with gr.Tab("📈 波形"):
                        plot_out = gr.Image(label="波形对比")
                    with gr.Tab("📋 详细"):
                        detail_html = gr.HTML()

            def on_protect(defend, speaker, adv, eps, demo_mode):
                if not defend:
                    raise gr.Error("请先上传要保护的声音")
                if not adv:
                    y = pipe.least_likely(speaker)
                    if y in speaker_first:
                        adv = speaker_first[y]
                        gr.Info(f"已自动选取 least-likely 说话人 {y} 的音频")
                    else:
                        raise gr.Error("请提供 least-likely 目标音频")
                it1, it2 = (300, 200) if demo_mode else (args.iters1, args.iters2)
                out_dir = os.path.join(args.out_dir, f"{speaker}_{int(time.time())}")
                gr.Info(f"开始对抗优化（{it1}+{it2} 步，约 2–5 分钟）…")
                rep = pipe.protect(defend, speaker, adv, eps=eps, iters1=it1, iters2=it2,
                                   out_dir=out_dir, progress=False)

                mcd_val = None
                try:
                    from vguard.metrics.mcd import mcd_mel
                    w1, _ = sf.read(rep["files"]["original"])
                    w2, _ = sf.read(rep["files"]["defended"])
                    # 与批量实验统一口径：attr=None（零均值单位方差）
                    mcd_val = float(mcd_mel(w1, w2, pipe.pp, attr=None,
                                            device=pipe.device))
                except Exception:  # noqa: BLE001
                    pass

                png = _plot_waveforms(rep["files"], os.path.join(out_dir, "waves.png"))
                return (_status_html(rep), rep["files"]["original"], rep["files"]["defended"],
                        rep["files"]["delta"], rep["files"]["conv_clean"],
                        rep["files"]["conv_def"], png, _metrics_html(rep, mcd_val),
                        _report_table_html(rep))

            run_btn.click(on_protect,
                          [defend_in, speaker_dd, adv_in, eps_sl, demo_cb],
                          [status_html, aud_orig, aud_def, aud_delta, aud_conv_clean,
                           aud_conv_def, plot_out, metrics_html, detail_html])

            if speaker_first:
                ex = [[speaker_first[pipe.speakers[0]], pipe.speakers[0]]]
                gr.Examples(examples=ex, inputs=[defend_in, speaker_dd],
                            label="💡 试试 VCTK 样例（点击载入）")

        with gr.Tab("🔄 转换演示"):
            gr.Markdown("零样本语音转换演示：**内容**来自一段音频、**声纹**来自另一段——"
                        "这正是攻击者伪造声音的方式；保护页生成的扰动会让这一步失效。")
            with gr.Row():
                src_in = gr.Audio(label="📄 内容音频 t̂", type="filepath")
                tgt_in = gr.Audio(label="🗣 声纹音频 x̂", type="filepath")
            conv_btn = gr.Button("转换", variant="primary")
            conv_out = gr.Audio(label="F(t̂, x̂)")

            def on_convert(src, tgt):
                if not src or not tgt:
                    raise gr.Error("请上传两段音频")
                out = os.path.join(args.out_dir, f"conv_{int(time.time())}.wav")
                from vguard.vc.adainvc import convert as vc_convert
                vc_convert(pipe.model, pipe.config, pipe.attr, src, tgt, out, pipe.device)
                return out

            conv_btn.click(on_convert, [src_in, tgt_in], [conv_out])

        with gr.Tab("📖 说明"):
            gr.Markdown(
                "## 方法\n"
                "1. 给说话人 X 的音频注入时域扰动 δ = 2ε·tanh(w)；\n"
                "2. 优化目标：让 X 的音频经 VC 转换后不再被判为 X（攻击项），"
                "同时把扰动压在心理声学掩蔽阈值 θx 之下（掩蔽项，人耳不可闻）；\n"
                "3. 判定（Eq.9-11）：**F_imp**（被保护音频仍被识别为 X）∧ "
                "**F_str**（被保护后 VC 转换不再被识别为 X）⇒ 防御成功。\n\n"
                "## 我们的复现结论\n"
                "- ε 权衡曲线：防御率 0.02→75%、0.05→75%、0.08→50%、0.10→50%；\n"
                "- 方法对比：时域 VG 75% ≫ 频域基线 25%；\n"
                "- 真实信道（mp3/8k/G.711）分析：F_imp 是最薄弱一环。\n\n"
                "论文：Li et al., *Voice Guard: Protecting Voice Privacy with Strong and "
                "Imperceptible Adversarial Perturbation in the Time Domain*, IJCAI 2023, "
                "pp.4812–4820。\n"
                "仓库：[github.com/Abyss-sword/Voice-Guard](https://github.com/Abyss-sword/Voice-Guard)")
            if os.path.exists(curve_img):
                gr.Image(value=curve_img, label="ε 权衡曲线（复现结果）", height=420)

    return demo


def _patch_gradio_client_schema():
    """gradio-client 1.3.0 对 JSON Schema 的 `additionalProperties: true` 未处理
    （bool schema 会进入 get_type 抛 TypeError / APIInfoParseError），导致 /info
    路由 500、launch 时 localhost 可达性检查失败。补丁：bool schema → "Any"。
    同时保留文件类型识别：FileData schema（含 `path: str` 字段）必须仍被解析为
    "filepath"，否则客户端不会上传音频、只发路径字符串，服务端 preprocess 崩溃。"""
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
                if isinstance(schema, dict) and isinstance(schema.get("type"), list):
                    # JSON Schema 可空字段 {"type": ["string","null"]}：取非 null 类型
                    ts = [t for t in schema["type"] if t != "null"]
                    schema = {**schema, "type": (ts[0] if ts else "null")}
                return _orig_get_type(schema)

            gcu.get_type = safe_get_type

            _orig_jstpt = gcu.json_schema_to_python_type

            def safe_jstpt(schema):
                t = _orig_jstpt(schema)
                if isinstance(t, str) and "path: str" in t:
                    return "filepath"
                return t

            gcu.json_schema_to_python_type = safe_jstpt
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
