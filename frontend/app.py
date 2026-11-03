"""Voice Guard Gradio 前端 v3（一键保护 + 转换演示 + 说明）。

特性：科技感主题（白昼/黑夜一键切换，默认黑夜）、中/英双语切换、判定卡与指标卡。

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

# ---------------------------------------------------------------- 文案（双语）

T = {
    "zh": {
        "hero_title": "🛡️ Voice Guard · 语音隐私主动防御",
        "hero_sub": "复现 IJCAI 2023《Voice Guard》：发布前给声音注入<b>人耳不可感知</b>的时域对抗扰动，"
                    "使零样本语音转换（AdaIN-VC）无法再伪造你的声纹。",
        "badges": ["⏱ 时域扰动", "👂 心理声学掩蔽", "🧪 身份+质量+内容 三维判定"],
        "dark_label": "🌙 夜间模式",
        "lang_label": "语言 / Language",
        "tab_protect": "🛡️ 一键保护",
        "tab_convert": "🔄 转换演示",
        "tab_about": "📖 说明",
        "steps": [
            ("① 输入", "上传要保护的声音，选择说话人"),
            ("② 参数", "扰动强度 ε（推荐 0.05），演示模式更快"),
            ("③ 执行", "点击「一键保护」，GPU 优化 2–5 分钟"),
            ("④ 验证", "试听并查看 VC 转换是否失效"),
        ],
        "g1": "① 输入",
        "defend_label": "🎙 要保护的声音（说话人 X）",
        "adv_label": "🎯 扰动方向音频（least-likely 说话人 Y；留空自动选取）",
        "speaker_label": "被保护说话人（注册库内）",
        "g2": "② 参数",
        "eps_label": "ε（扰动强度）",
        "eps_info": "0.05 为本实验最优工作点；越小越不可闻，越大防御越强",
        "demo_label": "⚡ 演示快速模式（300/200 步）",
        "demo_info": "关闭则用完整 3000/1500 步（更慢更稳）",
        "g3": "③ 执行",
        "run_btn": "🛡️ 一键保护",
        "g4": "④ 验证结果",
        "status_placeholder": "运行后这里显示防御判定。",
        "t_try": "🎧 试听",
        "t_plot": "📈 波形",
        "t_detail": "📋 明细",
        "a_orig": "原始（trim 后）",
        "a_def": "被保护音频",
        "a_delta": "扰动本身",
        "a_cc": "干净 → VC 转换",
        "a_cd": "被保护 → VC 转换",
        "plot_label": "波形对比",
        "detail_title": "defended 音频的 top-5 相似说话人（ECAPA）",
        "detail_th": ["说话人", "相似度"],
        "metric_items": [("ε", "{:.2f}"), ("c（掩蔽权重）", "{}"), ("MCD", "{:.1f}"),
                         ("音频时长", "{:.1f}s"), ("耗时", "{:.0f}s"), ("迭代", "{}")],
        "ok_head": "✅ 防御成功",
        "part_head": "⚠️ 部分成功：转换后仍可能被识别为 ",
        "fail_head": "❌ 防御未达标：被保护音频被判为 ",
        "st_def": "被保护音频判定",
        "st_conv": "被保护→VC 转换判定",
        "st_eq": "判定式",
        "st_eq_v": "F_imp ∧ F_str（Eq.9-11，ECAPA 口径）",
        "err_no_defend": "请先上传要保护的声音",
        "err_need_adv": "请提供 least-likely 目标音频",
        "info_auto_adv": "已自动选取 least-likely 说话人 {} 的音频",
        "info_start": "开始对抗优化（{}+{} 步，约 2–5 分钟）…",
        "examples_label": "💡 试试 VCTK 样例（点击载入）",
        "conv_md": "零样本语音转换演示：**内容**来自一段音频、**声纹**来自另一段——"
                   "这正是攻击者伪造声音的方式；保护页生成的扰动会让这一步失效。",
        "src_label": "📄 内容音频 t̂",
        "tgt_label": "🗣 声纹音频 x̂",
        "conv_btn": "转换",
        "conv_out_label": "F(t̂, x̂)",
        "err_conv": "请上传两段音频",
        "about_md": "## 方法\n"
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
                    "仓库：[github.com/Abyss-sword/Voice-Guard](https://github.com/Abyss-sword/Voice-Guard)",
        "curve_label": "ε 权衡曲线（复现结果）",
    },
    "en": {
        "hero_title": "🛡️ Voice Guard · Proactive Voice Privacy Defense",
        "hero_sub": "Reproduction of IJCAI 2023 *Voice Guard*: inject <b>imperceptible</b> time-domain "
                    "adversarial perturbations before publishing, so zero-shot voice conversion "
                    "(AdaIN-VC) can no longer forge your voiceprint.",
        "badges": ["⏱ Time-domain", "👂 Psychoacoustic masking", "🧪 Identity+Quality+Content"],
        "dark_label": "🌙 Dark mode",
        "lang_label": "语言 / Language",
        "tab_protect": "🛡️ Protect",
        "tab_convert": "🔄 Conversion demo",
        "tab_about": "📖 About",
        "steps": [
            ("① Input", "Upload the voice to protect, pick the speaker"),
            ("② Params", "Perturbation strength ε (0.05 recommended); demo mode is faster"),
            ("③ Run", "Click Protect — GPU optimization takes 2–5 min"),
            ("④ Verify", "Listen and check whether VC fails to forge"),
        ],
        "g1": "① Input",
        "defend_label": "🎙 Voice to protect (speaker X)",
        "adv_label": "🎯 Perturbation-direction audio (least-likely speaker Y; auto if empty)",
        "speaker_label": "Protected speaker (in registry)",
        "g2": "② Parameters",
        "eps_label": "ε (perturbation strength)",
        "eps_info": "0.05 is our best operating point; smaller = less audible, larger = stronger",
        "demo_label": "⚡ Demo fast mode (300/200 iters)",
        "demo_info": "Uncheck for full 3000/1500 iters (slower, more stable)",
        "g3": "③ Run",
        "run_btn": "🛡️ Protect",
        "g4": "④ Verification",
        "status_placeholder": "The defense verdict will appear here after running.",
        "t_try": "🎧 Listen",
        "t_plot": "📈 Waveform",
        "t_detail": "📋 Detail",
        "a_orig": "Original (trimmed)",
        "a_def": "Defended audio",
        "a_delta": "Perturbation itself",
        "a_cc": "Clean → VC conversion",
        "a_cd": "Defended → VC conversion",
        "plot_label": "Waveform comparison",
        "detail_title": "Top-5 similar speakers for defended audio (ECAPA)",
        "detail_th": ["Speaker", "Similarity"],
        "metric_items": [("ε", "{:.2f}"), ("c (mask weight)", "{}"), ("MCD", "{:.1f}"),
                         ("Duration", "{:.1f}s"), ("Elapsed", "{:.0f}s"), ("Iters", "{}")],
        "ok_head": "✅ Defense succeeded",
        "part_head": "⚠️ Partial: conversion may still be recognized as ",
        "fail_head": "❌ Failed: defended audio classified as ",
        "st_def": "Defended audio verdict",
        "st_conv": "Defended→VC verdict",
        "st_eq": "Criterion",
        "st_eq_v": "F_imp ∧ F_str (Eq.9-11, ECAPA)",
        "err_no_defend": "Please upload the voice to protect first",
        "err_need_adv": "Please provide a least-likely target audio",
        "info_auto_adv": "Auto-selected least-likely speaker {} audio",
        "info_start": "Starting adversarial optimization ({}+{} iters, ~2–5 min)…",
        "examples_label": "💡 Try a VCTK sample (click to load)",
        "conv_md": "Zero-shot voice conversion demo: **content** from one clip, **voice** from another — "
                   "exactly how attackers forge voices; the perturbation from the Protect tab breaks this.",
        "src_label": "📄 Content audio t̂",
        "tgt_label": "🗣 Voice audio x̂",
        "conv_btn": "Convert",
        "conv_out_label": "F(t̂, x̂)",
        "err_conv": "Please upload two audios",
        "about_md": "## Method\n"
                    "1. Inject time-domain perturbation δ = 2ε·tanh(w) into speaker X's audio;\n"
                    "2. Objective: VC conversion of the defended audio is no longer recognized as X "
                    "(attack term), while the perturbation stays under the psychoacoustic masking "
                    "threshold θx (masking term, inaudible);\n"
                    "3. Criterion (Eq.9-11): **F_imp** (defended still recognized as X) ∧ "
                    "**F_str** (defended→VC no longer recognized as X) ⇒ defense succeeds.\n\n"
                    "## Our reproduction results\n"
                    "- ε sweep: defense rate 0.02→75%, 0.05→75%, 0.08→50%, 0.10→50%;\n"
                    "- Method comparison: time-domain VG 75% ≫ frequency-domain 25%;\n"
                    "- Real-channel analysis (mp3/8k/G.711): F_imp is the weakest link.\n\n"
                    "Paper: Li et al., *Voice Guard*, IJCAI 2023, pp.4812–4820.\n"
                    "Repo: [github.com/Abyss-sword/Voice-Guard](https://github.com/Abyss-sword/Voice-Guard)",
        "curve_label": "ε trade-off curve (reproduction)",
    },
}

# ---------------------------------------------------------------- CSS（双模式）

CSS = """
/* ===== 白昼（默认浅色科技风）===== */
:root {
  --vg-accent1: #6366f1;
  --vg-accent2: #06b6d4;
  --vg-ink: #0f172a;
}
.vg-hero {
  background: radial-gradient(1200px 400px at 12% -20%, rgba(99,102,241,.35), transparent),
              radial-gradient(900px 320px at 88% 120%, rgba(6,182,212,.3), transparent),
              linear-gradient(120deg, #1e1b4b 0%, #312e81 45%, #164e63 100%);
  border-radius: 18px; padding: 26px 30px; color: #fff; margin-bottom: 16px;
  border: 1px solid rgba(148,163,184,.35); box-shadow: 0 10px 30px rgba(30,27,75,.25);
}
.vg-hero h2 { color: #fff; margin: 0 0 6px 0; font-size: 25px; letter-spacing: .3px; }
.vg-hero p { margin: 4px 0; opacity: .93; font-size: 14px; line-height: 1.55; }
.vg-badges { margin-top: 12px; }
.vg-badge {
  display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.4);
  border-radius: 999px; padding: 4px 14px; margin-right: 8px; font-size: 12.5px;
  backdrop-filter: blur(4px);
}
.vg-steps { display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
.vg-step {
  flex: 1; min-width: 148px; background: linear-gradient(160deg,#eef2ff,#ecfeff);
  border: 1px solid #c7d2fe; border-radius: 12px; padding: 10px 14px;
  font-size: 12.5px; color: #312e81; position: relative;
}
.vg-step b { display: block; font-size: 14.5px; margin-bottom: 2px; color: #1e1b4b; }
.vg-status-ok {
  background: linear-gradient(160deg,#ecfdf5,#f0fdfa); border: 1.5px solid #34d399;
  border-radius: 14px; padding: 14px 18px; font-size: 15px; color: #065f46;
  box-shadow: 0 4px 16px rgba(16,185,129,.15);
}
.vg-status-fail {
  background: linear-gradient(160deg,#fef2f2,#fff7ed); border: 1.5px solid #f87171;
  border-radius: 14px; padding: 14px 18px; font-size: 15px; color: #991b1b;
}
.vg-metrics { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.vg-metric {
  flex: 1; min-width: 104px; background: linear-gradient(160deg,#ffffff,#f1f5f9);
  border: 1px solid #e2e8f0; border-radius: 12px; padding: 9px 10px; text-align: center;
  box-shadow: 0 2px 8px rgba(15,23,42,.05);
}
.vg-metric .k { font-size: 11.5px; color: #64748b; }
.vg-metric .v { font-size: 17px; font-weight: 600; color: #0f172a; margin-top: 2px;
  font-family: 'Consolas','JetBrains Mono',monospace; }
.vg-note { font-size: 13px; color: #64748b; margin-top: 6px; }
.vg-detail { font-size: 13.5px; }
.vg-detail th { text-align: left; padding: 2px 16px 2px 0; color: #475569; font-weight: 600; }

/* ===== 黑夜模式（深空科技风）===== */
html.vg-dark {
  --body-background-fill: #0b1220;
  --body-background-color: #0b1220;
  --body-text-color: #e2e8f0;
  --body-text-color-subdued: #94a3b8;
  --background-fill-primary: #0b1220;
  --background-fill-secondary: #0f172a;
  --block-background-fill: #101a2e;
  --block-border-color: #1e293b;
  --block-title-text-color: #e2e8f0;
  --block-label-text-color: #94a3b8;
  --block-info-text-color: #7dd3fc;
  --input-background-fill: #0b1526;
  --input-border-color: #24324b;
  --input-border-color-focus: #6366f1;
  --border-color-primary: #1e293b;
  --button-primary-background-fill: linear-gradient(90deg,#6366f1,#06b6d4);
  --button-primary-background-fill-hover: linear-gradient(90deg,#4f46e5,#0891b2);
  --panel-background-fill: #0f172a;
  --table-border-color: #1e293b;
  --tab-selected-color: #6366f1;
  --checkbox-background-color: #1e293b;
  --checkbox-background-color-hover: #24324b;
  --checkbox-border-color: #334155;
  --color-accent: #818cf8;
  --block-shadow: 0 6px 24px rgba(0,0,0,.45);
  --link-text-color: #7dd3fc;
  --link-text-color-hover: #a5f3fc;
  --stat-background-fill: linear-gradient(90deg,#6366f1,#06b6d4);
  --body-border-color: #1e293b;
}
html.vg-dark .vg-hero {
  border: 1px solid rgba(129,140,248,.5);
  box-shadow: 0 0 40px rgba(99,102,241,.35), 0 10px 30px rgba(0,0,0,.5);
}
html.vg-dark .vg-step {
  background: linear-gradient(160deg,#131c33,#0e2433);
  border: 1px solid #24324b; color: #a5b4fc;
}
html.vg-dark .vg-step b { color: #e0e7ff; }
html.vg-dark .vg-status-ok {
  background: rgba(16,185,129,.1); border-color: #059669; color: #6ee7b7;
  box-shadow: 0 0 24px rgba(16,185,129,.25);
}
html.vg-dark .vg-status-fail {
  background: rgba(239,68,68,.1); border-color: #dc2626; color: #fca5a5;
}
html.vg-dark .vg-metric {
  background: linear-gradient(160deg,#131c33,#0e1f2e);
  border: 1px solid #24324b; box-shadow: none;
}
html.vg-dark .vg-metric .k { color: #7dd3fc; }
html.vg-dark .vg-metric .v { color: #e0f2fe; text-shadow: 0 0 12px rgba(125,211,252,.5); }
html.vg-dark .vg-detail th { color: #94a3b8; }
footer { display: none !important; }
"""

HEAD = ("<script>"
        "const saved=localStorage.getItem('vg-dark');"
        "if(saved===null||saved==='1'){document.documentElement.classList.add('vg-dark');}"
        "</script>")

DARK_JS = ("(v)=>{document.documentElement.classList.toggle('vg-dark',v);"
           "localStorage.setItem('vg-dark',v?'1':'0');return v;}")


# ---------------------------------------------------------------- HTML 生成

def _fmt(v, spec):
    return spec.format(v) if v is not None else "—"


def _status_html(report: dict, lang: str) -> str:
    L = T[lang]
    tgt = report["target_speaker"]
    asv = report.get("asv", {})
    sim_def = asv.get("defended", {}).get("sim", {}).get(tgt, None)
    sim_conv = asv.get("conv_def", {}).get("sim", {}).get(tgt, None)
    pred_def = asv.get("defended", {}).get("pred", "-")
    pred_conv = asv.get("conv_def", {}).get("pred", "-")

    imp = pred_def == tgt
    str_ = pred_conv != tgt
    if imp and str_:
        head = L["ok_head"]
    elif imp:
        head = L["part_head"] + pred_conv
    else:
        head = L["fail_head"] + pred_def

    rows = "".join(
        f"<tr><td>{k}</td><td style='padding-left:14px;'>{v}</td></tr>"
        for k, v in [
            (L["st_def"], pred_def + (f"（sim {sim_def:.3f}）" if sim_def is not None else "")),
            (L["st_conv"], pred_conv + (f"（sim {sim_conv:.3f}）" if sim_conv is not None else "")),
            (L["st_eq"], L["st_eq_v"]),
        ]
    )
    cls = "vg-status-ok" if (imp and str_) else "vg-status-fail"
    return (f'<div class="{cls}"><b>{head}</b>'
            f'<table style="margin-top:8px;font-size:13.5px;border-collapse:collapse;">{rows}</table></div>')


def _metrics_html(report: dict, mcd, lang: str) -> str:
    L = T[lang]
    vals = [report["eps"], report["c_final"], mcd, report["duration_s"],
            report["elapsed_s"], f'{report["iters1"]}+{report["iters2"]}']
    cells = "".join(
        f'<div class="vg-metric"><div class="k">{name}</div><div class="v">{_fmt(v, spec)}</div></div>'
        for (name, spec), v in zip(L["metric_items"], vals))
    return f'<div class="vg-metrics">{cells}</div>'


def _detail_html(report: dict, lang: str) -> str:
    L = T[lang]
    asv = report.get("asv", {})
    sims = asv.get("defended", {}).get("sim", {})
    top = sorted(sims.items(), key=lambda kv: -kv[1])[:5] if sims else []
    rows = "".join(f"<tr><td>{n}</td><td style='padding-left:16px;'>{s:.3f}</td></tr>" for n, s in top)
    return (f'<div class="vg-detail"><b>{L["detail_title"]}</b>'
            f'<table style="margin-top:8px;border-collapse:collapse;">'
            f'<tr><th>{L["detail_th"][0]}</th><th style="padding-left:16px;">{L["detail_th"][1]}</th></tr>'
            f'{rows}</table></div>')


def _hero_html(lang: str) -> str:
    L = T[lang]
    badges = "".join(f'<span class="vg-badge">{b}</span>' for b in L["badges"])
    return (f'<div class="vg-hero"><h2>{L["hero_title"]}</h2><p>{L["hero_sub"]}</p>'
            f'<div class="vg-badges">{badges}</div></div>')


def _steps_html(lang: str) -> str:
    L = T[lang]
    cells = "".join(f'<div class="vg-step"><b>{t}</b>{d}</div>' for t, d in L["steps"])
    return f'<div class="vg-steps">{cells}</div>'


def _plot_waveforms(paths, out_png):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0b1220")
    for key, label, kw in (("original", "original", dict(lw=0.7, color="#94a3b8")),
                           ("defended", "defended", dict(lw=0.7, color="#818cf8", alpha=0.95)),
                           ("delta", "perturbation (×5)", dict(lw=0.8, color="#f87171", alpha=0.7))):
        if key not in paths:
            continue
        w, sr = sf.read(paths[key])
        t = np.arange(len(w)) / sr
        ax.plot(t, w * (5 if key == "delta" else 1), label=label, **kw)
    ax.legend(loc="upper right", fontsize=10, facecolor="#0f172a",
              labelcolor="#cbd5e1", edgecolor="#24324b")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#24324b")
    ax.set_xlabel("time (s)", fontsize=10, color="#94a3b8")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------- 应用构建

def build_app(args, pipe: Pipeline, lang: str = "zh"):
    L = T[lang]

    speaker_first = {}
    if args.vctk and os.path.isdir(args.vctk):
        for s in pipe.speakers:
            d = os.path.join(args.vctk, s)
            if os.path.isdir(d):
                wavs = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
                if wavs:
                    speaker_first[s] = os.path.join(d, wavs[0])

    theme = gr.themes.Base()
    curve_img = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                             "docs", "fig_eps_curve.png"))

    with gr.Blocks(title="Voice Guard — Voice Privacy Defense", theme=theme,
                   css=CSS, head=HEAD) as demo:
        # 顶部控制条：昼夜 + 语言
        with gr.Row():
            dark_cb = gr.Checkbox(value=True, label=L["dark_label"], scale=1)
            lang_dd = gr.Dropdown(choices=["中文", "English"],
                                  value="中文" if lang == "zh" else "English",
                                  label=L["lang_label"], scale=1)

        hero_html = gr.HTML(_hero_html(lang))
        steps_html = gr.HTML(_steps_html(lang))

        with gr.Tab(L["tab_protect"]) as tab1:
            with gr.Group():
                g1_md = gr.Markdown(f"### {L['g1']}")
                with gr.Row():
                    defend_in = gr.Audio(label=L["defend_label"], type="filepath")
                    adv_in = gr.Audio(label=L["adv_label"], type="filepath")
                speaker_dd = gr.Dropdown(choices=pipe.speakers, value=pipe.speakers[0],
                                         label=L["speaker_label"])

            with gr.Group():
                g2_md = gr.Markdown(f"### {L['g2']}")
                with gr.Row():
                    eps_sl = gr.Slider(0.02, 0.3, value=0.05, step=0.01,
                                       label=L["eps_label"], info=L["eps_info"])
                    demo_cb = gr.Checkbox(value=True, label=L["demo_label"],
                                          info=L["demo_info"])

            with gr.Group():
                g3_md = gr.Markdown(f"### {L['g3']}")
                run_btn = gr.Button(L["run_btn"], variant="primary", size="lg")

            with gr.Group():
                g4_md = gr.Markdown(f"### {L['g4']}")
                status_html = gr.HTML(f'<div class="vg-note">{L["status_placeholder"]}</div>')
                metrics_html = gr.HTML()
                with gr.Tabs():
                    with gr.Tab(L["t_try"]):
                        with gr.Row():
                            aud_orig = gr.Audio(label=L["a_orig"])
                            aud_def = gr.Audio(label=L["a_def"])
                        with gr.Row():
                            aud_delta = gr.Audio(label=L["a_delta"])
                            aud_conv_clean = gr.Audio(label=L["a_cc"])
                            aud_conv_def = gr.Audio(label=L["a_cd"])
                    with gr.Tab(L["t_plot"]):
                        plot_out = gr.Image(label=L["plot_label"])
                    with gr.Tab(L["t_detail"]):
                        detail_html = gr.HTML()

            last_state = gr.State({})

            def on_protect(defend, speaker, adv, eps, demo_mode, state):
                if not defend:
                    raise gr.Error(L["err_no_defend"])
                if not adv:
                    y = pipe.least_likely(speaker)
                    if y in speaker_first:
                        adv = speaker_first[y]
                        gr.Info(L["info_auto_adv"].format(y))
                    else:
                        raise gr.Error(L["err_need_adv"])
                it1, it2 = (300, 200) if demo_mode else (args.iters1, args.iters2)
                out_dir = os.path.join(args.out_dir, f"{speaker}_{int(time.time())}")
                gr.Info(L["info_start"].format(it1, it2))
                rep = pipe.protect(defend, speaker, adv, eps=eps, iters1=it1, iters2=it2,
                                   out_dir=out_dir, progress=False)

                mcd_val = None
                try:
                    from vguard.metrics.mcd import mcd_mel
                    w1, _ = sf.read(rep["files"]["original"])
                    w2, _ = sf.read(rep["files"]["defended"])
                    mcd_val = float(mcd_mel(w1, w2, pipe.pp, attr=None, device=pipe.device))
                except Exception:  # noqa: BLE001
                    pass

                png = _plot_waveforms(rep["files"], os.path.join(out_dir, "waves.png"))
                state = {"report": rep, "mcd": mcd_val}
                return (_status_html(rep, lang), rep["files"]["original"], rep["files"]["defended"],
                        rep["files"]["delta"], rep["files"]["conv_clean"],
                        rep["files"]["conv_def"], png, _metrics_html(rep, mcd_val, lang),
                        _detail_html(rep, lang), state)

            run_btn.click(on_protect,
                          [defend_in, speaker_dd, adv_in, eps_sl, demo_cb, last_state],
                          [status_html, aud_orig, aud_def, aud_delta, aud_conv_clean,
                           aud_conv_def, plot_out, metrics_html, detail_html, last_state])

            if speaker_first:
                ex = [[speaker_first[pipe.speakers[0]], pipe.speakers[0]]]
                gr.Examples(examples=ex, inputs=[defend_in, speaker_dd],
                            label=L["examples_label"])

        with gr.Tab(L["tab_convert"]) as tab2:
            conv_md = gr.Markdown(L["conv_md"])
            with gr.Row():
                src_in = gr.Audio(label=L["src_label"], type="filepath")
                tgt_in = gr.Audio(label=L["tgt_label"], type="filepath")
            conv_btn = gr.Button(L["conv_btn"], variant="primary")
            conv_out = gr.Audio(label=L["conv_out_label"])

            def on_convert(src, tgt):
                if not src or not tgt:
                    raise gr.Error(L["err_conv"])
                out = os.path.join(args.out_dir, f"conv_{int(time.time())}.wav")
                from vguard.vc.adainvc import convert as vc_convert
                vc_convert(pipe.model, pipe.config, pipe.attr, src, tgt, out, pipe.device)
                return out

            conv_btn.click(on_convert, [src_in, tgt_in], [conv_out])

        with gr.Tab(L["tab_about"]) as tab3:
            about_md = gr.Markdown(L["about_md"])
            curve_out = None
            if os.path.exists(curve_img):
                curve_out = gr.Image(value=curve_img, label=L["curve_label"], height=420)

        # ---- 语言切换：重写全部文案 ----
        def on_lang(choice, speaker, state):
            nl = "zh" if choice == "中文" else "en"
            NL = T[nl]
            rep = state.get("report")
            return [
                gr.update(value=_hero_html(nl)),                    # 0 hero
                gr.update(value=_steps_html(nl)),                   # 1 steps
                gr.update(label=NL["dark_label"]),                  # 2 dark
                gr.update(label=NL["lang_label"]),                  # 3 lang_dd
                gr.update(value=f"### {NL['g1']}"),                 # 4 g1
                gr.update(label=NL["defend_label"]),                # 5 defend_in
                gr.update(label=NL["adv_label"]),                   # 6 adv_in
                gr.update(choices=pipe.speakers, value=speaker,
                          label=NL["speaker_label"]),               # 7 speaker_dd
                gr.update(value=f"### {NL['g2']}"),                 # 8 g2
                gr.update(label=NL["eps_label"], info=NL["eps_info"]),  # 9 eps
                gr.update(label=NL["demo_label"], info=NL["demo_info"]),  # 10 demo
                gr.update(value=f"### {NL['g3']}"),                 # 11 g3
                gr.update(value=NL["run_btn"]),                     # 12 run_btn
                gr.update(value=f"### {NL['g4']}"),                 # 13 g4
                gr.update(value=(_status_html(rep, nl) if rep
                                 else f'<div class="vg-note">{NL["status_placeholder"]}</div>')),  # 14 status
                gr.update(label=NL["a_orig"]),                      # 15..19 audios
                gr.update(label=NL["a_def"]),
                gr.update(label=NL["a_delta"]),
                gr.update(label=NL["a_cc"]),
                gr.update(label=NL["a_cd"]),
                gr.update(label=NL["plot_label"]),                  # 20 plot
                gr.update(value=NL["conv_md"]),                     # 21 conv_md
                gr.update(label=NL["src_label"]),                   # 22 src
                gr.update(label=NL["tgt_label"]),                   # 23 tgt
                gr.update(value=NL["conv_btn"]),                    # 24 conv_btn
                gr.update(label=NL["conv_out_label"]),              # 25 conv_out
                gr.update(value=NL["about_md"]),                    # 26 about
                gr.update(value=(_metrics_html(rep, state.get("mcd"), nl) if rep else "")),  # 27 metrics
                gr.update(value=(_detail_html(rep, nl) if rep else "")),                    # 28 detail
                state,                                              # 29 state
                gr.update(label=NL["tab_protect"]),                 # 30 tab1
                gr.update(label=NL["tab_convert"]),                 # 31 tab2
                gr.update(label=NL["tab_about"]),                   # 32 tab3
            ]

        lang_dd.change(on_lang, [lang_dd, speaker_dd, last_state],
                       [hero_html, steps_html, dark_cb, lang_dd, g1_md, defend_in, adv_in,
                        speaker_dd, g2_md, eps_sl, demo_cb, g3_md, run_btn, g4_md,
                        status_html, aud_orig, aud_def, aud_delta, aud_conv_clean, aud_conv_def,
                        plot_out, conv_md, src_in, tgt_in, conv_btn, conv_out, about_md,
                        metrics_html, detail_html, last_state, tab1, tab2, tab3])

        # 昼夜切换（纯前端 JS + localStorage 记忆）
        dark_cb.change(None, [dark_cb], None, js=DARK_JS)

    return demo


# ---------------------------------------------------------------- gradio 兼容补丁

def _patch_gradio_client_schema():
    """gradio-client 1.3.0 对 JSON Schema 的 `additionalProperties: true` 与可空
    字段 `{"type": ["string","null"]}` 未处理，会导致 /info 500 与 FileData 解析失败。
    补丁：bool/空 schema → "Any"；可空类型取非 null 分支；`path: str` 结构识别为
    filepath（保证客户端文件上传协议）。"""
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
    pipe = Pipeline(args.model_dir, args.registry, args.attack_registry,
                    ecapa_dir=args.ecapa_dir, device=args.device)
    demo = build_app(args, pipe, "zh")
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
