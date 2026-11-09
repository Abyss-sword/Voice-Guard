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
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from vguard.pipeline import Pipeline  # noqa: E402
from vguard.evaluation import identity_verdict  # noqa: E402

# ---------------------------------------------------------------- 文案（双语）

T = {
    "zh": {
        "hero_title": "🛡️ Voice Guard · 语音隐私主动防御",
        "hero_sub": "探索 IJCAI 2023《Voice Guard》：用时域对抗扰动与心理声学掩蔽，"
                    "尝试降低 AdaIN-VC 复制目标声纹的能力。请试听并核查每次结果。",
        "badges": ["⏱ 时域扰动", "👂 心理声学掩蔽", "🧪 ECAPA 身份维评估"],
        "dark_label": "🌙 夜间模式",
        "theme_dark": "🌙 黑夜",
        "theme_light": "☀️ 白昼",
        "lang_label": "语言 / Language",
        "tab_protect": "🛡️ 一键保护",
        "tab_convert": "🔄 转换演示",
        "tab_about": "📖 说明",
        "steps": [
            ("① 输入", "上传要保护的声音，选择说话人"),
            ("② 参数", "扰动强度 ε（推荐 0.05），演示模式更快"),
            ("③ 执行", "点击「一键保护」，等待优化完成"),
            ("④ 验证", "试听并查看 VC 转换是否失效"),
        ],
        "g1": "① 输入",
        "defend_label": "🎙 要保护的声音（说话人 X）",
        "adv_label": "🎯 扰动方向音频（least-likely 说话人 Y；留空自动选取）",
        "speaker_label": "被保护说话人（注册库内）",
        "g2": "② 参数",
        "eps_label": "ε（扰动强度）",
        "eps_info": "0.05 为历史抽样的起点；更大扰动也可能损害音质与身份保真",
        "demo_label": "⚡ 演示快速模式（300/200 步）",
        "demo_info": "关闭则用完整 3000/1500 步，耗时更长，不保证成功",
        "g3": "③ 执行",
        "run_btn": "🛡️ 一键保护",
        "g4": "④ 验证结果",
        "status_placeholder": "运行后这里显示防御判定。",
        "t_try": "🎧 试听",
        "t_plot": "📈 波形",
        "t_detail": "📋 明细",
        "a_orig": "原始音频（按所选裁剪选项）",
        "a_def": "被保护音频",
        "a_delta": "扰动本身",
        "a_cc": "干净 → VC 转换",
        "a_cd": "被保护 → VC 转换",
        "plot_label": "波形对比",
        "detail_title": "defended 音频的 top-5 相似说话人（ECAPA）",
        "detail_th": ["说话人", "相似度"],
        "metric_items": [("ε", "{:.2f}"), ("c（掩蔽权重）", "{}"), ("MCD", "{:.1f}"),
                         ("音频时长", "{:.1f}s"), ("耗时", "{:.0f}s"), ("迭代", "{}")],
        "ok_head": "✅ 身份维判定通过（请同时试听音质）",
        "part_head": "⚠️ 部分成功：转换后仍可能被识别为 ",
        "fail_head": "❌ 防御未达标：被保护音频被判为 ",
        "st_def": "被保护音频判定",
        "st_conv": "被保护→VC 转换判定",
        "st_eq": "判定式",
        "st_eq_v": "ECAPA：保护后身份保留，转换后身份改变；未联合 MOS/WER 阈值",
        "err_no_defend": "请先上传要保护的声音",
        "err_need_adv": "请提供 least-likely 目标音频",
        "info_auto_adv": "已自动选取 least-likely 说话人 {} 的音频",
        "info_start": "开始对抗优化（{}+{} 步，耗时取决于设备与音频长度）…",
        "examples_label": "💡 试试 VCTK 样例（点击载入）",
        "sample_btn": "💡 载入 VCTK 样例音频（p225）",
        "conv_md": "零样本语音转换演示：**内容**来自一段音频、**声纹**来自另一段——"
                   "可以检查保护页生成的扰动是否影响转换身份。",
        "src_label": "📄 内容音频 t̂",
        "tgt_label": "🗣 声纹音频 x̂",
        "conv_btn": "转换",
        "conv_out_label": "F(t̂, x̂)",
        "err_conv": "请上传两段音频",
        "about_md": "## 方法\n"
                    "1. 给说话人 X 的音频注入时域扰动 δ = 2ε·tanh(w)；\n"
                    "2. 优化目标：让 X 的音频经 VC 转换后不再被判为 X（攻击项），"
                    "同时尝试降低扰动超过心理声学掩蔽阈值 θx 的能量；\n"
                    "3. 干净转换先识别为 X，再要求保护音频仍识别为 X、保护后转换不再识别为 X。\n\n"
                    "## 云端实测与质量改进\n"
                    "- 首批五句：VG 保护音频 MOSNet 均值 1.397、WER 62.76%；\n"
                    "- 谱投影后继续优化：MOSNet 3.192、WER 5.71%，筛选后联合通过 2/4；\n"
                    "- 是小样本观察，包含失败。MOSNet 非人工 MOS，本页面未联合质量阈值。\n\n"
                    "论文：Li et al., *Voice Guard: Protecting Voice Privacy with Strong and "
                    "Imperceptible Adversarial Perturbation in the Time Domain*, IJCAI 2023, "
                    "pp.4812–4820。\n"
                    "仓库：[github.com/Abyss-sword/Voice-Guard](https://github.com/Abyss-sword/Voice-Guard)",
        "curve_label": "云端首批五句质量比较（原方法/实验性改进）",
    },
    "en": {
        "hero_title": "🛡️ Voice Guard · Proactive Voice Privacy Defense",
        "hero_sub": "Explore IJCAI 2023 Voice Guard: time-domain perturbations and psychoacoustic "
                    "masking aim to reduce AdaIN-VC voice imitation. Listen to and verify each result.",
        "badges": ["⏱ Time-domain", "👂 Psychoacoustic masking", "🧪 ECAPA identity evaluation"],
        "dark_label": "🌙 Dark mode",
        "theme_dark": "🌙 Dark",
        "theme_light": "☀️ Light",
        "lang_label": "语言 / Language",
        "tab_protect": "🛡️ Protect",
        "tab_convert": "🔄 Conversion demo",
        "tab_about": "📖 About",
        "steps": [
            ("① Input", "Upload the voice to protect, pick the speaker"),
            ("② Params", "Perturbation strength ε (0.05 recommended); demo mode is faster"),
            ("③ Run", "Click Protect and wait for optimization"),
            ("④ Verify", "Listen and check whether VC fails to forge"),
        ],
        "g1": "① Input",
        "defend_label": "🎙 Voice to protect (speaker X)",
        "adv_label": "🎯 Perturbation-direction audio (least-likely speaker Y; auto if empty)",
        "speaker_label": "Protected speaker (in registry)",
        "g2": "② Parameters",
        "eps_label": "ε (perturbation strength)",
        "eps_info": "0.05 is a historical starting point; stronger perturbations can harm quality and identity",
        "demo_label": "⚡ Demo fast mode (300/200 iters)",
        "demo_info": "Uncheck for full 3000/1500 iters; slower, without a success guarantee",
        "g3": "③ Run",
        "run_btn": "🛡️ Protect",
        "g4": "④ Verification",
        "status_placeholder": "The defense verdict will appear here after running.",
        "t_try": "🎧 Listen",
        "t_plot": "📈 Waveform",
        "t_detail": "📋 Detail",
        "a_orig": "Original (selected crop/context)",
        "a_def": "Defended audio",
        "a_delta": "Perturbation itself",
        "a_cc": "Clean → VC conversion",
        "a_cd": "Defended → VC conversion",
        "plot_label": "Waveform comparison",
        "detail_title": "Top-5 similar speakers for defended audio (ECAPA)",
        "detail_th": ["Speaker", "Similarity"],
        "metric_items": [("ε", "{:.2f}"), ("c (mask weight)", "{}"), ("MCD", "{:.1f}"),
                         ("Duration", "{:.1f}s"), ("Elapsed", "{:.0f}s"), ("Iters", "{}")],
        "ok_head": "✅ Identity criterion passed; also check audio quality",
        "part_head": "⚠️ Partial: conversion may still be recognized as ",
        "fail_head": "❌ Failed: defended audio classified as ",
        "st_def": "Defended audio verdict",
        "st_conv": "Defended→VC verdict",
        "st_eq": "Criterion",
        "st_eq_v": "ECAPA identity only; MOS/WER thresholds are not included",
        "err_no_defend": "Please upload the voice to protect first",
        "err_need_adv": "Please provide a least-likely target audio",
        "info_auto_adv": "Auto-selected least-likely speaker {} audio",
        "info_start": "Starting optimization ({}+{} iters; time depends on hardware and clip length)…",
        "examples_label": "💡 Try a VCTK sample (click to load)",
        "sample_btn": "💡 Load VCTK sample audio (p225)",
        "conv_md": "Zero-shot voice conversion demo: **content** from one clip, **voice** from another — "
                   "check whether the perturbation from the Protect tab changes the converted identity.",
        "src_label": "📄 Content audio t̂",
        "tgt_label": "🗣 Voice audio x̂",
        "conv_btn": "Convert",
        "conv_out_label": "F(t̂, x̂)",
        "err_conv": "Please upload two audios",
        "about_md": "## Method\n"
                    "1. Inject time-domain perturbation δ = 2ε·tanh(w) into speaker X's audio;\n"
                    "2. Objective: VC conversion of the defended audio is no longer recognized as X "
                    "(attack term), while the perturbation stays under the psychoacoustic masking "
                    "threshold θx (a training objective);\n"
                    "3. Require a correctly matched clean conversion, retained protected identity and changed converted identity.\n\n"
                    "## Measured cloud samples\n"
                    "- Five original VG clips: mean MOSNet 1.397, macro WER 62.76%;\n"
                    "- Projected refinement: mean MOSNet 3.192, macro WER 5.71%, screened joint pass 2/4;\n"
                    "- Small sample with failures. MOSNet is not human MOS; this interface evaluates identity only.\n\n"
                    "Paper: Li et al., *Voice Guard*, IJCAI 2023, pp.4812–4820.\n"
                    "Repo: [github.com/Abyss-sword/Voice-Guard](https://github.com/Abyss-sword/Voice-Guard)",
        "curve_label": "Five measured cloud clips: original vs experimental refinement",
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
@keyframes vg-grad {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}
@keyframes vg-float {
  0%, 100% { transform: translateY(0) scale(1); }
  50% { transform: translateY(-16px) scale(1.06); }
}
@keyframes vg-scan {
  0% { transform: translateY(-120%); }
  100% { transform: translateY(520%); }
}
@keyframes vg-fade {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: none; }
}
@keyframes vg-pulse {
  0%, 100% { box-shadow: 0 0 16px rgba(34,211,238,.35); }
  50% { box-shadow: 0 0 30px rgba(99,102,241,.7); }
}
.vg-hero {
  position: relative; overflow: hidden;
  background: radial-gradient(1200px 400px at 12% -20%, rgba(99,102,241,.4), transparent),
              radial-gradient(900px 320px at 88% 120%, rgba(6,182,212,.35), transparent),
              linear-gradient(120deg, #1e1b4b 0%, #312e81 45%, #164e63 100%);
  background-size: 100% 100%, 100% 100%, 220% 220%;
  animation: vg-grad 9s ease infinite;
  border-radius: 18px; padding: 26px 30px; color: #ffffff; margin-bottom: 16px;
  border: 1px solid rgba(148,163,184,.4); box-shadow: 0 10px 30px rgba(30,27,75,.28);
}
.vg-hero h2 { color: #ffffff !important; margin: 0 0 6px 0; font-size: 25px; letter-spacing: .3px; }
.vg-hero p { color: #eef2ff !important; margin: 4px 0; opacity: .95; font-size: 14px; line-height: 1.55; }
.vg-hero b { color: #a5f3fc; }
.vg-hero::after {
  content: ""; position: absolute; left: 0; right: 0; top: 0; height: 34%;
  background: repeating-linear-gradient(0deg, rgba(255,255,255,.045) 0 1px, transparent 1px 4px);
  animation: vg-scan 7s linear infinite; pointer-events: none;
}
.vg-orb {
  position: absolute; border-radius: 999px; filter: blur(2px); pointer-events: none;
}
.vg-orb1 { width: 130px; height: 130px; right: 12%; top: -46px;
  background: radial-gradient(circle, rgba(129,140,248,.55), transparent 70%);
  animation: vg-float 6s ease-in-out infinite; }
.vg-orb2 { width: 90px; height: 90px; right: 30%; bottom: -34px;
  background: radial-gradient(circle, rgba(34,211,238,.5), transparent 70%);
  animation: vg-float 8s ease-in-out infinite reverse; }
.vg-badges { margin-top: 12px; }
.vg-badge {
  display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.42);
  border-radius: 999px; padding: 4px 14px; margin-right: 8px; font-size: 12.5px;
  color: #f1f5ff !important; backdrop-filter: blur(4px);
}
.vg-steps { display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
.vg-step {
  flex: 1; min-width: 148px; background: linear-gradient(160deg,#eef2ff,#ecfeff);
  border: 1px solid #c7d2fe; border-radius: 12px; padding: 10px 14px;
  font-size: 12.5px; color: #312e81; position: relative;
  transition: transform .2s ease, box-shadow .2s ease;
}
.vg-step:hover { transform: translateY(-3px); box-shadow: 0 8px 22px rgba(99,102,241,.18); }
.vg-step b { display: block; font-size: 14.5px; margin-bottom: 2px; color: #1e1b4b; }
.vg-status-ok {
  background: linear-gradient(160deg,#ecfdf5,#f0fdfa); border: 1.5px solid #34d399;
  border-radius: 14px; padding: 14px 18px; font-size: 15px; color: #065f46;
  box-shadow: 0 4px 16px rgba(16,185,129,.15);
  animation: vg-fade .5s ease both;
}
.vg-status-fail {
  background: linear-gradient(160deg,#fef2f2,#fff7ed); border: 1.5px solid #f87171;
  border-radius: 14px; padding: 14px 18px; font-size: 15px; color: #991b1b;
  animation: vg-fade .5s ease both;
}
.vg-metrics { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.vg-metric {
  flex: 1; min-width: 104px; background: linear-gradient(160deg,#ffffff,#f1f5f9);
  border: 1px solid #e2e8f0; border-radius: 12px; padding: 9px 10px; text-align: center;
  box-shadow: 0 2px 8px rgba(15,23,42,.05);
  animation: vg-fade .5s ease both;
  transition: transform .2s ease;
}
.vg-metric:hover { transform: translateY(-2px); }
.vg-metric .k { font-size: 11.5px; color: #64748b; }
.vg-metric .v { font-size: 17px; font-weight: 600; color: #0f172a; margin-top: 2px;
  font-family: 'Consolas','JetBrains Mono',monospace; }
.vg-note { font-size: 13px; color: #64748b; margin-top: 6px; }
.vg-detail { font-size: 13.5px; animation: vg-fade .5s ease both; }
.vg-detail th { text-align: left; padding: 2px 16px 2px 0; color: #475569; font-weight: 600; }
button.primary {
  transition: transform .18s ease, box-shadow .18s ease !important;
  box-shadow: 0 4px 16px rgba(99,102,241,.35) !important;
}
button.primary:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 24px rgba(34,211,238,.55) !important;
}

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
  --checkbox-label-background-fill: #0f172a;
  --checkbox-label-text-color: #e2e8f0;
  --checkbox-label-border-color: #24324b;
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
html.vg-dark .vg-step:hover { box-shadow: 0 8px 22px rgba(34,211,238,.25); }
html.vg-dark .vg-step b { color: #e0e7ff; }
html.vg-dark .vg-status-ok {
  background: rgba(16,185,129,.1); border-color: #059669; color: #6ee7b7;
  box-shadow: 0 0 24px rgba(16,185,129,.25);
  animation: vg-fade .5s ease both, vg-pulse 3s ease-in-out infinite;
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

HEAD = ""

RADIO_JS = ("(v)=>{const dark=(v===null||v.indexOf('🌙')>=0);"
            "document.documentElement.classList.toggle('vg-dark',dark);"
            "localStorage.setItem('vg-dark',dark?'1':'0');return v;}")

LOAD_JS = ("()=>{const saved=localStorage.getItem('vg-dark');"
           "const dark=(saved===null||saved==='1');"
           "document.documentElement.classList.toggle('vg-dark',dark);"
           "return dark?'🌙 黑夜':'☀️ 白昼';}")

# 切换语言时同步翻译 gradio 内置文案（4.44 无 i18n 接口，只能 DOM 文本替换；
# 多次延时重试以覆盖组件重渲染）
LANG_JS = r"""(v)=>{
  const en = v==='English';
  const pairs = en ? [
    ['点击上传，或将音频拖放到此处', 'Click to upload, or drop audio here'],
    ['将音频拖放到此处，或点击上传', 'Drop audio here, or click to upload'],
    ['将音频拖放到此处', 'Drop audio here'],
    ['点击上传', 'Click to upload'],
    ['或', 'or'],
    ['清除', 'Clear'],
    ['录制', 'Record'],
  ] : [
    ['Click to upload, or drop audio here', '点击上传，或将音频拖放到此处'],
    ['Drop audio here, or click to upload', '将音频拖放到此处，或点击上传'],
    ['Drop audio here', '将音频拖放到此处'],
    ['Click to upload', '点击上传'],
    ['or', '或'],
    ['Clear', '清除'],
    ['Record', '录制'],
  ];
  const patch = ()=>{
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(n=>{
      if (!n.nodeValue) return;
      let s = n.nodeValue;
      for (const [a,b] of pairs) s = s.split(a).join(b);
      if (s !== n.nodeValue) n.nodeValue = s;
    });
  };
  [150, 600, 1500].forEach(ms => setTimeout(patch, ms));
  return v;
}"""


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
    str_ = pred_conv != "-" and pred_conv != tgt
    screened_ok = identity_verdict(asv, tgt)["screened_identity_ok"]
    if not identity_verdict(asv, tgt)["clean_vc_identity_ok"]:
        head = ("⚠️ 干净转换未通过身份筛选，不能归因于防护" if lang == "zh" else
                "Clean conversion failed identity screening; defense cannot be credited")
    elif screened_ok:
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
    cls = "vg-status-ok" if screened_ok else "vg-status-fail"
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
    return (f'<div class="vg-hero"><div class="vg-orb vg-orb1"></div><div class="vg-orb vg-orb2"></div>'
            f'<h2>{L["hero_title"]}</h2><p>{L["hero_sub"]}</p>'
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
                                             "docs", "figures", "2026-11-09", "01_quality_comparison.png"))

    with gr.Blocks(title="Voice Guard — Voice Privacy Defense", theme=theme,
                   css=CSS) as demo:
        # 顶部控制条：昼夜选项 + 语言选项（格式统一）
        with gr.Row():
            theme_radio = gr.Radio(choices=[L["theme_dark"], L["theme_light"]],
                                   value=L["theme_dark"],
                                   label="🌗 主题 / Theme", scale=1)
            lang_dd = gr.Dropdown(choices=["中文", "English"],
                                  value="中文" if lang == "zh" else "English",
                                  label=L["lang_label"], scale=1)

        # 页面加载时按 localStorage 恢复主题并同步选项
        demo.load(None, [], [theme_radio], js=LOAD_JS)

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
                method_dd = gr.Dropdown(choices=["vg", "vg-projected"], value="vg",
                    label="保护方法 / Method",
                    info="vg-projected 额外优化 1000 步；实验性质量改进，需核查每次结果")
                context_cb = gr.Checkbox(value=False, label="保留完整音频 / Preserve full audio",
                    info="保留安静的开头和结尾；可能改变转换与评分结果")

            with gr.Group():
                g3_md = gr.Markdown(f"### {L['g3']}")
                run_btn = gr.Button(L["run_btn"], variant="primary", size="lg")

            with gr.Group():
                g4_md = gr.Markdown(f"### {L['g4']}")
                status_html = gr.HTML(f'<div class="vg-note">{L["status_placeholder"]}</div>')
                metrics_html = gr.HTML()
                with gr.Tabs():
                    with gr.Tab(L["t_try"]) as tt1:
                        with gr.Row():
                            aud_orig = gr.Audio(label=L["a_orig"])
                            aud_def = gr.Audio(label=L["a_def"])
                        with gr.Row():
                            aud_delta = gr.Audio(label=L["a_delta"])
                            aud_conv_clean = gr.Audio(label=L["a_cc"])
                            aud_conv_def = gr.Audio(label=L["a_cd"])
                    with gr.Tab(L["t_plot"]) as tt2:
                        plot_out = gr.Image(label=L["plot_label"])
                    with gr.Tab(L["t_detail"]) as tt3:
                        detail_html = gr.HTML()

            last_state = gr.State({"lang": lang, "report": None, "mcd": None})

            def on_protect(defend, speaker, adv, eps, demo_mode, state, method, preserve_context,
                           progress=gr.Progress(track_tqdm=False)):
                CL = T[state.get("lang", "zh")]
                if not defend:
                    raise gr.Error(CL["err_no_defend"])
                if not adv:
                    y = pipe.least_likely(speaker)
                    if y in speaker_first:
                        adv = speaker_first[y]
                        gr.Info(CL["info_auto_adv"].format(y))
                    else:
                        raise gr.Error(CL["err_need_adv"])
                it1, it2 = (300, 200) if demo_mode else (args.iters1, args.iters2)
                out_dir = os.path.join(args.out_dir, f"{speaker}_{uuid.uuid4().hex}")
                gr.Info(CL["info_start"].format(it1, it2))

                # 实时进度：保护在独立线程执行，主线程轮询进度回调更新进度条
                box = {"frac": 0.0, "desc": ""}

                def _cb(frac, desc=""):
                    box["frac"] = max(box["frac"], float(frac))
                    if desc:
                        box["desc"] = desc

                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                    fut = _ex.submit(pipe.protect, defend, speaker, adv, eps=eps,
                                     iters1=it1, iters2=it2, out_dir=out_dir, progress=False,
                                     refine_steps=1000 if method == "vg-projected" else 0,
                                     preserve_context=preserve_context, progress_cb=_cb)
                    while not fut.done():
                        progress(min(0.99, box["frac"]), desc=box["desc"] or "protect")
                        time.sleep(0.4)
                    rep = fut.result()
                progress(1.0, desc="done")

                mcd_val = None
                try:
                    from vguard.metrics.mcd import mcd_mel
                    w1, _ = sf.read(rep["files"]["original"])
                    w2, _ = sf.read(rep["files"]["defended"])
                    mcd_val = float(mcd_mel(w1, w2, pipe.pp, attr=None, device=pipe.device))
                except Exception:  # noqa: BLE001
                    pass

                png = _plot_waveforms(rep["files"], os.path.join(out_dir, "waves.png"))
                state = {"lang": state.get("lang", "zh"), "report": rep, "mcd": mcd_val}
                return (_status_html(rep, state["lang"]), rep["files"]["original"],
                        rep["files"]["defended"], rep["files"]["delta"],
                        rep["files"]["conv_clean"], rep["files"]["conv_def"], png,
                        _metrics_html(rep, mcd_val, state["lang"]),
                        _detail_html(rep, state["lang"]), state)

            run_btn.click(on_protect,
                          [defend_in, speaker_dd, adv_in, eps_sl, demo_cb, last_state, method_dd, context_cb],
                          [status_html, aud_orig, aud_def, aud_delta, aud_conv_clean,
                           aud_conv_def, plot_out, metrics_html, detail_html, last_state],
                          concurrency_id="gpu", concurrency_limit=1)

            sample_btn = gr.Button(L["sample_btn"], size="sm", visible=bool(speaker_first))
            if speaker_first:

                def on_sample():
                    name = next(s for s in pipe.speakers if s in speaker_first)
                    return gr.update(value=speaker_first[name]), name

                sample_btn.click(on_sample, None, [defend_in, speaker_dd])

        with gr.Tab(L["tab_convert"]) as tab2:
            conv_md = gr.Markdown(L["conv_md"])
            with gr.Row():
                src_in = gr.Audio(label=L["src_label"], type="filepath")
                tgt_in = gr.Audio(label=L["tgt_label"], type="filepath")
            conv_btn = gr.Button(L["conv_btn"], variant="primary")
            conv_out = gr.Audio(label=L["conv_out_label"])

            def on_convert(src, tgt, state):
                CL = T[state.get("lang", "zh")]
                if not src or not tgt:
                    raise gr.Error(CL["err_conv"])
                out = os.path.join(args.out_dir, f"conv_{uuid.uuid4().hex}.wav")
                from vguard.vc.adainvc import convert as vc_convert
                vc_convert(pipe.model, pipe.config, pipe.attr, src, tgt, out, pipe.device)
                return out, state

            conv_btn.click(on_convert, [src_in, tgt_in, last_state], [conv_out, last_state],
                           concurrency_id="gpu", concurrency_limit=1)

        with gr.Tab(L["tab_about"]) as tab3:
            about_md = gr.Markdown(L["about_md"])
            curve_out = None
            if os.path.exists(curve_img):
                curve_out = gr.Image(value=curve_img, label=L["curve_label"], height=420)

        # ---- 语言切换：重写全部文案 ----
        def on_lang(choice, speaker, theme_val, state):
            nl = "zh" if choice == "中文" else "en"
            NL = T[nl]
            rep = state.get("report")
            state = {"lang": nl, "report": rep, "mcd": state.get("mcd")}
            keep_dark = "🌙" in str(theme_val)
            return [
                gr.update(value=_hero_html(nl)),                    # 0 hero
                gr.update(value=_steps_html(nl)),                   # 1 steps
                gr.update(choices=[NL["theme_dark"], NL["theme_light"]],
                          value=NL["theme_dark"] if keep_dark else NL["theme_light"]),  # 2 theme_radio
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
                state,                                              # 29 state（含 lang）
                gr.update(label=NL["tab_protect"]),                 # 30 tab1
                gr.update(label=NL["tab_convert"]),                 # 31 tab2
                gr.update(label=NL["tab_about"]),                   # 32 tab3
                gr.update(label=NL["t_try"]),                       # 33 tt1
                gr.update(label=NL["t_plot"]),                      # 34 tt2
                gr.update(label=NL["t_detail"]),                    # 35 tt3
                gr.update(value=NL["sample_btn"]),                  # 36 sample_btn
            ]

        lang_dd.change(on_lang, [lang_dd, speaker_dd, theme_radio, last_state],
                       [hero_html, steps_html, theme_radio, lang_dd, g1_md, defend_in, adv_in,
                        speaker_dd, g2_md, eps_sl, demo_cb, g3_md, run_btn, g4_md,
                        status_html, aud_orig, aud_def, aud_delta, aud_conv_clean, aud_conv_def,
                        plot_out, conv_md, src_in, tgt_in, conv_btn, conv_out, about_md,
                        metrics_html, detail_html, last_state, tab1, tab2, tab3,
                        tt1, tt2, tt3, sample_btn])

        # 同步翻译 gradio 内置文案（音频拖放提示等）
        lang_dd.change(None, [lang_dd], None, js=LANG_JS)

        # 昼夜切换（纯前端 JS + localStorage 记忆，与语言选项格式统一）
        theme_radio.change(None, [theme_radio], None, js=RADIO_JS)

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
    demo.queue(max_size=8, default_concurrency_limit=1)
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
