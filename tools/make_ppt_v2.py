"""验收 PPT v2：深色科技风全量重做（外观 + 内容）。"""
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

OUT = r"D:\Voice Guard\docs\验收PPT_v2.pptx"
CURVE = r"D:\Voice Guard\docs\fig_eps_curve.png"
CURVE_DARK = r"D:\Voice Guard\docs\fig_eps_curve_dark.png"

BG = RGBColor(0x0B, 0x12, 0x20)
PANEL = RGBColor(0x12, 0x1B, 0x31)
PANEL2 = RGBColor(0x17, 0x22, 0x3B)
INK = RGBColor(0xE8, 0xEE, 0xF7)
SUB = RGBColor(0x9F, 0xB0, 0xC9)
ACC1 = RGBColor(0x81, 0x8C, 0xF8)
ACC2 = RGBColor(0x22, 0xD3, 0xEE)
GREEN = RGBColor(0x34, 0xD3, 0x99)
RED = RGBColor(0xF8, 0x71, 0x71)
AMBER = RGBColor(0xFB, 0xBF, 0x24)
BORDER = RGBColor(0x24, 0x32, 0x4B)
FONT = "Microsoft YaHei"
MONO = "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def _bg(slide, color=BG):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def _box(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    sp = slide.shapes.add_shape(shape, x, y, w, h)
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(1)
    sp.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            sp.adjustments[0] = radius
        except Exception:  # noqa: BLE001
            pass
    return sp


def _text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
          font=FONT, anchor=MSO_ANCHOR.TOP, line_spacing=1.12):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = runs if isinstance(runs, list) else [runs]
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = line_spacing
        if isinstance(line, tuple) and len(line) == 2 and isinstance(line[0], str):
            segs = [line]
        elif isinstance(line, str):
            segs = [(line, {})]
        else:
            segs = list(line)
        for text, kw in segs:
            r = para.add_run()
            r.text = text
            f = r.font
            f.name = kw.get("font", font)
            f.size = Pt(kw.get("size", size))
            f.bold = kw.get("bold", bold)
            f.color.rgb = kw.get("color", color)
    return tb


def _header(slide, kicker, title, page):
    _box(slide, Inches(0.55), Inches(0.42), Inches(0.14), Inches(0.86), fill=ACC1)
    _box(slide, Inches(0.55), Inches(0.42), Inches(0.07), Inches(0.86), fill=ACC2)
    _text(slide, Inches(0.85), Inches(0.34), Inches(9.5), Inches(0.3), kicker,
          size=11, color=ACC2, bold=True, font=FONT)
    _text(slide, Inches(0.85), Inches(0.58), Inches(10.5), Inches(0.6), title,
          size=26, color=INK, bold=True)
    _text(slide, Inches(12.3), Inches(6.95), Inches(0.7), Inches(0.35), f"{page:02d}",
          size=11, color=SUB, align=PP_ALIGN.RIGHT)
    _text(slide, Inches(0.55), Inches(6.95), Inches(6), Inches(0.35),
          "Voice Guard 复现 · IJCAI 2023 · 选题 A036", size=10, color=SUB)


def _card(slide, x, y, w, h, title, lines, accent=ACC1, tsize=14, lsize=12.5):
    _box(slide, x, y, w, h, fill=PANEL, line=BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    _box(slide, x, y, Inches(0.09), h, fill=accent)
    _text(slide, x + Inches(0.22), y + Inches(0.14), w - Inches(0.4), Inches(0.4), title,
          size=tsize, color=INK, bold=True)
    _text(slide, x + Inches(0.22), y + Inches(0.56), w - Inches(0.4), h - Inches(0.7), lines,
          size=lsize, color=SUB, line_spacing=1.18)


def _table(slide, x, y, w, rows_data, col_w=None, header_color=RGBColor(0x1E, 0x2A, 0x47),
           row_h=0.42, hsize=13, bsize=12.5):
    n_rows = len(rows_data)
    n_cols = len(rows_data[0])
    gt = slide.shapes.add_table(n_rows, n_cols, x, y, w, Inches(row_h * n_rows)).table
    if col_w:
        for i, cw in enumerate(col_w):
            gt.columns[i].width = Inches(cw)
    for ri, row in enumerate(rows_data):
        for ci, cell in enumerate(row):
            c = gt.cell(ri, ci)
            c.fill.solid()
            c.fill.fore_color.rgb = header_color if ri == 0 else (PANEL2 if ri % 2 else PANEL)
            c.margin_left = Inches(0.08)
            c.margin_right = Inches(0.08)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            para = tf.paragraphs[0]
            para.alignment = PP_ALIGN.CENTER if (ri == 0 or ci > 0) else PP_ALIGN.LEFT
            r = para.add_run()
            r.text = str(cell)
            f = r.font
            f.name = FONT
            f.size = Pt(hsize if ri == 0 else bsize)
            f.bold = ri == 0
            f.color.rgb = INK if ri == 0 else (ACC2 if isinstance(cell, str) and cell.startswith("**") else SUB)
    return gt


# ---------------------------------------------------------------- 1 封面
s = prs.slides.add_slide(BLANK)
_bg(s)
_box(s, Inches(0), Inches(0), Inches(13.333), Inches(7.5), fill=BG)
_box(s, Inches(-1.2), Inches(-1.4), Inches(6), Inches(4.6), fill=RGBColor(0x1E, 0x1B, 0x4B),
     shape=MSO_SHAPE.OVAL)
_box(s, Inches(9.6), Inches(4.6), Inches(6), Inches(4.6), fill=RGBColor(0x11, 0x3A, 0x47),
     shape=MSO_SHAPE.OVAL)
_box(s, Inches(0.9), Inches(2.02), Inches(0.2), Inches(1.05), fill=ACC1)
_box(s, Inches(0.9), Inches(2.02), Inches(0.1), Inches(1.05), fill=ACC2)
_text(s, Inches(1.3), Inches(1.55), Inches(11), Inches(0.4), "语音识别与合成 · 课程大作业",
      size=15, color=ACC2, bold=True)
_text(s, Inches(1.3), Inches(2.0), Inches(11.5), Inches(1.1), "Voice Guard：语音隐私的主动防御",
      size=44, color=INK, bold=True)
_text(s, Inches(1.3), Inches(3.12), Inches(11.2), Inches(0.45),
      "复现 IJCAI 2023《Voice Guard》时域对抗扰动 + 心理声学掩蔽 · 并探索真实信道下的失效与改进",
      size=17, color=SUB)
_text(s, Inches(1.3), Inches(4.1), Inches(8), Inches(0.4),
      [("时域扰动 ⏱    心理声学掩蔽 👂    多维判定 🧪", {"color": RGBColor(0xC7, 0xD2, 0xFE), "size": 14})])
_text(s, Inches(1.3), Inches(5.6), Inches(11), Inches(0.9),
      ["组员：成员 A（算法与实验）　成员 B（工程与部署）",
       "选题 A036 / Audio / 音频主动防御 · 验收 2025-12-22",
       "github.com/Abyss-sword/Voice-Guard"],
      size=14, color=SUB)

# ---------------------------------------------------------------- 2 背景与问题
s = prs.slides.add_slide(BLANK)
_bg(s)
_header(s, "BACKGROUND", "背景与问题：为什么要主动防御？", 2)
_text(s, Inches(0.85), Inches(1.45), Inches(11.6), Inches(0.9),
      [("零样本语音转换（AdaIN-VC / VQMIVC）：任意内容 + 目标声纹 → 逼真伪造语音", {"bold": True, "color": INK}),
       "→ 电信诈骗与声纹伪造。被动检测只能事后止损 → 论文主张发布前注入不可感知扰动的主动防御。"],
      size=15, color=SUB)
_card(s, Inches(0.85), Inches(2.5), Inches(3.83), Inches(3.9), "已有方法的缺陷",
      ["Attack-VC（SLT'21）在 mel 频域加扰动：", "· 需经声码器上采样回波形，跨域损耗稀释扰动",
       "· 评价只看身份变化，不看质量与内容 → 伪防御"], accent=RED, tsize=14.5)
_card(s, Inches(4.88), Inches(2.5), Inches(3.83), Inches(3.9), "本文的应对",
      ["① 扰动推进到时域（免跨域损耗），为此重写可微 mel 提取器 H",
       "② 首次引入心理声学掩蔽（θx 阈下噪声，人耳不可闻）",
       "③ 标准化三维判定：身份 + 自然度 + 内容"], accent=GREEN, tsize=14.5)
_card(s, Inches(8.91), Inches(2.5), Inches(3.6), Inches(3.9), "我们的复现目标",
      ["完整复现核心方法（Eq.3/7/8、Alg.1）",
       "搭起可运行系统（Gradio + 华为云）",
       "抽样实验验证 + 真实信道加分项探索"], accent=ACC1, tsize=14.5)

# ---------------------------------------------------------------- 3 核心方法
s = prs.slides.add_slide(BLANK)
_bg(s)
_header(s, "METHOD", "核心方法：时域扰动 × 心理声学掩蔽 × 两阶段", 3)
_box(s, Inches(0.85), Inches(1.4), Inches(6.0), Inches(3.1), fill=PANEL, line=BORDER,
     shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
_text(s, Inches(1.1), Inches(1.56), Inches(5.6), Inches(0.35), "攻击公式（Eq.3/7）", size=14, color=ACC2, bold=True)
for i, (eq, note) in enumerate([
        ("δ = 2ε · tanh(w)", "时域扰动，幅值 ≤ 2ε"),
        ("L_vc = MSE(Es∘H(x+δ), Es(ŷ)) − 0.1·MSE(Es∘H(x+δ), Es(x))", "移向 least-likely 说话人 ŷ，同时远离自己"),
        ("L_θ = mean(ReLU(PSD_δ − θx))", "噪声压在掩蔽阈值 θx 之下（hinge）"),
        ("L = L_vc + c · L_θ", "c 由二分搜索控制（Alg.1，≤8 轮）"),
]):
    _text(s, Inches(1.1), Inches(1.95) + Inches(0.62) * i, Inches(5.6), Inches(0.55),
          [(eq, {"font": MONO, "size": 12.5, "color": INK}), ("   " + note, {"size": 11, "color": SUB})])
_box(s, Inches(7.05), Inches(1.4), Inches(5.45), Inches(3.1), fill=PANEL, line=BORDER,
     shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
_text(s, Inches(7.3), Inches(1.56), Inches(5), Inches(0.35), "两阶段 + c 二分（Algorithm 1）", size=14, color=ACC2, bold=True)
_text(s, Inches(7.3), Inches(1.98), Inches(5), Inches(2.4),
      [("阶段 1 · 3000 步", {"bold": True, "color": INK}),
       "纯 L_vc 求「强」：先把身份翻转做出来",
       ("阶段 2 · 1500 步", {"bold": True, "color": INK}),
       "加 c·L_θ 求「隐」：c 成功×2 / 失败折半，最多 8 轮",
       "我们的验证：ε=0.01 时 loss_th 1500 步收敛到 ~5 dB（噪声真正藏进阈下）"],
      size=12.5, color=SUB, line_spacing=1.25)
_text(s, Inches(0.85), Inches(4.75), Inches(11.6), Inches(1.7),
      [("判定（Eq.9-11）：", {"bold": True, "color": INK}),
       "F_imp：被保护音频仍被识别为 X 且 MOS≥3、ASR≥10%；",
       "F_str：被保护后经 VC 转换不再被识别为 X（或质量/内容崩坏）；",
       "F_d = F_imp ∧ F_str 才算防御成功。"],
      size=14.5, color=SUB, line_spacing=1.3)

# ---------------------------------------------------------------- 4 复现工程
s = prs.slides.add_slide(BLANK)
_bg(s)
_header(s, "ENGINEERING", "复现工程：从残缺开源到可运行系统", 4)
_table(s, Inches(0.85), Inches(1.5), Inches(7.4),
       [["官方发布", "本仓库的处理"],
        ["仅 3 个文件，缺 Transform / asr_model", "补全扰动 PSD 与 ASR 模块并忠实转录"],
        ["VC 权重在 Google Drive（不可达）", "官方 AdaIN-VC 权重（NTU 直链，可核验）"],
        ["评估 GE2E 未开源", "独立 ECAPA（主口径）+ Es 线性探头（val 94.5%）"],
        ["48kHz VCTK", "22kHz 镜像（hf-mirror）"]],
       col_w=[4.0, 3.4])
_text(s, Inches(8.5), Inches(1.5), Inches(4.0), Inches(5.0),
      [("关键验证", {"bold": True, "color": ACC2, "size": 14}),
       "· 可微 H 与官方预处理逐点误差 ≤1e-4（float32）",
       "· θx / Transform 同 96dB 尺度归一化（测试全过）",
       "· 双注册库：攻击用 Es / 评估用 ECAPA",
       "· 权重/数据/依赖全链 curl 断点续传（受限网络可复现）",
       "· 系统：pipeline 服务 + Gradio 前端（中英双语/昼夜双模）+ Docker + 华为云手册"],
      size=13, color=SUB, line_spacing=1.3)

# ---------------------------------------------------------------- 5 实验结果
s = prs.slides.add_slide(BLANK)
_bg(s)
_header(s, "RESULTS", "实验结果：核心主张完整复现", 5)
if os.path.exists(CURVE_DARK):
    s.shapes.add_picture(CURVE_DARK, Inches(0.85), Inches(1.42), width=Inches(6.2))
else:
    _box(s, Inches(0.85), Inches(1.42), Inches(6.2), Inches(3.6), fill=PANEL, line=BORDER)
    _text(s, Inches(0.85), Inches(2.6), Inches(6.2), Inches(0.5), "（ε 曲线图）", align=PP_ALIGN.CENTER, color=SUB)
_table(s, Inches(7.25), Inches(1.42), Inches(5.25),
       [["方法（ε=0.05，筛选后）", "防御率"],
        ["频域 emb", "0%"],
        ["频域 fb", "25%"],
        ["频域 e2e", "0%"],
        ["时域 vg-nop（消融）", "75%"],
        ["**VG（时域+掩蔽）", "**75%"]],
       col_w=[3.6, 1.65])
_text(s, Inches(0.85), Inches(5.15), Inches(11.6), Inches(1.9),
      [("ε 权衡曲线（左，筛选后）：", {"bold": True, "color": INK}),
       "0.02→75% · 0.05→75% · 0.08→50% · 0.10→50%；"
       "低 ε 平台受「顽固句」限制（F_str），高 ε 由 F_imp 退化主导 —— 最优工作点 ε≈0.05。",
       "频域基线筛选后几乎全失败（emb 0/4、e2e 0/4、fb 1/4）；MCD 频域 ~3 vs 时域 ~35–44。",
       ("云端 T4 实测：", {"bold": True, "color": ACC2}),
       "3000/1500 步 194.67s/句、CUDA 峰值 175.8MiB；身份维通过（p225→p225 / 转换后 p234），"
       "但 MOS 3.33→1.02、WER 0→40%（未复现「强且不可感知」）。",
       ("质量改进 refine：", {"bold": True, "color": ACC2}),
       "平均 MOS 1.397→3.192、平均逐句 WER 62.76%→5.71%（固定参数验证联合门槛 1/2，小样本）。"],
      size=13, color=SUB, line_spacing=1.28)

# ---------------------------------------------------------------- 6 加分项
s = prs.slides.add_slide(BLANK)
_bg(s)
_header(s, "BONUS", "加分项：真实信道下的失效模式与 EOT 改进", 6)
_table(s, Inches(0.85), Inches(1.45), Inches(5.0),
       [["信道", "VG", "VG+EOT"],
        ["none", "60%", "40%"],
        ["mp3-128k", "80%", "60%"],
        ["mp3-32k", "60%", "60%"],
        ["aac-64k", "40%", "40%"],
        ["G.711", "60%", "**20%"],
        ["8kHz", "60%", "**20%"]],
       col_w=[2.0, 1.5, 1.5])
_text(s, Inches(6.1), Inches(1.45), Inches(6.4), Inches(5.1),
      [("三个发现（控制变量，6 信道 × 5 句）", {"bold": True, "color": ACC2, "size": 14}),
       ("① mp3/aac 几乎不影响防御", {"color": GREEN}),
       "　转换相似度 ~0.107 不变——有损编码没有「抹掉」扰动",
       ("② 8k/G.711 双向分化", {"color": AMBER}),
       "　部分句子攻击被削弱（sim 0.071→0.168），部分反而更不像 X",
       ("③ 真正薄弱环是 F_imp", {"color": RED}),
       "　8k 使被保护音频自身相似度降 0.16（0.447→0.284）",
       ("EOT 负面结果：", {"bold": True, "color": INK}),
       "攻击侧信道稳健训练让防御全面变差（机理：强化攻击 → 伤及 F_imp）；"
       "正确方向 = 防御侧 min_δ [攻击项+掩蔽项+F_imp-信道项]"],
      size=13, color=SUB, line_spacing=1.32)

# ---------------------------------------------------------------- 7 总结
s = prs.slides.add_slide(BLANK)
_bg(s)
_header(s, "SUMMARY", "总结与展望", 7)
_card(s, Inches(0.85), Inches(1.6), Inches(3.83), Inches(3.2), "已完成",
      ["完整复现（H 对齐 1e-4 / θx / 两阶段）",
       "核心主张复现：时域 75% ≫ 频域 25%",
       "ε 权衡曲线 + 双区间机理",
       "信道分析 + EOT 受控负面结果"], accent=GREEN)
_card(s, Inches(4.88), Inches(1.6), Inches(3.83), Inches(3.2), "局限（如实声明）",
      ["未训练（资源不计入难度）",
       "抽样规模小于论文；单句 2–4s 需 20–60min",
       "AdaIN-VC+GL 质量上限 → F_str 质量维恒真",
       "评估 ASV 域差异影响绝对值（趋势可复现）"], accent=AMBER)
_card(s, Inches(8.91), Inches(1.6), Inches(3.6), Inches(3.2), "展望",
      ["防御侧信道稳健损失（F_imp-信道项）",
       "黑盒迁移（VQMIVC）验证",
       "多信道联合稳健 + 实时化"],
      accent=ACC1)
_text(s, Inches(0.85), Inches(5.2), Inches(11.6), Inches(0.6),
      "致谢：IJCAI 2023 论文作者的开源工作；官方 AdaIN-VC 权重；hf-mirror 数据镜像。",
      size=12.5, color=SUB, align=PP_ALIGN.CENTER)

# ---------------------------------------------------------------- 8 分工
s = prs.slides.add_slide(BLANK)
_bg(s)
_header(s, "TEAM", "分工与贡献", 8)
_card(s, Inches(0.85), Inches(1.7), Inches(5.6), Inches(3.4), "成员 A · 算法与实验",
      ["可微特征 H 与掩蔽阈 θx、攻击算法（两阶段+c 二分）",
       "ASV 探头（Es 线性分类，val 94.5%）",
       "ε 扫描 / 方法对比 / 消融实验",
       "信道失效分析与 EOT 对照实验"], accent=ACC1)
_card(s, Inches(6.9), Inches(1.7), Inches(5.6), Inches(3.4), "成员 B · 工程与部署",
      ["受限网络环境基建（curl wheel 安装方案）",
       "双注册库与评估栈（ECAPA / Whisper / MOSNet）",
       "Gradio 前端（双语/昼夜）与 Docker/华为云部署物料",
       "报告、图表与 PPT"], accent=ACC2)
_text(s, Inches(0.85), Inches(5.5), Inches(11.6), Inches(0.5),
      "（姓名以最终提交版报告 §6 为准）", size=12, color=SUB, align=PP_ALIGN.CENTER)

prs.save(OUT)
print("saved:", OUT)
