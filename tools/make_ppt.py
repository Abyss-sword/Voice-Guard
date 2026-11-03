"""生成验收 PPT 草稿（9 分钟 / 7 页），结果数字后续用脚本填充更新。"""
import sys

from pptx import Presentation
from pptx.util import Inches, Pt

OUT = r"D:\Voice Guard\docs\验收PPT_草稿.pptx"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]


def slide(title, bullets):
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(0.6), Inches(0.3), Inches(12.1), Inches(1.0))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    body = s.shapes.add_textbox(Inches(0.7), Inches(1.4), Inches(11.9), Inches(5.7))
    tf = body.text_frame
    for i, b in enumerate(bullets):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.text = b
        para.font.size = Pt(18)
        para.level = 0
    return s


slide("Voice Guard 复现与探索——语音隐私的主动防御", [
    "选题 A036 · Audio / 音频主动防御 · IJCAI 2023 (Li et al., pp.4812-4820)",
    "组员：A（算法与实验）  B（工程与部署）",
    "交付：可运行系统（前端 + 部署）+ 核心方法复现 + 实验验证 + 加分项",
])

slide("背景与核心问题", [
    "零样本语音转换（AdaIN-VC / VQMIVC）：任意内容 → 任意声纹 → 电信诈骗/伪造",
    "被动检测只能事后止损 → 论文主张主动防御：发布前注入不可感知扰动",
    "已有方法（Attack-VC, SLT'21）缺陷：① 频域扰动经声码器上采样 = 跨域损耗；② 评价只看身份",
    "论文三贡献：时域扰动（可微 mel 提取 H）· 心理声学掩蔽（首次用于 VC 防御）· 多维评价标准化",
])

slide("核心方法：时域 + 掩蔽 + 两阶段", [
    "δ = 2ε·tanh(w)；L_vc = MSE(Es∘H(x+δ), Es(y)) − 0.1·MSE(Es∘H(x+δ), Es(x))（y = least-likely 说话人）",
    "心理声学：掩蔽者识别 → 双斜率扩散 → 全局掩蔽阈 θx；L_θ = mean(ReLU(PSD_δ − θx))",
    "联合 L = L_vc + c·L_θ；两阶段：3000 步求「强」→ 1500 步 × c 二分（≤8 轮）求「隐」",
    "判定 Eq.9-11：F_imp（defended 仍为 X 且 MOS≥3、ASR≥10%）∧ F_str（转换后不再为 X）",
    "【图】掩蔽阈值 vs 扰动 PSD：噪声藏在人耳听阈之下",
])

slide("复现工程：从残缺开源到可运行系统", [
    "官方仓库仅 3 个文件，缺失 Transform（扰动 PSD）与 asr_model → 补全并忠实转录",
    "可微 mel 提取 H 与官方预处理数值对齐 ≤1e-4（float32 精度；librosa 1.0 兼容显式处理）",
    "权重：官方 AdaIN-VC（NTU 直链，24kHz/512 维 mel）；数据：VCTK 22kHz 镜像",
    "评估栈：攻击用 Es 注册库 · 评估用独立 ECAPA（对应论文 GE2E 角色）· Es 线性探头 val 94.5%",
    "系统：pipeline 服务 + Gradio 前端 + Docker/华为云 ModelArts 部署",
])

slide("实验结果（抽样复现，ECAPA 口径 + 论文筛选协议）", [
    "ε 扫描（5 句 × 4 档，clean-VC 筛选后）：防御率 0.02→75%、0.05→75%、0.08→50%、0.10→50%",
    "权衡曲线双区间：低 ε 平台（F_imp 100%，唯一失败=F_str 顽固句）；高 ε 下滑（F_imp 退化，扰动伤及自身声纹）",
    "最优工作点 ε≈0.05；MCD 28.9→44.8 单调上升",
    "筛选协议（论文 Sec.5.3）如实执行：1 句 clean-VC 失败被剔除并记录",
    "【图】防御率/F_imp/F_str/MCD 四线图（docs/fig_eps_curve.png）",
])

slide("加分项：真实信道下的失效模式与 EOT 改进", [
    "原假设「压缩抹掉扰动」仅部分成立：mp3/aac 对转换相似度几乎无影响（~0.1 不变）",
    "电话信道（8k/G.711）双向分化：部分句子攻击被削弱（p227_044 conv 相似度 0.071→0.168），部分反而更不像 X",
    "真正脆弱点是 F_imp：8k 使 defended 自身相似度降最多 0.16（p227_044: 0.447→0.284）",
    "改进：EOT 可微信道稳健训练（可微重采样/低通/量化/增益，生成器 CPU 化修复）",
    "【数字】VG vs VG+EOT 信道稳健性对比（exp04 实验运行中，验收前填）",
])

slide("总结", [
    "完整复现核心方法 + 可运行系统 + 真实场景增量（加分项）",
    "局限：未训练、抽样规模、单句时耗（4s 音频 @3000+1500 步 ≈ 40min）",
    "展望：黑盒迁移（VQMIVC）、多信道联合稳健、实时化",
    "分工：A=特征/攻击/实验；B=工程/前端/部署",
])

prs.save(OUT)
print("saved:", OUT)
