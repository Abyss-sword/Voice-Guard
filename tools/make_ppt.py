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

slide("实验结果（抽样复现）", [
    "【T1 表】emb / fb / e2e / VG w/o P / VG × Defend / MCD / ASV(def) / ASV(conv) / 筛选率",
    "预期形态：VG 显著优于频域基线；ASV(def)≈X 而 ASV(conv)≉X",
    "【曲线】防御率 vs MCD 分桶（复现论文 Fig.3 只升不降）",
    "筛选协议（论文 Sec.5.3）与筛选通过率如实报告",
])

slide("加分项：真实信道下的失效与改进", [
    "问题：微信/抖音/电话二次压缩抹掉时域扰动（8kHz 切高频；编码器丢弃低能量掩蔽噪声）",
    "验证：mp3/aac/opus/G.711 各码率 × 防御率衰减曲线（控制变量）",
    "机理：扰动 PSD / 编码器量化底 / 掩蔽阈三线图",
    "改进：EOT 信道稳健训练（可微重采样/低通/量化/增益）+ 码率感知能量分配",
    "【数字】32k mp3：VG ~30% → VG+EOT ~55%（待实验填充）",
])

slide("总结", [
    "完整复现核心方法 + 可运行系统 + 真实场景增量（加分项）",
    "局限：未训练、抽样规模、单句时耗（4s 音频 @3000+1500 步 ≈ 40min）",
    "展望：黑盒迁移（VQMIVC）、多信道联合稳健、实时化",
    "分工：A=特征/攻击/实验；B=工程/前端/部署",
])

prs.save(OUT)
print("saved:", OUT)
