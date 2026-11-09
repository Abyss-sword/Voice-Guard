"""Render the course Markdown report to a Unicode PDF, plus page PNGs for review.

Requires reportlab and pypdfium2; the Codex bundled Python includes these packages.
"""
from __future__ import annotations
import argparse
from html import escape
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether


def markup(text):
    text=text.translate(str.maketrans({'−':'-','–':'-','—':'-','≥':'>=','≤':'<='}))
    text=escape(text)
    text=re.sub(r'`([^`]+)`',r'\1',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',text)
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('docs/04_报告.md'))
    parser.add_argument('--output',type=Path,default=Path('deliverables/Voice_Guard_课程报告.pdf'))
    parser.add_argument('--preview-dir',type=Path,default=Path('outputs/course/report-pages'))
    parser.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/simsun.ttc'))
    args=parser.parse_args()
    pdfmetrics.registerFont(TTFont('CourseCN',str(args.font),subfontIndex=0))
    pdfmetrics.registerFontFamily('CourseCN',normal='CourseCN',bold='CourseCN',italic='CourseCN',boldItalic='CourseCN')
    body=ParagraphStyle('body',fontName='CourseCN',fontSize=10.5,leading=17,
                        spaceAfter=7,wordWrap='CJK',alignment=TA_JUSTIFY,textColor=colors.black,
                        allowWidows=0,allowOrphans=0)
    title=ParagraphStyle('title',parent=body,fontSize=21,leading=29,spaceAfter=14,keepWithNext=True)
    h1=ParagraphStyle('h1',parent=body,fontSize=15,leading=21,spaceBefore=14,spaceAfter=8,keepWithNext=True)
    h2=ParagraphStyle('h2',parent=body,fontSize=12,leading=18,spaceBefore=9,spaceAfter=6,keepWithNext=True)
    cell=ParagraphStyle('cell',parent=body,fontSize=8.4,leading=12,spaceAfter=0,alignment=0)
    caption=ParagraphStyle('caption',parent=body,fontSize=9,leading=14,alignment=0,spaceAfter=10)
    lines=args.input.read_text(encoding='utf-8').splitlines()
    story=[];width=A4[0]-44*mm
    index=0
    while index<len(lines):
        line=lines[index].strip()
        if not line or line.startswith('<!--'):
            index+=1;continue
        if line.startswith('|'):
            table_rows=[]
            while index<len(lines) and lines[index].strip().startswith('|'):
                cells=[value.strip() for value in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',value or '-') for value in cells):
                    table_rows.append([Paragraph(markup(value),cell) for value in cells])
                index+=1
            if table_rows:
                columns=len(table_rows[0])
                fractions=[1/columns]*columns
                if columns==3:fractions=[.17,.40,.43]
                if columns==2:fractions=[.23,.77]
                t=Table(table_rows,colWidths=[width*f for f in fractions],repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EDF2')),
                    ('GRID',(0,0),(-1,-1),.45,colors.HexColor('#D9D9D9')),
                    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),
                    ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),
                    ('BOTTOMPADDING',(0,0),(-1,-1),6)]))
                block=[Spacer(1,3),t,Spacer(1,10)]
                if story and isinstance(story[-1],Paragraph) and story[-1].style.name in ('h1','h2'):
                    block.insert(0,story.pop())
                story.append(KeepTogether(block)) if len(table_rows)<=8 else story.extend(block)
            continue
        match=re.match(r'!\[([^]]*)\]\(([^)]+)\)',line)
        if match:
            path=(args.input.parent/match.group(2)).resolve()
            if not path.is_file():raise FileNotFoundError(path)
            from PIL import Image as PILImage
            with PILImage.open(path) as picture:w,h=picture.size
            block=[Spacer(1,5),Image(str(path),width=width,height=width*h/w),Spacer(1,7)]
            following=index+1
            while following<len(lines) and not lines[following].strip():following+=1
            if following<len(lines) and lines[following].strip().startswith('图 '):
                block.append(Paragraph(markup(lines[following].strip()),caption))
                index=following
            story.append(KeepTogether(block))
        elif line.startswith('# '):
            heading=markup(line[2:]).replace('Voice Guard ','Voice Guard<br/>',1)
            story.append(Paragraph(heading,title))
        elif line.startswith('## '):story.append(Paragraph(markup(line[3:]),h1))
        elif line.startswith('### '):story.append(Paragraph(markup(line[4:]),h2))
        else:story.append(Paragraph(markup(line),caption if line.startswith('图 ') else body))
        index+=1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def decorate(canvas,doc):
        canvas.saveState();canvas.setFont('CourseCN',8)
        canvas.setFillColor(colors.HexColor('#4D5968'))
        canvas.drawString(22*mm,A4[1]-15*mm,'Voice Guard 语音隐私主动防御复现与质量改进')
        canvas.drawRightString(A4[0]-22*mm,13*mm,f'{doc.page}')
        canvas.restoreState()
    document=SimpleDocTemplate(str(args.output),pagesize=A4,leftMargin=22*mm,rightMargin=22*mm,
        topMargin=24*mm,bottomMargin=21*mm,title='Voice Guard 语音隐私主动防御复现与质量改进报告',author='Voice Guard 课程项目')
    document.build(story,onFirstPage=decorate,onLaterPages=decorate)
    import pypdfium2 as pdfium
    pdf=pdfium.PdfDocument(str(args.output))
    args.preview_dir.mkdir(parents=True,exist_ok=True)
    for i,page in enumerate(pdf):
        image=page.render(scale=1.5).to_pil()
        image.save(args.preview_dir/f'page-{i+1:02d}.png')
    print(f'PDF pages: {len(pdf)}; {args.output}')


if __name__=='__main__':main()
