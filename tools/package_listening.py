"""Prepare an unscored, label-masked listening kit from real archived audio.

This creates no human ratings and makes no claim of a formal MOS experiment.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'outputs/deploy/quality-evidence/outputs'
    comparison=json.loads((ROOT/'docs/evidence/2026-11-04/quality/comparison.json').read_text(encoding='utf-8'))
    base=ROOT/'outputs/course/listening-kit'
    participant=base/'participant'
    participant.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for row in comparison['rows']:
        utterance=row['utterance']
        original=source/'cloud/main_vg'/utterance
        evaluation='refine_dev_pcm16_eval' if row['split']=='development' else 'refine_validation_pcm16_eval'
        refined=source/'quality'/evaluation/utterance/'scale_1/defended.wav'
        jobs.extend((utterance,label,path) for label,path in (
            ('clean',original/'original_trimmed.wav'),('original_vg',original/'defended.wav'),('projected',refined)))
    random.Random(20261109).shuffle(jobs)
    answers=[]
    rows=[]
    for index,(utterance,label,path) in enumerate(jobs,1):
        name=f'clip_{index:02d}.wav'
        shutil.copy2(path,participant/name)
        answers.append({'clip':name,'utterance':utterance,'condition':label,
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        rows.append(f'<tr><td>{index:02d}</td><td><audio controls preload="metadata" src="{name}"></audio></td>'
                    '<td><input type="number" min="1" max="5" step="1" aria-label="自然度"></td>'
                    '<td><input type="text" aria-label="所听内容"></td><td><input type="text" aria-label="备注"></td></tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>语音试听记录</title>
<style>body{font:18px sans-serif;max-width:1100px;margin:35px auto;color:#172c40}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:10px}input{max-width:180px}audio{width:280px}button{padding:12px;margin:20px 0}</style>
<h1>语音试听记录</h1><p>这是尚未收集评分的试听材料，不是已有人工 MOS 结果。用耳机在安静环境试听，先调至舒适音量，之后保持播放器及系统音量不变。音频保持原始幅度，没有分别响度归一化。每条自然度可选填 1（很差）到 5（很好），也可记录所听内容和噪声、缺词等现象。允许跳过，不填写姓名。</p>
<p>文件只使用随机编号；请勿查看研究者答案。相同说话人的多句不能视为独立受试者。本材料不是正式标准化听评；未收集有效记录前不得报告人工分数。</p>
<table><thead><tr><th>编号</th><th>音频</th><th>自然度 1–5</th><th>所听内容</th><th>备注</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table><button id="save">下载本次记录 CSV</button>
<script>document.getElementById('save').onclick=()=>{const quote=x=>'"'+String(x).replaceAll('"','""')+'"';const data=[['clip','naturalness_optional','transcript_optional','note_optional'],...[...document.querySelectorAll('tbody tr')].map((r,i)=>['clip_'+String(i+1).padStart(2,'0')+'.wav',...[...r.querySelectorAll('input')].map(x=>x.value)])];const url=URL.createObjectURL(new Blob(['\\ufeff'+data.map(r=>r.map(quote).join(',')).join('\\r\\n')],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='listening-response.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};</script></html>'''
    (participant/'index.html').write_text(page,encoding='utf-8')
    (base/'researcher-answer-key.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    destination=ROOT/'outputs/course/resources/VoiceGuard-listening-participant.zip'
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(participant.iterdir()):archive.write(path,path.name)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:raise ValueError('Invalid listening archive')
        if len(archive.namelist())!=16:raise ValueError('Expected 15 audio files and one page')
    receipt={'archive':destination.name,'bytes':destination.stat().st_size,
             'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
             'clips':15,'human_ratings_collected':0,'amplitude_normalized':False,
             'source_answer_key_sha256':hashlib.sha256((base/'researcher-answer-key.json').read_bytes()).hexdigest(),
             'note':'Prepared material only; no human MOS or formal listening study claim'}
    (ROOT/'docs/evidence/2026-11-09/listening-kit.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
