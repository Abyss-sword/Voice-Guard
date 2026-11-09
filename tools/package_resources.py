"""Package downloaded model assets with byte hashes and restore instructions.

No private keys, credentials, datasets or synthetic sharing links are included.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/course/resources/VoiceGuard-model-resources.zip'))
    parser.add_argument('--whisper',type=Path)
    args=parser.parse_args()
    files=[p for name in ('adain-vc-official','ecapa-local','mosnet-hf')
           for p in (ROOT/'checkpoints'/name).glob('*') if p.is_file()]
    required=[ROOT/'checkpoints'/name for name in ('adain-vc-official/vctk_model.ckpt',
              'ecapa-local/embedding_model.ckpt','mosnet-hf/pytorch_model.bin')]
    if not all(p.is_file() for p in required):raise FileNotFoundError('Required model assets missing')
    entries=[(p,p.relative_to(ROOT).as_posix()) for p in sorted(files)]
    if args.whisper:
        if not args.whisper.is_file():raise FileNotFoundError(args.whisper)
        entries.append((args.whisper,'whisper/base.pt'))
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'kind':'model resources, not a complete deployment',
        'files':[{'path':name,'bytes':p.stat().st_size,'sha256':file_sha256(p)} for p,name in entries],
        'included_whisper':bool(args.whisper),
        'not_included':['VCTK dataset','registries: generate from the recorded enrollment protocol','Python/CUDA environment'],
        'sources':{'AdaIN-VC':'https://github.com/jjery2243542/adaptive_voice_conversion',
                   'ECAPA':'https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286',
                   'MOSNet':'https://huggingface.co/1NEYRON1/mosnet/tree/4162a3e63ac725ecf7dce8dee3aa71e7afd9c65d',
                   'Whisper':'https://github.com/openai/whisper'},
        'sharing_url':None,'license_note':'Upstream model/data terms apply; no new license is granted by this archive.'}
    instructions='''# Voice Guard 模型资源

解压 checkpoints/ 到源码仓库根目录；按 RESOURCE-MANIFEST.json 逐个校验 SHA256。
这些是公开上游预训练模型，不是本项目新训练权重。来源 URL 与原下载记录随包保留。
参考各上游仓库的模型与数据使用条款；本压缩包不更改它们的授权。

若有 whisper/base.pt，复制至运行用户的 ~/.cache/whisper/base.pt（或指定的缓存目录）。
未包含 Whisper 时，安装 deploy/evaluation-requirements.txt 后由 whisper.load_model('base') 下载。
按根 README 获取 VCTK 子集，运行 scripts/reproduce.py registry 重建两份注册库。
本包不含 Python/CUDA 环境、完整数据集或注册库，不能单独启动服务。

把本 ZIP 上传课程要求的云盘，在根 README 填真实分享 URL、提取码（如有）及本包 SHA256。
本地生成压缩包不等于已经上传云盘。
'''
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p,name in entries:archive.write(p,name)
        archive.writestr('RESOURCE-MANIFEST.json',json.dumps(metadata,ensure_ascii=False,indent=2))
        archive.writestr('RESTORE.md',instructions)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:raise ValueError('Archive CRC check failed')
    write_json(args.output.with_suffix('.receipt.json'),{**metadata,'archive_bytes':args.output.stat().st_size,
        'archive_sha256':file_sha256(args.output),'archive_crc_verified':True})
    print(args.output,args.output.stat().st_size,file_sha256(args.output))


if __name__=='__main__':main()
