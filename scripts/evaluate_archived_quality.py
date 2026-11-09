"""Score archived completed cloud trials locally, with immutable audio/model provenance.

This evaluates saved audio only. It does not regenerate missing GPU trials or ASV.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.run_state import completed_row,file_sha256,write_json,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='Recovered outputs/cloud directory')
    parser.add_argument('--out',type=Path,default=Path('outputs/course/local-quality'))
    parser.add_argument('--vctk',type=Path,default=Path('data/VCTK22'))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--whisper-cache',type=Path,default=Path('outputs/course/whisper-cache'))
    args=parser.parse_args()
    import soundfile as sf
    import torch
    import whisper
    from vguard.metrics.mos import load_mosnet,mos
    from vguard.metrics.wer import transcribe,wer,cer
    torch.set_num_threads(args.threads)
    recognizer=whisper.load_model('base',device=args.device,download_root=str(args.whisper_cache))
    model=load_mosnet(device=args.device)
    groups=[directory for directory in sorted(args.root.iterdir()) if directory.is_dir() and
            (directory.name.startswith('main_') or directory.name.startswith('eps_'))]
    jobs={directory:[sub for sub in sorted(directory.iterdir()) if sub.is_dir() and completed_row(sub)]
          for directory in groups}
    keys=('original_trimmed','defended','conv_from_clean','conv_from_defended')
    model_paths=[args.whisper_cache/'base.pt',*[p for p in Path('checkpoints/mosnet-hf').glob('*') if p.is_file()]]
    inputs=[path for directories in jobs.values() for sub in directories
            for path in [sub/'result.json',*[sub/f'{key}.wav' for key in keys]]]
    refs=[args.vctk/sub.name.split('_')[0]/f'{sub.name}.txt' for directories in jobs.values() for sub in directories]
    refs.append(args.vctk/'p226/p226_002.txt')
    manifest={'device':args.device,'threads':args.threads,'decode':'Existing transcribe: language=en, Whisper defaults; CPU fp32',
        'code_sha256':{str(p):file_sha256(p) for p in (Path(__file__),Path('vguard/metrics/mos.py'),Path('vguard/metrics/wer.py'))},
        'model_sha256':{str(p):file_sha256(p) for p in model_paths},
        'input_sha256':{str(p):file_sha256(p) for p in inputs},
        'transcript_sha256':{str(p):file_sha256(p) for p in refs},
        'versions':{name:importlib.metadata.version(name) for name in
                    ('torch','transformers','openai-whisper','librosa','numpy','soundfile')},
        'note':'Scores of archived completed cloud audio, evaluated locally; not a new attack/ASV run'}
    args.out.mkdir(parents=True,exist_ok=True)
    manifest_path=args.out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Scoring inputs changed; choose a new output directory')
    write_json(manifest_path,manifest)
    cache_dir=args.out/'audio-cache';cache_dir.mkdir(exist_ok=True)
    source_ref=(args.vctk/'p226/p226_002.txt').read_text().strip()
    started=time.monotonic()
    for group,directories in jobs.items():
        mos_rows=[];wer_rows=[];details=[]
        for sub in directories:
            target_ref=(args.vctk/sub.name.split('_')[0]/f'{sub.name}.txt').read_text().strip()
            mos_row={'utterance':sub.name};wer_row={'utterance':sub.name,'reference_mode':'transcript'}
            for key in keys:
                audio=sub/f'{key}.wav';digest=file_sha256(audio);cache=cache_dir/f'{digest}.json'
                if cache.exists():scored=json.loads(cache.read_text(encoding='utf-8'))
                else:
                    wave,sr=sf.read(audio,dtype='float32')
                    scored={'audio_sha256':digest,'mos':mos(model,wave,sr),'hypothesis':transcribe(recognizer,str(audio))}
                    write_json(cache,scored)
                reference=source_ref if key.startswith('conv_') else target_ref
                mos_row[f'mos_{key}']=round(scored['mos'],3)
                wer_row[f'wer_{key}']=round(wer(scored['hypothesis'],reference),4)
                wer_row[f'cer_{key}']=round(cer(scored['hypothesis'],reference),4)
                details.append({'utterance':sub.name,'artifact':key,'reference':reference,**scored})
            mos_rows.append(mos_row);wer_rows.append(wer_row)
            output=args.out/group.name
            write_csv(output/'mos.csv',mos_rows);write_csv(output/'wer.csv',wer_rows)
            write_json(output/'transcripts.json',details)
            print(group.name,sub.name,mos_row['mos_defended'],wer_row['wer_defended'],flush=True)
    write_json(args.out/'completion.json',{'complete':True,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'elapsed_s':round(time.monotonic()-started,2),'groups':{g.name:len(v) for g,v in jobs.items()},
        'unique_audio_files':len(list(cache_dir.glob('*.json'))),'manifest_sha256':file_sha256(manifest_path)})


if __name__=='__main__':main()
