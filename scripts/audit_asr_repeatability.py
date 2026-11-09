"""Post-hoc decoding control: the same five WAVs, two seeds, default vs greedy.

No audio/attack parameter is changed. This does not replace frozen quality scores.
"""
from __future__ import annotations
from datetime import datetime,timezone
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,write_csv,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-dir',type=Path,default=Path('outputs/cloud/main_vg'))
    parser.add_argument('--out-dir',type=Path,default=Path('outputs/course/2026-11-09/asr_repeatability_posthoc'))
    args=parser.parse_args()
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import numpy as np
    import torch
    from vguard.metrics.wer import load_whisper,wer
    params=json.loads((args.exp_dir/'manifest.json').read_text(encoding='utf-8'))['parameters']
    jobs=sorted(p for p in args.exp_dir.iterdir() if (p/'result.json').is_file())
    if len(jobs)!=5:raise ValueError('Expected the five original VG utterances')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    signature={'created_utc':datetime.now(timezone.utc).isoformat(),'type':'Post-hoc ASR decoding control',
        'design_reason':'The same saved baseline WAVs have different WER across the main quality table and channel audit.',
        'seeds':[2026,2027],'modes':{'default':{'language':'en'},'greedy':{'language':'en','temperature':0}},
        'script_sha256':file_sha256(Path(__file__)),
        'model_sha256':file_sha256(Path.home()/'.cache/whisper/base.pt'),
        'audio_sha256':{p.name:file_sha256(p/'defended.wav') for p in jobs},
        'reference_sha256':{p.name:file_sha256(Path(params['vctk'])/p.name.split('_')[0]/f'{p.name}.txt') for p in jobs},
        'torch_version':torch.__version__,'note':'Default Whisper includes temperature fallback; greedy is a separate diagnostic, not replacement scoring.'}
    if (args.out_dir/'summary.json').exists():raise ValueError('Use a fresh diagnostic directory')
    write_json(args.out_dir/'manifest.json',signature)
    model=load_whisper('base',device='cuda')
    started=time.monotonic();rows=[]
    for directory in jobs:
        reference=(Path(params['vctk'])/directory.name.split('_')[0]/f'{directory.name}.txt').read_text(encoding='utf-8').strip()
        for mode,options in signature['modes'].items():
            for seed in signature['seeds']:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                result=model.transcribe(str(directory/'defended.wav'),**options)
                row={'utterance':directory.name,'mode':mode,'seed':seed,'hypothesis':result['text'].strip(),
                     'reference':reference,'wer':wer(result['text'],reference),
                     'segment_temperatures':[s['temperature'] for s in result['segments']]}
                rows.append(row);write_csv(args.out_dir/'results.csv',rows)
                print(json.dumps(row),flush=True)
    stats={}
    for mode in signature['modes']:
        selected=[r for r in rows if r['mode']==mode]
        changed=[p.name for p in jobs if len({r['hypothesis'] for r in selected if r['utterance']==p.name})>1]
        stats[mode]={'changed_transcript_count':len(changed),'changed_utterances':changed,
                     'macro_wer_by_seed':{str(seed):sum(r['wer'] for r in selected if r['seed']==seed)/5 for seed in signature['seeds']}}
    write_json(args.out_dir/'summary.json',{'complete':len(rows)==20,'rows':rows,'statistics':stats,
               'elapsed_s':round(time.monotonic()-started,2),'protocol':signature})


if __name__=='__main__':main()
