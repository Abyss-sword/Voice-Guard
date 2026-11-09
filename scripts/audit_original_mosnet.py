"""Score existing immutable WAVs with the original MOSNet; preserve old observations."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vguard.metrics.mos_original import OriginalMOSNet,WEIGHT_SHA256
from vguard.run_state import file_sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root',type=Path,default=Path('outputs/course/remote/outputs'))
    parser.add_argument('--out',type=Path,default=Path('outputs/paper_quality/original-mosnet-audit.json'))
    parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(1);model=OriginalMOSNet(device=args.device)
    specs={'main_clean':'cloud/main_vg/*/original_trimmed.wav','main_vg':'cloud/main_vg/*/defended.wav',
       'first_refined_dev':'quality/refine_dev_pcm16_eval/*/scale_1/defended.wav',
       'first_refined_validation':'quality/refine_validation_pcm16_eval/*/scale_1/defended.wav'}
    for directory in ('quality_new_baseline','quality_new_refined','quality_new_refined_context_posthoc'):
        for scale in (0,1):specs[f'{directory}_scale_{scale}']=f'course/2026-11-09/{directory}/*/scale_{scale}/defended.wav'
    rows=[];started=time.monotonic()
    for group,pattern in specs.items():
        for path in sorted(args.archive_root.glob(pattern)):
            wave,sr=sf.read(path,dtype='float32')
            rows.append({'group':group,'path':str(path.relative_to(args.archive_root)).replace('\\','/'),
                         'sha256':file_sha256(path),'mos_original':model.score(wave,sr)})
        group_rows=[row for row in rows if row['group']==group]
        print(group,len(group_rows),float(np.mean([row['mos_original'] for row in group_rows])) if group_rows else None,flush=True)
        write_json(args.out,{'evaluator':'Original MOSNet CNN-BLSTM, Hamming symmetric, reflect padding, Keras hard sigmoid',
            'weights_sha256':WEIGHT_SHA256,'code_sha256':file_sha256(ROOT/'vguard/metrics/mos_original.py'),
            'elapsed_s':round(time.monotonic()-started,2),'rows':rows,'historical_scores_overwritten':False,
            'groups':{key:{'count':len(values),'mean_mos':float(np.mean(values))} for key in specs
                      if (values:=[row['mos_original'] for row in rows if row['group']==key])}})


if __name__=='__main__':main()
