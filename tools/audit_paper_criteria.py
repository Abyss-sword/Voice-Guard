"""Additional paper-structure audit of archived audio, without rewriting old scores."""
import argparse
import json
import sys
from pathlib import Path
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vguard.metrics.mos_original import OriginalMOSNet,WEIGHT_SHA256
from vguard.run_state import file_sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root',type=Path,default=Path('outputs/course/remote/outputs'))
    parser.add_argument('--out',type=Path,default=Path('outputs/paper_quality/archived-paper-criteria.json'))
    args=parser.parse_args();torch.set_num_threads(1);model=OriginalMOSNet()
    groups={};rows=[]
    def rescore(path):
        wave,sr=sf.read(path,dtype='float32');return model.score(wave,sr)
    for group in ['quality_new_baseline','quality_new_refined','quality_new_refined_context_posthoc']:
        root=args.archive_root/'course/2026-11-09'/group
        for directory in sorted(root.iterdir()):
            if not (directory/'scale_1/result.json').is_file():continue
            protected=json.loads((directory/'scale_1/result.json').read_text(encoding='utf-8'))
            control=json.loads((directory/'scale_0/result.json').read_text(encoding='utf-8'))
            target=protected['utterance'].split('_')[0]
            new_mos={}
            for scale in (0,1):
                for name in ('defended','converted'):
                    path=directory/f'scale_{scale}'/(name+'.wav')
                    new_mos[f'{scale}_{name}']={'score':rescore(path),'sha256':file_sha256(path)}
            screened=control['asv_converted']==target
            clean_quality=new_mos['0_converted']['score']>=3 and control['wer_converted']<=.10
            quality=protected['asv_defended']==target and new_mos['1_defended']['score']>=3 and protected['wer_defended']<=.10
            identity=protected['asv_converted']!=target
            disruption=identity or new_mos['1_converted']['score']<3 or protected['wer_converted']>.10
            row={'group':group,'utterance':protected['utterance'],'mos_original':new_mos,
                'saved_control_sha256':file_sha256(directory/'scale_0/result.json'),
                'saved_protected_sha256':file_sha256(directory/'scale_1/result.json'),
                'protected_wer':protected['wer_defended'],'clean_vc_wer':control['wer_converted'],
                'protected_vc_wer':protected['wer_converted'],'clean_identity_ok':screened,
                'clean_full_quality_ok':bool(screened and clean_quality),'protected_quality_ok':bool(quality),
                'strict_joint':bool(screened and quality and identity),
                'paper_structure_identity_screen':bool(screened and quality and disruption),
                'paper_structure_full_screen':bool(screened and clean_quality and quality and disruption)}
            rows.append(row)
        selected=[row for row in rows if row['group']==group]
        groups[group]={'total':len(selected),'identity_screened':sum(row['clean_identity_ok'] for row in selected),
            'full_control_screened':sum(row['clean_full_quality_ok'] for row in selected),
            **{key:sum(row[key] for row in selected) for key in ('protected_quality_ok','strict_joint','paper_structure_identity_screen','paper_structure_full_screen')}}
        print(group,groups[group],flush=True)
        write_json(args.out,{'groups':groups,'rows':rows,'weights_sha256':WEIGHT_SHA256,
            'evaluation_note':'Post-hoc additional audit. Original MOSNet rescores original WAVs; saved historical default-Whisper WER and ASV retained. Not the new greedy-decoding experiment. WER<=10% operationalizes the ambiguous paper ASR threshold.',
            'historical_observations_overwritten':False,'code_sha256':file_sha256(Path(__file__))})


if __name__=='__main__':main()
