"""Experimental refinement of existing VG outputs; quality_sweep evaluates separately."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.attack import refine_masked
from vguard.audio import load_audio
from vguard.masking import generate_th
from vguard.run_state import file_sha256, utterance_seed, write_json
from vguard.speaker import SpeakerRegistry
from vguard.vc.adainvc import load_adainvc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--speakers', default='p225,p226')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=.0003)
    parser.add_argument('--margin-db', type=float, default=3)
    args=parser.parse_args()
    if (args.steps < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not np.isfinite(args.margin_db) or args.margin_db < 0):
        parser.error('steps/lr must be positive; margin must be finite and nonnegative')
    base=json.loads((args.exp_dir/'manifest.json').read_text())['parameters']
    for path, digest in base['input_sha256'].items():
        if file_sha256(path) != digest: raise ValueError(f'Asset changed: {path}')
    speakers=set(args.speakers.split(','))
    jobs=[directory for directory in sorted(args.exp_dir.iterdir())
          if directory.is_dir() and directory.name.split('_')[0] in speakers
          and (directory/'result.json').is_file()]
    if {directory.name.split('_')[0] for directory in jobs} != speakers:
        parser.error('all requested speakers must have a completed source result')
    signature={'parameters':{**base,'method':'vg-projected','steps':args.steps,
                             'learning_rate':args.learning_rate,'margin_db':args.margin_db},
               'speakers':args.speakers,'source':str(args.exp_dir),
               'source_manifest_sha256':file_sha256(args.exp_dir/'manifest.json'),
               'source_audio_sha256':{str(path):file_sha256(path) for directory in jobs for path in
                                     (directory/'original_trimmed.wav',directory/'defended.wav')},
               'target_reference_sha256':{str(path):file_sha256(path) for folder in sorted(Path(base['vctk']).iterdir())
                                         if folder.is_dir() for path in sorted(folder.glob('*.wav'))[:1]},
               'code_sha256':file_sha256(Path(__file__)),
               'attack_sha256':file_sha256(ROOT/'vguard/attack.py'),
               'masking_sha256':file_sha256(ROOT/'vguard/masking.py'),
               'dependency_sha256':{name:file_sha256(ROOT/name) for name in
                                    ('vguard/features.py','vguard/audio.py','vguard/speaker.py',
                                     'vguard/vc/adainvc.py','third_party/ada_in_vc/model.py')}}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest=args.out_dir/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Refinement configuration changed; choose another directory')
    write_json(manifest,signature)
    model,config,attr=load_adainvc(base['model_dir'],device='cuda')
    pp=config['preprocess']
    kwargs={key:pp[key] for key in ('sample_rate','preemph','n_fft','hop_length','win_length','n_mels','ref_db','max_db','top_db')}
    with np.load(base['attack_registry'],allow_pickle=False) as values:
        registry=SpeakerRegistry(values['embeddings'],list(values['names'])).to('cuda')
    for directory in jobs:
        speaker=directory.name.split('_')[0]
        output=args.out_dir/directory.name
        if (output/'optimization.json').exists() and (output/'defended.wav').is_file():
            saved=json.loads((output/'optimization.json').read_text())
            if (saved['defended_sha256'] != file_sha256(output/'defended.wav') or
                    saved['original_sha256'] != file_sha256(output/'original_trimmed.wav')):
                raise ValueError(f'Completed audio changed: {output}')
            continue
        output.mkdir(parents=True,exist_ok=True)
        started=time.monotonic()
        original,sr=sf.read(directory/'original_trimmed.wav',dtype='float32')
        defended,sr2=sf.read(directory/'defended.wav',dtype='float32')
        if sr != sr2 or sr != pp['sample_rate'] or original.shape != defended.shape:
            raise ValueError('Source audio alignment mismatch')
        embedding=registry.embeddings[registry.names.index(speaker)]
        least_likely,_=registry.least_likely(speaker,embedding)
        target=sorted((Path(base['vctk'])/least_likely).glob('*.wav'))[0]
        adv=load_audio(str(target),sr)
        theta,psd_max=generate_th(original,sr,pp['n_fft'],pp['hop_length'],pp['win_length'])
        torch.manual_seed(utterance_seed(2026,directory.name))
        wave,trace=refine_masked(model,original,defended-original,adv,theta,psd_max,attr,kwargs,
                                steps=args.steps,learning_rate=args.learning_rate,margin_db=args.margin_db,eps=base['eps'])
        if not np.isfinite(wave).all() or np.max(np.abs(wave)) > 1:
            raise ValueError('Invalid or clipping refined waveform')
        sf.write(output/'defended.wav',wave,sr,subtype='FLOAT')
        shutil.copy2(directory/'original_trimmed.wav',output/'original_trimmed.wav')
        write_json(output/'optimization.json',{'algorithm':'experimental projected refinement',
            'utterance':directory.name,'trace':trace,'elapsed_s':round(time.monotonic()-started,2),
            'original_sha256':file_sha256(directory/'original_trimmed.wav'),
            'starting_defended_sha256':file_sha256(directory/'defended.wav'),
            'adv_reference_sha256':file_sha256(target), 'defended_sha256':file_sha256(output/'defended.wav'),
            'evaluated':False})


if __name__=='__main__':
    main()
