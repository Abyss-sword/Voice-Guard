"""Verify pinned VQMIVC loading, paired noise and a known clean conversion."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.vc.vqmivc import VQMIVC
from vguard.run_state import file_sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score',action='store_true')
    args=parser.parse_args()
    out=Path('outputs/vqmivc/smoke')
    if out.exists(): raise ValueError('Preserve prior smoke; use a fresh process directory')
    receipt=json.loads(Path('checkpoints/vqmivc-official/download-receipt.json').read_text())
    for name,value in receipt['files'].items():
        if file_sha256(Path('checkpoints/vqmivc-official')/name)!=value['sha256']:
            raise ValueError('Downloaded asset changed')
    upstream=json.loads(Path('third_party/vqmivc/UPSTREAM.json').read_text())
    for name,digest in upstream['files_sha256'].items():
        if file_sha256(name)!=digest: raise ValueError('Pinned source changed: '+name)
    torch.set_num_threads(1)
    model=VQMIVC()
    source=Path('data/VCTK22/p226/p226_002.wav')
    reference=Path('data/VCTK22/p226/p226_073.wav')
    out.mkdir(parents=True)
    a,b=out/'clean-a.wav',out/'clean-b.wav'
    first=model.convert(source,reference,a)
    second=model.convert(source,reference,b)
    if file_sha256(a)!=file_sha256(b): raise ValueError('Paired vocoder noise is not repeatable')
    wave,sr=sf.read(a,dtype='float32')
    if sr!=16000 or not np.isfinite(wave).all() or np.std(wave)<1e-5:
        raise ValueError('Invalid clean conversion')
    models=(model.encoder,model.pitch_encoder,model.speaker_encoder,model.decoder,model.vocoder)
    record={'source_commit':upstream['commit'],'resource_receipt_sha256':file_sha256('checkpoints/vqmivc-official/download-receipt.json'),
        'adapter_sha256':file_sha256('vguard/vc/vqmivc.py'),'source_sha256':file_sha256(source),
        'reference_sha256':file_sha256(reference),'clean_conversion':first,'repeat':second,
        'wav_sha256':file_sha256(a),'repeat_byte_identical':True,
        'all_models_frozen':all(not p.requires_grad for m in models for p in m.parameters()),
        'scope':'Known development reference; plumbing smoke, not defense evidence.'}
    if args.score:
        from vguard.metrics.staged import StagedEvaluator
        evaluator=StagedEvaluator(out/'score-cache.json')
        record['clean_score']=evaluator.score(a,source.with_suffix('.txt').read_text().strip())
    write_json(out/'smoke.json',record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
