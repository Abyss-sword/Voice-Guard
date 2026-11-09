"""Inspect the known clean reference's rate conversion and native model features."""
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
    destination=Path('outputs/vqmivc/resampling-diagnosis.json')
    if destination.exists():raise ValueError('Preserve previous diagnosis')
    torch.set_num_threads(1);model=VQMIVC()
    import resampy
    paths=[Path('data/VCTK22/p226/p226_073.wav'),
        Path('outputs/discriminant/2026-11-09-v2/controls/p226_073/original.wav')]
    rows=[];waves=[];features=[];embeddings=[]
    with torch.no_grad():
        for path in paths:
            wave,sr=sf.read(path);converted=resampy.resample(wave,sr,16000,axis=0) if sr!=16000 else wave
            mel,_=model.features(path);embedding=model.speaker_encoder(mel)
            rows.append({'path':str(path),'sha256':file_sha256(path),'sample_rate':sr,
                'samples':len(wave),'rms':float(np.sqrt(np.mean(wave**2))),
                'abs_peak':float(np.max(np.abs(wave))),'feature_shape':list(mel.shape),
                'feature_mean':float(mel.mean()),'feature_std':float(mel.std())})
            waves.append(converted);features.append(mel);embeddings.append(embedding)
        n=min(map(len,waves));delta=waves[0][:n]-waves[1][:n]
        result={'post_hoc_clean_only':True,'code_sha256':file_sha256(__file__),'rows':rows,
            'resampled_reference_snr_db':float(10*np.log10(np.mean(waves[0][:n]**2)/max(np.mean(delta**2),1e-20))),
            'resampled_max_abs_difference':float(np.max(np.abs(delta))),
            'speaker_embedding_cosine':float(torch.nn.functional.cosine_similarity(*embeddings).mean())}
        if features[0].shape==features[1].shape:
            result['feature_relative_l2']=float((features[0]-features[1]).norm()/features[0].norm())
    write_json(destination,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
