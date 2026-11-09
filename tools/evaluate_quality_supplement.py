"""Evaluate frozen validation WAVs with declared GE2E and small.en models."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,write_json
from vguard.metrics.wer import load_whisper,wer,cer


def main():
    from resemblyzer import VoiceEncoder,preprocess_wav
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['development','validation'],default='validation');args=parser.parse_args()
    directory=Path('docs/evidence/2026-11-09-quality')
    protocol=json.loads((directory/'recovery-protocol.json').read_text())
    supplement=json.loads((directory/'supplemental-protocol.json').read_text())
    if supplement['scorer_sha256']!=file_sha256(Path(__file__)):raise ValueError('Supplement scorer changed')
    resource=Path.home()/'.cache/whisper'/f"{supplement['asr']['name']}.pt"
    if file_sha256(resource)!=supplement['asr']['expected_sha256']:raise ValueError('ASR resource mismatch')
    if file_sha256(supplement['asv']['registry'])!=supplement['asv']['registry_sha256']:raise ValueError('GE2E registry mismatch')
    root=Path(protocol['output'])
    primary=json.loads((root/(args.split+'-results.json')).read_text())['rows']
    encoder=VoiceEncoder(device='cuda',verbose=False);recognizer=load_whisper(supplement['asr']['name'],'cuda')
    with np.load(supplement['asv']['registry'],allow_pickle=False) as registry:
        names=list(registry['names']);embeddings=np.asarray(registry['embeddings'])
    cached_path=root/(args.split+'-supplement-cache.json')
    cache=json.loads(cached_path.read_text()) if cached_path.exists() else {}
    def score(path,reference):
        digest=file_sha256(path);key=digest+':'+reference
        if key in cache:return cache[key]
        wave,sr=sf.read(path,dtype='float32')
        processed=preprocess_wav(wave,source_sr=sr)
        embedding=encoder.embed_utterance(processed)
        similarities=embeddings@embedding
        hypothesis=recognizer.transcribe(str(path),language='en',temperature=0,fp16=False)['text'].strip()
        record={'sha256':digest,'asv_ge2e':str(names[int(np.argmax(similarities))]),
            'similarities':{str(name):float(value) for name,value in zip(names,similarities)},
            'hypothesis':hypothesis,'reference':reference,'wer_small_en':wer(hypothesis,reference),'cer_small_en':cer(hypothesis,reference)}
        cache[key]=record;write_json(cached_path,cache)
        return record
    rows=[]
    for row in primary:
        utterance,content=row['utterance'],row['content'];speaker=utterance.split('_')[0]
        clean=root/'controls'/utterance
        protected=root/args.split/row['profile']/utterance/f"step_{row['checkpoint']}"
        scored={
            'original':score(clean/'original.wav',row['control']['original']['reference']),
            'clean_vc':score(clean/(content+'.wav'),row['control']['converted']['reference']),
            'defended':score(protected/'defended.wav',row['defended']['reference']),
            'protected_vc':score(protected/(content+'.wav'),row['converted']['reference'])}
        control_identity=scored['clean_vc']['asv_ge2e']==speaker
        control_quality=row['control']['converted']['mos_original']>=3 and scored['clean_vc']['wer_small_en']<=.10
        protected_quality=scored['defended']['asv_ge2e']==speaker and row['defended']['mos_original']>=3 and scored['defended']['wer_small_en']<=.10
        identity=scored['protected_vc']['asv_ge2e']!=speaker
        disruption=identity or row['converted']['mos_original']<3 or scored['protected_vc']['wer_small_en']>.10
        result={'utterance':utterance,'content':content,'profile':row['profile'],'checkpoint':row['checkpoint'],
            'scores':scored,'clean_identity_ok':control_identity,'clean_full_quality_ok':bool(control_identity and control_quality),
            'protected_quality_ok':bool(protected_quality),'strict_joint':bool(control_identity and protected_quality and identity),
            'paper_structure_identity_screen':bool(control_identity and protected_quality and disruption),
            'paper_structure_full_screen':bool(control_identity and control_quality and protected_quality and disruption),
            'mos_original':row['defended']['mos_original'],'supplemental_protocol_sha256':file_sha256(directory/'supplemental-protocol.json')}
        rows.append(result)
        write_json(root/(args.split+'-supplement.json'),{'rows':rows,'primary_results_sha256':file_sha256(root/(args.split+'-results.json'))})
        print(utterance,content,'GE2E strict',result['strict_joint'],'clean full',result['clean_full_quality_ok'],flush=True)


if __name__=='__main__':main()
