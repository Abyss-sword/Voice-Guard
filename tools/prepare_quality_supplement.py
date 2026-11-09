"""Prepare GE2E enrollment and declare secondary evaluators before validation."""
import importlib.metadata,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,write_json
from vguard.audio import load_audio


def main():
    import resemblyzer,whisper
    from resemblyzer import VoiceEncoder,preprocess_wav
    directory=Path('docs/evidence/2026-11-09-quality')
    declaration=directory/'supplemental-protocol.json'
    protocol=json.loads((directory/'recovery-protocol.json').read_text())
    if (Path(protocol['output'])/'validation-state.json').exists():raise ValueError('Declare evaluators before validation')
    if declaration.exists():raise ValueError('Supplement already frozen')
    package=Path(resemblyzer.__file__).parent
    weights=package/'pretrained.pt'
    if file_sha256(weights)!='39373b86598fa3da9fcddee6142382efe09777e8d37dc9c0561f41f0070f134e':
        raise ValueError('GE2E pretrained model changed')
    torch.set_num_threads(1);encoder=VoiceEncoder(device='cuda',verbose=False)
    with np.load(protocol['eval_registry'],allow_pickle=False) as prior:names=list(prior['names'])
    enrollments={};embeddings=[];started=time.monotonic()
    for name in names:
        files=sorted((Path(protocol['vctk'])/name).glob('*.wav'))[:30]
        if len(files)!=30:raise ValueError('Need 30 enrollment utterances')
        values=[]
        for path in files:
            wave=load_audio(str(path),24000)
            wave=preprocess_wav(wave,source_sr=24000)
            values.append(encoder.embed_utterance(wave));enrollments[path.as_posix()]=file_sha256(path)
        embedding=np.mean(values,axis=0);embedding/=np.linalg.norm(embedding)
        embeddings.append(embedding);print('enrolled',name,flush=True)
    registry=Path('outputs/paper_quality/ge2e-registry.npz')
    np.savez(registry,names=np.asarray(names),embeddings=np.asarray(embeddings))
    receipt={'model':'Resemblyzer GE2E pretrained, not asserted to be exact Voice Guard authors checkpoint',
        'package_version':importlib.metadata.version('Resemblyzer'),'weights_sha256':file_sha256(weights),
        'package_source_sha256':{path.name:file_sha256(path) for path in sorted(package.glob('*.py'))},
        'enrollment_sha256':enrollments,'registry':registry.as_posix(),'registry_sha256':file_sha256(registry),
        'elapsed_s':round(time.monotonic()-started,2),'utterances_per_speaker':30,'speaker_count':len(names)}
    write_json(directory/'ge2e-resources.json',receipt)
    write_json(declaration,{'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'recovery_protocol_sha256':file_sha256(directory/'recovery-protocol.json'),
        'purpose':'Secondary independent evaluation of the same validation audio. Original base/ECAPA observations retained, no per-utterance evaluator choice.',
        'asr':{'name':'small.en','url':whisper._MODELS['small.en'],'expected_sha256':whisper._MODELS['small.en'].split('/')[-2],
               'package_version':importlib.metadata.version('openai-whisper'),'language':'en','temperature':0,'fp16':False},
        'asv':{'name':'Resemblyzer GE2E','registry_sha256':file_sha256(registry),'registry':registry.as_posix(),
               'resources_sha256':file_sha256(directory/'ge2e-resources.json')},
        'mos':'Original MOSNet scores from primary experiment; same fixed gate MOS>=3, WER<=.10',
        'scorer_sha256':file_sha256(ROOT/'tools/evaluate_quality_supplement.py'),
        'validation_examined':False})


if __name__=='__main__':main()
