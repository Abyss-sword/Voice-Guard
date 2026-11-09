"""Freeze a paired rate-path diagnostic, transporting only the fixed perturbation."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'outputs/vqmivc/vendor'))
import resampy
from vguard.run_state import file_sha256,write_json


def main():
    evidence=Path('docs/evidence/2026-11-09-vqmivc-transport')
    destination=evidence/'protocol.json'
    if destination.exists():raise ValueError('Transport protocol already frozen')
    parent_path=Path('docs/evidence/2026-11-09-vqmivc/protocol.json')
    p=json.loads(parent_path.read_text())
    for name,sha in {**p['assets'],**p['code_sha256']}.items():
        if file_sha256(name)!=sha:raise ValueError('Parent input changed: '+name)
    root=Path('outputs/vqmivc/transport-inputs')
    if root.exists():raise ValueError('Preserve existing transport inputs')
    assets=dict(p['assets']);assets[parent_path.as_posix()]=file_sha256(parent_path)
    for name in ('outputs/vqmivc/clean-control-diagnosis/protocol.json',
                 'outputs/vqmivc/clean-control-diagnosis/results.json','outputs/vqmivc/resampling-diagnosis.json'):
        assets[name]=file_sha256(name)
    inputs={};rows=[]
    for u in p['utterances']:
        native=Path('data/VCTK22')/u.split('_')[0]/(u+'.wav')
        full,rate=sf.read(native,dtype='float64')
        clean=resampy.resample(full,rate,16000,axis=0) if rate!=16000 else full
        original,adain_rate=sf.read(p['inputs'][u]['original'],dtype='float64')
        folder=root/u;folder.mkdir(parents=True)
        clean_path=folder/'original.wav';sf.write(clean_path,clean,16000,subtype='PCM_16')
        inputs[u]={'original':clean_path.as_posix(),'defended':{}}
        assets[clean_path.as_posix()]=file_sha256(clean_path)
        for profile,path in p['inputs'][u]['defended'].items():
            defended,sr=sf.read(path,dtype='float64')
            if sr!=adain_rate or defended.shape!=original.shape:raise ValueError('Parent layouts differ')
            delta=resampy.resample(defended-original,sr,16000,axis=0)
            if abs(len(delta)-len(clean))>1:raise ValueError('Unexpected transport sample count')
            delta=np.pad(delta,(0,max(0,len(clean)-len(delta))))[:len(clean)]
            transported=clean+delta
            if not np.isfinite(transported).all() or np.max(np.abs(transported))>=1:
                raise ValueError('Invalid/clipping transport')
            output=folder/(profile+'.wav');sf.write(output,transported,16000,subtype='PCM_16')
            assets[output.as_posix()]=file_sha256(output);inputs[u]['defended'][profile]=output.as_posix()
            rows.append({'utterance':u,'profile':profile,'clean_native_rate':rate,
                'parent_rate':sr,'transport_rate':16000,'samples':len(clean),
                'transport_snr_before_pcm_db':float(10*np.log10(np.mean(clean**2)/max(np.mean(delta**2),1e-20)))})
    code=dict(p['code_sha256'])
    for name in ('tools/prepare_vqmivc_transport.py','scripts/run_vqmivc_transport.py'):
        code[name]=file_sha256(name)
    p.update(frozen_utc=datetime.now(timezone.utc).isoformat(),
        output='outputs/vqmivc/transport-2026-11-09',inputs=inputs,assets=assets,code_sha256=code,
        direct_transfer_protocol_sha256=file_sha256(parent_path),
        transport_rows=rows,selection='No method selection; identical six observed references, two contents, three methods and paired seeds as direct transfer.',
        design='Post-hoc rate-path diagnostic, preregistered before transport scoring: y16=R(native original)+R(defended24-original24). Clean and protected references share the native original resampling; only the existing perturbation is transported. Entire references are exported PCM16. This changes the input channel and is not direct waveform transfer, white-box optimization or fresh validation.',
        limitations=p['limitations']+['Post-hoc channel diagnostic prompted by observed direct-transfer clean failures.',
            'Transport changes the reference preprocessing; cannot replace the direct-transfer result or prove resampling causality.',
            '16 kHz reference quality is scored separately; 24 kHz masking bounds do not imply a 16 kHz masking guarantee.'])
    write_json(destination,p);print(destination,file_sha256(destination))


if __name__=='__main__':main()
