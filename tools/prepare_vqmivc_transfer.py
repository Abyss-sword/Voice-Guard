"""Freeze the prespecified observed-sentence VQMIVC transfer diagnostic."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,write_json


def main():
    evidence=Path('docs/evidence/2026-11-09-vqmivc')
    destination=evidence/'protocol.json'
    if destination.exists(): raise ValueError('Transfer protocol already frozen')
    read=lambda p:json.loads(Path(p).read_text())
    plan=read(evidence/'transfer-plan.json')
    parent_path=Path('docs/evidence/2026-11-09-discriminant/protocol.json')
    if file_sha256(parent_path)!=plan['parent_protocol_sha256']:raise ValueError('Parent changed')
    parent=read(parent_path)
    smoke_path=Path('outputs/vqmivc/smoke/smoke.json')
    smoke=read(smoke_path)
    if not smoke['repeat_byte_identical'] or not smoke['all_models_frozen']:
        raise ValueError('Real inference smoke must pass first')
    receipt_path=Path('checkpoints/vqmivc-official/download-receipt.json')
    if file_sha256(receipt_path)!=smoke['resource_receipt_sha256']:
        raise ValueError('Resources differ from smoke')
    assets={}
    for name in (parent_path,evidence/'transfer-plan.json',smoke_path,receipt_path,
                 Path('third_party/vqmivc/UPSTREAM.json')):
        assets[name.as_posix()]=file_sha256(name)
    for name,info in read(receipt_path)['files'].items():
        path=Path('checkpoints/vqmivc-official')/name
        if file_sha256(path)!=info['sha256']:raise ValueError('Resource changed')
        assets[path.as_posix()]=info['sha256']
    assets.update(read('third_party/vqmivc/UPSTREAM.json')['files_sha256'])
    for name in parent['assets']:
        if name.startswith('checkpoints/') or name.endswith('registry.npz'):
            assets[name]=parent['assets'][name]
    inputs={}
    for u in plan['utterances']:
        folder=Path(plan['parent_output'])
        original=folder/'controls'/u/'original.wav'
        inputs[u]={'original':original.as_posix(),'defended':{}}
        for profile in plan['profiles']:
            path=folder/'development'/profile/u/'defended.wav'
            optimization=path.with_name('optimization.json')
            meta=read(optimization)
            if meta['protocol_sha256']!=plan['parent_protocol_sha256'] or meta['defended_sha256']!=file_sha256(path):
                raise ValueError('Unmatched source defense')
            inputs[u]['defended'][profile]=path.as_posix()
            assets[optimization.as_posix()]=file_sha256(optimization)
            assets[path.as_posix()]=file_sha256(path)
        assets[original.as_posix()]=file_sha256(original)
    for u in plan['utterances']+plan['contents']:
        for ext in ('.txt','.wav'):
            path=Path('data/VCTK22')/u.split('_')[0]/(u+ext)
            assets[path.as_posix()]=file_sha256(path)
    code=dict(parent['code_sha256'])
    for name in ('vguard/vc/vqmivc.py','scripts/run_vqmivc_transfer.py',
                 'tools/prepare_vqmivc_transfer.py','tools/smoke_vqmivc.py'):
        code[name]=file_sha256(name)
    if code['vguard/vc/vqmivc.py']!=smoke['adapter_sha256']:raise ValueError('Adapter changed after smoke')
    for name,digest in {**assets,**code}.items():
        if file_sha256(name)!=digest:raise ValueError('Input changed: '+name)
    protocol={**plan,'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'output':'outputs/vqmivc/transfer-2026-11-09','seed':2026110908,
        'inputs':inputs,'assets':assets,'code_sha256':code,
        'ge2e_weights_sha256':parent['ge2e_weights_sha256'],'whisper_sha256':parent['whisper_sha256'],
        'selection':'No VQMIVC selection; all three parent profiles retained, no transfer-driven optimization.',
        'limitations':['Observed development sentences; no unseen-sentence transfer claim.',
            'Matching clean controls determine eligibility; low clean quality may limit interpretation.',
            'Released upstream model and vocoder; paper checkpoint equivalence not established.',
            'Fixed paired vocoder seed controls inference noise; only one noise realization per pair.']}
    write_json(destination,protocol)
    print(destination,file_sha256(destination))


if __name__=='__main__':main()
