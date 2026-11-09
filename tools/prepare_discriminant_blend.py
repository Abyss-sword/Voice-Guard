"""Freeze a global blend study only after its parent's failed development gate."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,write_json


def main():
    parent=Path('docs/evidence/2026-11-09-discriminant')
    destination=Path('docs/evidence/2026-11-09-discriminant-blend/protocol.json')
    if destination.exists(): raise ValueError('Blend protocol already frozen')
    read=lambda f:json.loads((parent/f).read_text())
    p=read('protocol.json'); d=read('extension-decision.json')
    if d['protocol_sha256']!=file_sha256(parent/'protocol.json') or d['selection_sha256']!=file_sha256(parent/'selection.json'):
        raise ValueError('Parent decision does not match its frozen selection')
    output=Path(p['output'])
    if d['validation_allowed'] or (output/'validation-state.json').exists():
        raise ValueError('This continuation requires a failed gate and untouched reserved validation')
    state=json.loads((output/'development-state.json').read_text())
    if state['state']!='complete': raise ValueError('Parent development incomplete')
    assets=dict(p['assets'])
    for u in p['development']:
        for name in ('original.wav',*(c+'.wav' for c in p['evaluation_contents'])):
            path=output/'controls'/u/name
            assets[path.as_posix()]=file_sha256(path)
        for profile in (p['baseline']['name'],'lda_margin_snr20'):
            for name in ('defended.wav','optimization.json'):
                path=output/'development'/profile/u/name
                assets[path.as_posix()]=file_sha256(path)
    for name in ('development-results.json','score-cache.json'):
        path=output/name;assets[path.as_posix()]=file_sha256(path)
    code=dict(p['code_sha256'])
    for name in ('scripts/run_discriminant_blend.py','tools/prepare_discriminant_blend.py',
            'tools/decide_blend_validation.py'):
        code[name]=file_sha256(name)
    for name,digest in {**assets,**code}.items():
        if file_sha256(name)!=digest:raise ValueError('Frozen source changed: '+name)
    p.update(frozen_utc=datetime.now(timezone.utc).isoformat(),seed=2026110907,
        output='outputs/discriminant_blend/2026-11-09',
        type='Global interpolation of fixed native-discriminant and baseline waveforms',
        parent_output=output.as_posix(),parent_protocol_sha256=file_sha256(parent/'protocol.json'),
        parent_decision_sha256=file_sha256(parent/'extension-decision.json'),
        profiles=[{'name':'lda_blend25','snr_db':20.,'mask_relax_db':0.,'blend':.25},
            {'name':'lda_blend50','snr_db':20.,'mask_relax_db':0.,'blend':.5},
            {'name':'lda_blend75','snr_db':20.,'mask_relax_db':0.,'blend':.75}],
        blend_design='Use one global coefficient selected on all twelve observed development sentences. No per-sentence scorer-guided choice. Development reuses exact parent baseline and LDA20 WAVs; validation reruns baseline and the fixed 800-step LDA20 proposal, then blends using the one frozen coefficient.',
        validation_examined=False,assets=assets,code_sha256=code)
    p.pop('implementation_correction',None)
    p['relaxed_budget_note']='All blend profiles use the SNR20 and original masking budget. The parent relaxed-budget profile is not used.'
    write_json(destination,p)
    print(destination,file_sha256(destination))


if __name__=='__main__':main()
