"""Post-hoc clean-only 2x2 diagnosis of resampling and vocoder seed sensitivity."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,utterance_seed,write_json
from vguard.vc.vqmivc import VQMIVC
from vguard.metrics.staged import StagedEvaluator


def main():
    out=Path('outputs/vqmivc/clean-control-diagnosis')
    if out.exists():raise ValueError('Preserve prior diagnosis')
    transfer_path=Path('docs/evidence/2026-11-09-vqmivc/protocol.json')
    p=json.loads(transfer_path.read_text())
    for name,sha in {**p['assets'],**p['code_sha256']}.items():
        if file_sha256(name)!=sha:raise ValueError('Frozen transfer input changed: '+name)
    source=Path('data/VCTK22/p226/p226_002.wav')
    references={'native_rate':Path('data/VCTK22/p226/p226_073.wav'),
        'adain_rate_pcm':Path(p['inputs']['p226_073']['original'])}
    seeds={'smoke_seed':2026110906,'transfer_seed':utterance_seed(p['seed'],'p226_073:p226_002')}
    out.mkdir(parents=True)
    protocol={'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':file_sha256(source),'references':{k:{'path':str(v),'sha256':file_sha256(v)} for k,v in references.items()},
        'seeds':seeds,'source_sha256_code':file_sha256(__file__),'parent_protocol_sha256':file_sha256(transfer_path),
        'scope':'Post-hoc known clean control only; fixed 2x2 combinations, no defense or method selection.'}
    write_json(out/'protocol.json',protocol)
    torch.set_num_threads(1);model=VQMIVC();evaluator=StagedEvaluator(out/'score-cache.json')
    rows=[]
    for label,reference in references.items():
        for seed_label,seed in seeds.items():
            wave=out/(label+'-'+seed_label+'.wav')
            metadata=model.convert(source,reference,wave,seed=seed)
            row={'reference_kind':label,'seed_kind':seed_label,'conversion':metadata,
                'score':evaluator.score(wave,source.with_suffix('.txt').read_text().strip())}
            rows.append(row);write_json(out/'results.json',{'rows':rows})
            print(json.dumps({k:row[k] for k in ('reference_kind','seed_kind')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
