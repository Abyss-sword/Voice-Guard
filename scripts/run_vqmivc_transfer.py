"""Score a frozen black-box transfer diagnostic without optimization or selection."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vguard.vc.vqmivc import VQMIVC
from vguard.metrics.staged import StagedEvaluator,gates
from vguard.run_state import file_sha256,utterance_seed,write_json

PROTOCOL=Path('docs/evidence/2026-11-09-vqmivc/protocol.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-hours',type=float,default=.5)
    args=parser.parse_args()
    read=lambda p:json.loads(Path(p).read_text())
    p=read(PROTOCOL);digest=file_sha256(PROTOCOL)
    for name,sha in {**p['assets'],**p['code_sha256']}.items():
        if file_sha256(name)!=sha:raise ValueError('Frozen input changed: '+name)
    import resemblyzer
    if file_sha256(Path(resemblyzer.__file__).parent/'pretrained.pt')!=p['ge2e_weights_sha256']:
        raise ValueError('GE2E checkpoint changed')
    for name,sha in p['whisper_sha256'].items():
        if file_sha256(Path.home()/'.cache/whisper'/(name+'.pt'))!=sha:raise ValueError('Whisper changed')
    out=Path(p['output']);out.mkdir(parents=True,exist_ok=True)
    state_path=out/'state.json'
    if state_path.exists():
        old=read(state_path)
        if old['protocol_sha256']!=digest:raise ValueError('Existing state differs')
        if old['state']=='complete':print('Already complete; preserve outputs');return
    state={'state':'running','protocol_sha256':digest,'started_utc':datetime.now(timezone.utc).isoformat()}
    write_json(state_path,state);started=time.monotonic();rows=[]
    def bounded():
        if time.monotonic()-started>args.max_hours*3600:raise TimeoutError('Transfer time budget reached')
    try:
        torch.set_num_threads(1)
        model=VQMIVC();evaluator=StagedEvaluator(out/'score-cache.json')
        for u in p['utterances']:
            original_path=Path(p['inputs'][u]['original'])
            text=(Path('data/VCTK22')/u.split('_')[0]/(u+'.txt')).read_text().strip()
            original=evaluator.score(original_path,text)
            defended_scores={name:evaluator.score(path,text) for name,path in p['inputs'][u]['defended'].items()}
            for c in p['contents']:
                bounded();seed=utterance_seed(p['seed'],u+':'+c)
                source=Path('data/VCTK22')/c.split('_')[0]/(c+'.wav')
                content_text=source.with_suffix('.txt').read_text().strip()
                control=out/'controls'/u/(c+'.wav')
                control_meta=control.with_suffix('.conversion.json')
                if not control_meta.exists():
                    info=model.convert(source,original_path,control,seed=seed)
                    write_json(control_meta,{'conversion':info,'sha256':file_sha256(control),'protocol_sha256':digest})
                metadata=read(control_meta)
                if metadata['sha256']!=file_sha256(control) or metadata['protocol_sha256']!=digest:
                    raise ValueError('Clean conversion changed')
                clean=evaluator.score(control,content_text)
                for name in p['profiles']:
                    bounded();job=out/name/u;record=job/(c+'.json')
                    if record.exists():
                        row=read(record)
                        if row['protocol_sha256']!=digest:raise ValueError('Saved row differs')
                    else:
                        reference=Path(p['inputs'][u]['defended'][name]);wave=job/(c+'.wav')
                        info=model.convert(source,reference,wave,seed=seed)
                        converted=evaluator.score(wave,content_text)
                        row={'utterance':u,'content':c,'profile':name,'protocol_sha256':digest,
                            'original':original,'clean_vc':clean,'defended':defended_scores[name],
                            'protected_vc':converted,'conversion':info,'clean_conversion':metadata['conversion'],
                            'gates_ge2e':gates(u,original,clean,defended_scores[name],converted),
                            'gates_ecapa':gates(u,original,clean,defended_scores[name],converted,asv='ecapa',asr='base')}
                        write_json(record,row)
                    rows.append(row);write_json(out/'results.json',{'rows':rows})
                    print(json.dumps({'utterance':u,'content':c,'profile':name,'gates':row['gates_ge2e']},ensure_ascii=False),flush=True)
        state.update(state='complete',rows=len(rows))
    except BaseException as error:
        state.update(state='interrupted_or_failed',error=str(error),rows=len(rows));raise
    finally:
        state.update(elapsed_s=round(time.monotonic()-started,2),updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path,state)


if __name__=='__main__':main()
