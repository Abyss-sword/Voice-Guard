"""Run the declared post-hoc context diagnostic after the frozen campaign ends."""
from __future__ import annotations
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deploy.run_course_campaign import execute
from vguard.run_state import file_sha256,write_json


def main():
    os.chdir(ROOT)
    out=Path('outputs/course/2026-11-09')
    campaigns=sorted(out.glob('campaign-*.json'))
    if not campaigns or json.loads(campaigns[-1].read_text(encoding='utf-8'))['state']!='complete':
        raise ValueError('Finish the original frozen campaign before this post-hoc diagnostic')
    protocol_path=Path('docs/evidence/2026-11-09/posthoc-context-protocol.json')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    for path,digest in protocol['code_sha256'].items():
        if file_sha256(path)!=digest:raise ValueError(f'Declared diagnostic code differs: {path}')
    command=[sys.executable,'-u','scripts/quality_sweep.py','--exp-dir',protocol['input'],
             '--out-dir',protocol['output'],'--speakers',','.join(protocol['speakers']),
             '--scales','0,1','--wav-subtype','PCM_16','--restore-context']
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    state_path=out/f'context-diagnostic-{time.time_ns()}.json'
    record={'started_utc':datetime.now(timezone.utc).isoformat(),'state':'running','command':command,
            'protocol_sha256':file_sha256(protocol_path),'limit_seconds':900,
            'type':'Post-hoc context diagnostic; does not replace frozen validation'}
    write_json(state_path,record)
    started=time.monotonic()
    try:
        code=execute(command,900,out/'quality_new_refined_context_posthoc.log')
        if code:raise subprocess.CalledProcessError(code,command)
        summary=json.loads((Path(protocol['output'])/'summary.json').read_text(encoding='utf-8'))
        frozen=json.loads(Path('docs/evidence/2026-11-09/frozen-protocol.json').read_text(encoding='utf-8'))
        expected={(r['utterance'],s) for r in frozen['new_utterances'] for s in (0,1)}
        if len(summary['rows'])!=20 or {(r['utterance'],r['scale']) for r in summary['rows']}!=expected:
            raise ValueError('Context diagnostic is incomplete')
        record['state']='complete'
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired,KeyboardInterrupt,ValueError) as error:
        record.update(state='interrupted_or_failed',error=str(error))
        raise
    finally:
        record.update(elapsed_s=round(time.monotonic()-started,2),updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path,record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
