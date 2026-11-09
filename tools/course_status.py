"""Compact status for the bounded cloud campaign; no completion from process exit alone."""
from pathlib import Path
import json
import sys

root=Path('outputs/course/2026-11-09')
states=sorted(root.glob('campaign-*.json'))
if not states:raise SystemExit('No campaign state')
state=json.loads(states[-1].read_text(encoding='utf-8'))
print(json.dumps({'state':state['state'],'started_utc':state['started_utc'],
    'stages_completed':sum(stage.get('returncode')==0 for stage in state['stages']),
    'latest_stage':state['stages'][-1], 'error':state.get('error')},ensure_ascii=False))
for name in ('new_utterances_vg','new_utterances_projected','quality_new_baseline','quality_new_refined',
             'channels_baseline','contents_baseline','channels_refined_dev','contents_refined_dev',
             'channels_refined_validation','contents_refined_validation'):
    directory=root/name
    if directory.exists():
        print(name,'results',len(list(directory.rglob('result.json'))),'optimizations',len(list(directory.rglob('optimization.json'))),
              'summary',(directory/'summary.json').exists())
if '--summary' in sys.argv:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from vguard.evaluation import summarize_pairs
    for name in ('quality_new_baseline','quality_new_refined'):
        path=root/name/'summary.json'
        if path.exists():
            rows=json.loads(path.read_text(encoding='utf-8'))['rows']
            if len(rows)==20:
                print(json.dumps({'experiment':name,**summarize_pairs([r for r in rows if r['scale']==1])}))
