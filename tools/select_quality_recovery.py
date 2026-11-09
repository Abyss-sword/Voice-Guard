"""Choose one global recovery profile using only completed development results."""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vguard.run_state import file_sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('docs/evidence/2026-11-09-quality/recovery-protocol.json'))
    parser.add_argument('--summary-only',action='store_true');args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'));root=Path(protocol['output'])
    rows=json.loads((root/'development-results.json').read_text())['rows']
    grouped=defaultdict(list)
    for row in rows:grouped[(row['profile'],row['checkpoint'])].append(row)
    profiles={profile['name']:profile for profile in protocol['profiles']};summaries=[]
    for (name,step),values in grouped.items():
        summaries.append({'profile':name,'checkpoint':step,'count':len(values),
            'strict_joint':sum(row['strict_joint'] for row in values),
            'paper_structure_full_screen':sum(row['paper_structure_full_screen'] for row in values),
            'protected_quality_ok':sum(row['protected_quality_ok'] for row in values),
            'mean_wer':sum(row['defended']['wer'] for row in values)/len(values),
            'mean_mos_original':sum(row['defended']['mos_original'] for row in values)/len(values),
            'mean_actual_snr_db':sum(row['snr_db'] for row in values)/len(values),
            'snr_budget_db':profiles[name]['snr_db']})
    print(json.dumps(summaries,indent=2))
    if args.summary_only:return
    state=json.loads((root/'development-state.json').read_text())
    if state['state']!='complete':raise ValueError('Development must finish before selection')
    expected={(profile['name'],step,utterance,content) for profile in protocol['profiles']
              for step in protocol['checkpoints'] for utterance in protocol['development']
              for content in protocol['evaluation_contents']['development']}
    actual={(row['profile'],row['checkpoint'],row['utterance'],row['content']) for row in rows}
    if actual!=expected or len(rows)!=len(expected):raise ValueError('Development incomplete or duplicated')
    def key(row):return (row['strict_joint'],row['paper_structure_full_screen'],row['protected_quality_ok'],
        -row['mean_wer'],row['mean_mos_original'],row['snr_budget_db'],-row['checkpoint'],row['profile'])
    selected=max(summaries,key=key)
    path=args.protocol.parent/'recovery-selection.json'
    if path.exists():raise ValueError('Selection already frozen')
    write_json(path,{'frozen_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':file_sha256(args.protocol),
        'development_results_sha256':file_sha256(root/'development-results.json'),
        'profile':profiles[selected['profile']],'checkpoint':selected['checkpoint'],
        'selected_development_summary':selected,'all_development_summaries':summaries,
        'rule':protocol['selection_rule'],'validation_examined':False})
    print('Frozen selection:',selected['profile'],selected['checkpoint'])


if __name__=='__main__':main()
