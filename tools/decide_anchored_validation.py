"""Apply the predeclared development gate, without examining new validation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def main():
    evidence = Path('docs/evidence/2026-11-09-joint-anchored')
    path = evidence / 'extension-decision.json'
    if path.exists(): raise ValueError('Extension decision already frozen')
    protocol = json.loads((evidence / 'protocol.json').read_text())
    selection = json.loads((evidence / 'selection.json').read_text())
    if selection['protocol_sha256'] != file_sha256(evidence / 'protocol.json'):
        raise ValueError('Selection protocol differs')
    if (Path(protocol['output']) / 'validation-state.json').exists():
        raise ValueError('Extension validation already examined')
    chosen = selection['selected']
    allowed = chosen['full_joint'] >= 5 and chosen['protected_quality'] == 3
    write_json(path, {'decided_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': file_sha256(evidence / 'protocol.json'),
        'selection_sha256': file_sha256(evidence / 'selection.json'),
        'validation_allowed': allowed, 'development_full_successes': chosen['full_joint'],
        'development_protected_quality': chosen['protected_quality'],
        'gate': protocol['validation_gate'], 'new_validation_examined': False})
    print(json.loads(path.read_text()), flush=True)
    if not allowed: raise SystemExit(3)


if __name__ == '__main__': main()
