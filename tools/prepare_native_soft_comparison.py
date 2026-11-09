"""Freeze a development-only soft-mask extension before examining validation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def main():
    initial = Path('docs/evidence/2026-11-09-staged')
    protocol = json.loads((initial / 'protocol.json').read_text())
    selection = json.loads((initial / 'selection.json').read_text())
    if (Path(protocol['output']) / 'validation-state.json').exists():
        raise ValueError('Validation already examined')
    destination = initial / 'final/protocol.json'
    if destination.exists(): raise ValueError('Preserve frozen final protocol')
    protocol.update(frozen_utc=datetime.now(timezone.utc).isoformat(),
        output='outputs/staged_recovery/2026-11-09/final',
        type='Development extension: first staged winner versus native paper-style soft masking',
        profiles=[selection['selected']['profile'],
            {'name': 'native_soft_alpha8', 'objective': 'soft', 'final_snr_db': 0.}],
        soft_searches=8, soft_epsilon=.05,
        soft_interpretation='Raw MSE-.1 repulsion; 3000 strong, 8 explicit attempts of up to 1500 conceal steps with alpha starting 1. Internal native Es flag only; same strong restart; amplitude 2eps*tanh per public fragment, headroom clip, no SNR cap. Fallback strong solution retained and evaluated as failure if quality fails. Eight attempts replace the public double-increment while fragment.',
        prior_development_protocol_sha256=file_sha256(initial / 'protocol.json'),
        prior_selection_sha256=file_sha256(initial / 'selection.json'),
        prior_development_results_sha256=file_sha256(Path('outputs/staged_recovery/2026-11-09/development-results.json')),
        validation_examined=False,
        development_extension_reason='Native enrollment discrepancy verified; compare soft-mask search to projection on development only. Previous study remains immutable.')
    for name in ('vguard/native_soft_recovery.py', 'scripts/run_native_soft_comparison.py',
                 'tools/prepare_native_soft_comparison.py'):
        protocol['code_sha256'][name] = file_sha256(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, protocol)
    print(destination, file_sha256(destination))


if __name__ == '__main__': main()
