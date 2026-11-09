"""Freeze a bounded extension with ten untouched validation sentences."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def main():
    source = Path('docs/evidence/2026-11-09-joint/protocol.json')
    destination = Path('docs/evidence/2026-11-09-joint-anchored/protocol.json')
    if destination.exists(): raise ValueError('Anchored protocol already frozen')
    protocol = json.loads(source.read_text())
    with np.load('outputs/registry.npz') as package: names = list(package['names'])[:10]
    validation = []
    for name in names:
        files = sorted((Path('data/VCTK22') / str(name)).glob('*.wav'))
        if len(files) < 73: raise ValueError('Need at least 73 original sentences')
        validation.append(files[72].stem)
    forbidden = set(protocol['validation']) | set(protocol['development'])
    for prior_path in ('docs/evidence/2026-11-09-quality/recovery-protocol.json',
                       'docs/evidence/2026-11-09-staged/final/protocol.json'):
        forbidden.update(json.loads(Path(prior_path).read_text())['validation'])
    forbidden.update(r['utterance'] for r in json.loads(Path('docs/evidence/2026-11-09/frozen-protocol.json').read_text())['new_utterances'])
    if set(validation) & forbidden:
        raise ValueError('Extension validation overlaps examined sentences')
    assets = {p: sha for p, sha in protocol['assets'].items() if not p.startswith('data/')}
    all_inputs = protocol['development'] + validation + protocol['optimization_contents'] + protocol['evaluation_contents'] + protocol['baseline_optimization_contents']
    for u in all_inputs:
        for suffix in ('.wav', '.txt'):
            path = Path('data/VCTK22') / u.split('_')[0] / (u + suffix)
            assets[path.as_posix()] = file_sha256(path)
    with np.load(protocol['native_target_bank']) as native, np.load('outputs/registry.npz') as external:
        if list(native['names']) != list(external['names']): raise ValueError('Surrogate enrollment order differs')
    protocol.update(frozen_utc=datetime.now(timezone.utc).isoformat(),
        output='outputs/joint_anchored/2026-11-09', validation=validation,
        validation_examined=False,
        validation_selection='Sorted original sentence at index 72 for each of the first ten enrolled speakers, distinct from the first joint study and previous optimization sentences.',
        type='Relative baseline output anchors with native-waveform / ECAPA surrogate ensemble',
        profiles=[{'name': 'anchored_joint_snr20', 'guarded': True}], assets=assets,
        extension_reason='First joint study development improved but its initial frozen validation pairs did not retain the benefit. A fixed ECAPA threshold allows already-strong baseline disruption to weaken. New soft targets preserve baseline output margins and reduce own-speaker similarity on ECAPA and native Es. This is a new protocol and new validation, not retuning the first validation scores.',
        first_joint_protocol_sha256=file_sha256(source),
        first_joint_selection_sha256=file_sha256(source.with_name('selection.json')),
        first_joint_validation_partly_examined=True,
        gradient_design='Same 100-forward / last-2-backward GL. ECAPA relative class-margin target min(-.12, baseline margin-.02); own similarity targets baseline-.08 on both ECAPA and native Es. Native Es uses actual synthesized-waveform features and detached trim boundaries. Original guard remains unchanged.',
        validation_gate='Run ten fresh validation sentences only if development full-clean successes >=5/6 and protected quality 3/3; otherwise stop extension at development and preserve negative results.',
        limitations=['Ten enrolled speakers, one new sentence each; not a claim of unseen-pretraining identities.',
            'Different fresh sentences from the first joint study; only compare paired baselines within each study.',
            'Soft surrogate targets and truncated phase/crop gradients do not guarantee held-out success.',
            'No human listening study or second VC model.'],
        limits={'development_hours': .35, 'validation_hours': .65, 'automatic_shutdown': False})
    observation_path = Path('outputs/joint_anchored/first-joint-partial-observation.json')
    if observation_path.exists(): raise ValueError('Preserve extension decision snapshot')
    partial = json.loads((Path(json.loads(source.read_text())['output']) / 'validation-results.json').read_text())
    write_json(observation_path, partial)
    protocol['decision_snapshot_sha256'] = file_sha256(observation_path)
    protocol['decision_snapshot_rows'] = len(partial['rows'])
    for name in ('vguard/anchored_joint_recovery.py', 'scripts/run_anchored_joint.py',
                 'tools/prepare_anchored_joint.py', 'tools/decide_anchored_validation.py'):
        protocol['code_sha256'][name] = file_sha256(name)
    for name, sha in {**assets, **protocol['code_sha256']}.items():
        if file_sha256(name) != sha: raise ValueError('Frozen dependency changed: ' + name)
    write_json(destination, protocol)
    print(destination, file_sha256(destination), 'validation', validation)


if __name__ == '__main__': main()
