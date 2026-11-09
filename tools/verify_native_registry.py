"""Check the corrected production builder against the frozen native target bank."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def main():
    evidence = Path('docs/evidence/2026-11-09-staged/final')
    protocol = json.loads((evidence / 'protocol.json').read_text())
    root = Path(protocol['output']).parent
    registry = root / 'production-registry-native.npz'
    output = root / 'production-registry-check.json'
    if output.exists(): raise ValueError('Preserve verification')
    with np.load(registry) as built, np.load(protocol['native_target_bank']) as expected:
        if list(built['names']) != list(expected['names']): raise ValueError('Enrollment names differ')
        np.testing.assert_allclose(built['embeddings'], expected['embeddings'], atol=1e-5, rtol=1e-5)
        max_error = float(np.max(np.abs(built['embeddings']-expected['embeddings'])))
        shape = list(built['embeddings'].shape)
    manifest = json.loads(Path(str(registry)+'.manifest.json').read_text())
    if manifest['preprocessing'] != 'native-get_spectrograms-trim-normalize-raw-Es-v1':
        raise ValueError('Wrong production preprocessing')
    if manifest['builder_sha256'] != file_sha256('scripts/build_registry.py'):
        raise ValueError('Builder changed')
    record = {'registry_sha256': file_sha256(registry),
        'native_bank_sha256': file_sha256(protocol['native_target_bank']),
        'builder_sha256': manifest['builder_sha256'], 'shape': shape,
        'max_abs_embedding_error': max_error, 'passed': True,
        'existing_registry_overwritten': False,
        'purpose': 'Production enrollment correction verified on the same 20x30 enrollment WAVs; not a new defense success or validation retuning.'}
    write_json(output, record); print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
