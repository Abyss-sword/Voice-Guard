"""Archive discriminant-development evidence, including corrected and interrupted attempts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tarfile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    evidence = Path('docs/evidence/2026-11-09-discriminant')
    protocol = read(evidence / 'protocol.json'); run = Path(protocol['output'])
    decision, analysis = read(evidence / 'extension-decision.json'), read(evidence / 'analysis.json')
    if not analysis['counts_verified']:
        raise ValueError('Audited main and complete main evidence required')
    archive_path = run.parent / 'discriminant-evidence.tar.gz'
    receipt_path = Path(str(archive_path) + '.receipt.json')
    if archive_path.exists() or receipt_path.exists(): raise ValueError('Preserve immutable archive')
    measured = {}
    splits = ('development', 'validation') if decision['validation_allowed'] else ('development',)
    if not decision['validation_allowed'] and (run / 'validation-state.json').exists():
        raise ValueError('Failed gate must not score validation')
    for split in splits:
        if read(run / (split + '-state.json'))['state'] != 'complete': raise ValueError('Incomplete split')
        for row in read(run / (split + '-results.json'))['rows']:
            job = run / split / row['profile'] / row['utterance']
            control = run / 'controls' / row['utterance']
            for key, path in [('original', control / 'original.wav'),
                    ('clean_vc', control / (row['content'] + '.wav')),
                    ('defended', job / 'defended.wav'), ('protected_vc', job / (row['content'] + '.wav'))]:
                if file_sha256(path) != row[key]['sha256']: raise ValueError('Scored main WAV changed')
                measured[path.as_posix()] = row[key]['sha256']
    expected_code = protocol['code_sha256']
    for name, digest in {**expected_code, **protocol['assets']}.items():
        if file_sha256(name) != digest: raise ValueError('Frozen dependency changed: ' + name)
    import torchaudio
    write_json(evidence / 'runtime-environment.json', {'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'torch': torch.__version__, 'torchaudio': torchaudio.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'note': 'Post-run environment snapshot; does not retrospectively assert backend flags.',
        'ctc_checkpoint': {p: sha for p, sha in protocol['assets'].items() if p.startswith('checkpoints/ctc/')},
        'automatic_shutdown': False})
    entries = {}
    for directory in (run, evidence, run.parent / '2026-11-09'):
        for path in directory.rglob('*'):
            if path.is_file() and not path.is_symlink(): entries[path.as_posix()] = path
    for path in run.parent.iterdir():
        if path.is_file() and (path.suffix in ('.log', '.json', '.py') or path.name == 'initial-source.tar') and '.receipt.' not in path.name:
            entries[path.as_posix()] = path
    for name in protocol['assets']:
        if not name.startswith('checkpoints/'): entries[name] = Path(name)
    for name, digest in protocol['enrollment_sha256'].items():
        if file_sha256(name) != digest: raise ValueError('Enrollment input changed')
        entries[name] = Path(name)
    for directory in ('vguard', 'scripts', 'tools', 'deploy', 'tests', 'third_party/ada_in_vc'):
        for path in Path(directory).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.sh', '.txt', '.yaml'):
                entries['executed-source/' + path.as_posix()] = path
    with tarfile.open(archive_path, 'w:gz') as package:
        for name, path in sorted(entries.items()): package.add(path, arcname=name, recursive=False)
    receipt = {'archive': archive_path.name, 'bytes': archive_path.stat().st_size,
        'sha256': file_sha256(archive_path), 'measured_wavs_verified_against_scores': len(measured),
        'wav_files': sum(name.endswith('.wav') for name in entries),
        'development_only': not decision['validation_allowed'],
        'validation_allowed': decision['validation_allowed'],
        'files_sha256': {name: file_sha256(path) for name, path in sorted(entries.items())},
        'note': 'Model weights excluded. Existing model/evaluator packages plus the exact CTC checkpoint are required. Includes new inputs, prototypes, WAVs, transcript scores, main protocol, decisions and executed code.'}
    write_json(receipt_path, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'files_sha256'}, indent=2))


if __name__ == '__main__': main()
