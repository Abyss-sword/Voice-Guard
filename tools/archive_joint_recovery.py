"""Bundle and verify joint-defense evidence, inputs and the executed source bytes."""
import argparse
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(archive_path, receipt_path, destination, evidence):
    receipt = read(receipt_path)
    if archive_path.stat().st_size != receipt['bytes'] or file_sha256(archive_path) != receipt['sha256']:
        raise ValueError('Archive size/digest mismatch')
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()): raise ValueError('Use an empty extraction directory')
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or set(names) != set(receipt['files_sha256']):
            raise ValueError('Unexpected, missing or duplicate archive members')
        for member in members:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or ':' in member.name:
                raise ValueError('Unsafe archive entry')
            if not (destination / member.name).resolve().is_relative_to(destination):
                raise ValueError('Archive entry escapes extraction root')
        archive.extractall(destination, filter='data')
    for name, digest in receipt['files_sha256'].items():
        if file_sha256(destination / name) != digest: raise ValueError('Extracted digest mismatch: ' + name)
    output = {k: v for k, v in receipt.items() if k != 'files_sha256'}
    output.update(all_files_verified=len(receipt['files_sha256']), local_archive_verified=True)
    write_json(evidence / 'archive-verification.json', output)
    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('outputs/joint_recovery/joint-evidence.tar.gz'))
    parser.add_argument('--evidence', type=Path, default=Path('docs/evidence/2026-11-09-joint'))
    parser.add_argument('--verify-to', type=Path)
    parser.add_argument('--development-only', action='store_true',
        help='Only for an extension explicitly stopped by its preregistered development gate.')
    args = parser.parse_args()
    receipt_path = Path(str(args.archive) + '.receipt.json')
    if args.verify_to:
        verify(args.archive, receipt_path, args.verify_to, args.evidence); return
    if args.archive.exists() or receipt_path.exists(): raise ValueError('Preserve immutable archive')
    protocol = read(args.evidence / 'protocol.json')
    run = Path(protocol['output'])
    if args.development_only:
        if read(args.evidence / 'extension-decision.json')['validation_allowed']:
            raise ValueError('Development-only archive requires a failed extension gate')
        if (run / 'validation-state.json').exists(): raise ValueError('Validation exists; audit it normally')
        splits = ('development',)
    else:
        if not read(args.evidence / 'analysis.json')['counts_verified']:
            raise ValueError('Audited validation required')
        splits = ('development', 'validation')
    measured = {}
    for split in splits:
        if read(run / (split + '-state.json'))['state'] != 'complete': raise ValueError('Incomplete split')
        for row in read(run / (split + '-results.json'))['rows']:
            control = run / 'controls' / row['utterance']
            job = run / split / row['profile'] / row['utterance']
            for key, path in [('original', control / 'original.wav'),
                ('clean_vc', control / (row['content'] + '.wav')),
                ('defended', job / 'defended.wav'), ('protected_vc', job / (row['content'] + '.wav'))]:
                digest = file_sha256(path)
                if digest != row[key]['sha256']: raise ValueError('Scored waveform changed: ' + str(path))
                if path.as_posix() in measured and measured[path.as_posix()] != digest:
                    raise ValueError('Inconsistent duplicate waveform')
                measured[path.as_posix()] = digest
    diagnostic_path = args.evidence / 'clean-chain-diagnostic.json'
    diagnostic_rows = read(diagnostic_path)['rows'] if diagnostic_path.exists() else []
    for row in diagnostic_rows:
        path = Path(row['path'])
        digest = file_sha256(path)
        if digest != row['score']['sha256']: raise ValueError('Diagnostic waveform changed')
        measured[path.as_posix()] = digest
    entries = {}
    for folder in (run, args.evidence):
        for path in folder.rglob('*'):
            if path.is_file() and not path.is_symlink(): entries[path.as_posix()] = path
    for path in run.parent.iterdir():
        if path.is_file() and path.suffix in ('.log', '.json') and '.receipt.' not in path.name:
            entries[path.as_posix()] = path
    for name, digest in protocol['assets'].items():
        path = Path(name)
        if file_sha256(path) != digest: raise ValueError('Frozen asset changed: ' + name)
        # Model weights are distributed by the predecessor resource packages.
        if not name.startswith('checkpoints/'):
            entries[name] = path
    for name, digest in protocol['code_sha256'].items():
        if file_sha256(name) != digest: raise ValueError('Executed code changed: ' + name)
    for folder in ('vguard', 'scripts', 'tools', 'deploy', 'tests', 'third_party/ada_in_vc'):
        for path in Path(folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.sh', '.txt', '.yaml'):
                entries['executed-source/' + path.as_posix()] = path
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, 'w:gz') as archive:
        for name, path in sorted(entries.items()): archive.add(path, arcname=name, recursive=False)
    receipt = {'archive': args.archive.name, 'bytes': args.archive.stat().st_size,
        'sha256': file_sha256(args.archive), 'measured_wavs_verified_against_scores': len(measured),
        'wav_files': sum(name.endswith('.wav') for name in entries),
        'development_only': args.development_only,
        'files_sha256': {name: file_sha256(path) for name, path in sorted(entries.items())},
        'note': 'Model weights excluded; use the existing complete model/ASR packages. Original input WAVs, enrolled prototypes, outputs and frozen execution bytes included. ECAPA is an optimization surrogate, not independent evaluation.'}
    write_json(receipt_path, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'files_sha256'}, indent=2))


if __name__ == '__main__': main()
