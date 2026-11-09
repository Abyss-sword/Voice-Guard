"""Export or verify immutable recovery evidence, including every measured WAV."""
from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('outputs/paper_quality/final-recovery-evidence.tar.gz'))
    parser.add_argument('--verify-to', type=Path)
    args = parser.parse_args()
    receipt_path = args.archive.with_suffix(args.archive.suffix + '.receipt.json')
    if args.verify_to:
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if file_sha256(args.archive) != receipt['sha256'] or args.archive.stat().st_size != receipt['bytes']:
            raise ValueError('Archive mismatch')
        destination = args.verify_to.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.archive, 'r:gz') as archive:
            members = archive.getmembers()
            for member in members:
                path = PurePosixPath(member.name)
                if not member.isfile() or path.is_absolute() or '..' in path.parts or ':' in member.name:
                    raise ValueError('Unsafe member: ' + member.name)
                if not (destination / member.name).resolve().is_relative_to(destination):
                    raise ValueError('Escapes destination')
            archive.extractall(destination, filter='data')
        for name, digest in receipt['files_sha256'].items():
            if file_sha256(destination / name) != digest:
                raise ValueError('Extracted evidence mismatch: ' + name)
        out = {'archive_sha256': receipt['sha256'], 'bytes': receipt['bytes'],
               'all_files_verified': len(receipt['files_sha256']),
               'wav_files_verified': sum(name.endswith('.wav') for name in receipt['files_sha256']),
               'measurement_audio_hashes_verified': receipt['measurement_audio_hashes_verified'],
               'verified': True}
        write_json(Path('docs/evidence/2026-11-09-quality/recovery-archive-verification.json'), out)
        print(out)
        return
    if args.archive.exists() or receipt_path.exists():
        raise ValueError('Refusing to overwrite immutable archive')
    run = Path('outputs/paper_quality/2026-11-09-recovery')
    state = json.loads((run / 'validation-state.json').read_text())
    primary = json.loads((run / 'validation-results.json').read_text())['rows']
    secondary = json.loads((run / 'validation-supplement.json').read_text())['rows']
    if state['state'] != 'complete' or len(primary) != 30 or len(secondary) != 30:
        raise ValueError('Complete primary and secondary validation required')
    checked = {}
    for split in ('development', 'validation'):
        rows = json.loads((run / (split + '-results.json')).read_text())['rows']
        for row in rows:
            protected = run / split / row['profile'] / row['utterance'] / f"step_{row['checkpoint']}"
            clean = run / 'controls' / row['utterance']
            for path, score in [(clean / 'original.wav', row['control']['original']),
                                (clean / (row['content'] + '.wav'), row['control']['converted']),
                                (protected / 'defended.wav', row['defended']),
                                (protected / (row['content'] + '.wav'), row['converted'])]:
                digest = checked.setdefault(str(path), file_sha256(path))
                if digest != score['sha256']:
                    raise ValueError('Measured audio differs: ' + str(path))
    entries = {}
    for directory in (run, Path('outputs/paper_quality/mosnet-parity'), Path('docs/evidence/2026-11-09-quality')):
        for path in sorted(directory.rglob('*')):
            if path.is_file() and not path.is_symlink(): entries[path.as_posix()] = path
    for path in Path('outputs/paper_quality').iterdir():
        if path.is_file() and path.suffix in {'.json', '.log', '.npz', '.txt'}:
            entries[path.as_posix()] = path
    for directory in ('vguard', 'scripts', 'tools', 'deploy', 'third_party/ada_in_vc'):
        for path in sorted(Path(directory).rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.sh', '.txt'}:
                entries['executed-source/' + path.as_posix()] = path
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, 'w:gz') as archive:
        for name, path in sorted(entries.items()): archive.add(path, arcname=name, recursive=False)
    receipt = {'archive': args.archive.name, 'bytes': args.archive.stat().st_size,
               'sha256': file_sha256(args.archive), 'measurement_audio_hashes_verified': len(checked),
               'files_sha256': {name: file_sha256(path) for name, path in sorted(entries.items())},
               'note': 'Snapshot at export; frozen protocol identifies exact code/assets actually used. New delivery code must not be backdated.'}
    write_json(receipt_path, receipt)
    print({k: v for k, v in receipt.items() if k != 'files_sha256'})


if __name__ == '__main__':
    main()
