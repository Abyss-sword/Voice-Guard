"""Export and locally verify the paired staged experiment without overwriting prior archives."""
import argparse
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('outputs/staged_recovery/final-staged-evidence.tar.gz'))
    parser.add_argument('--verify-to', type=Path)
    parser.add_argument('--evidence', type=Path, default=Path('docs/evidence/2026-11-09-staged/final'))
    args = parser.parse_args()
    receipt_path = args.archive.with_suffix('.tar.gz.receipt.json') if args.archive.suffix != '.gz' else Path(str(args.archive) + '.receipt.json')
    evidence = args.evidence
    if args.verify_to:
        receipt = read(receipt_path)
        if args.archive.stat().st_size != receipt['bytes'] or file_sha256(args.archive) != receipt['sha256']:
            raise ValueError('Downloaded archive hash/size differs')
        destination = args.verify_to.resolve(); destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.archive, 'r:gz') as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if not member.isfile() or path.is_absolute() or '..' in path.parts or ':' in member.name:
                    raise ValueError('Unsafe archive member')
                if not (destination / member.name).resolve().is_relative_to(destination): raise ValueError('Archive escapes destination')
            archive.extractall(destination, filter='data')
        for name, expected in receipt['files_sha256'].items():
            if file_sha256(destination / name) != expected: raise ValueError('Extracted member differs: ' + name)
        record = {k: v for k, v in receipt.items() if k != 'files_sha256'}
        record.update(all_files_verified=len(receipt['files_sha256']), local_archive_verified=True)
        write_json(evidence / 'archive-verification.json', record)
        print(record); return
    if args.archive.exists() or receipt_path.exists(): raise ValueError('Immutable archive already exists')
    protocol = read(evidence / 'protocol.json'); run = Path(protocol['output'])
    analysis = read(evidence / 'analysis.json')
    if not analysis['counts_verified'] or read(run / 'validation-state.json')['state'] != 'complete':
        raise ValueError('Complete, audited validation required')
    measured = {}
    predecessor = run.parent if evidence.name == 'final' else run
    for study in dict.fromkeys((predecessor, run)):
        for split in ('development', 'validation'):
            result_path = study / (split + '-results.json')
            if not result_path.exists(): continue
            for row in read(result_path)['rows']:
                control = study / 'controls' / row['utterance']
                protected = study / split / row['profile'] / row['utterance']
                paths = [('original', control / 'original.wav'), ('clean_vc', control / (row['content'] + '.wav')),
                         ('defended', protected / 'defended.wav'), ('protected_vc', protected / (row['content'] + '.wav'))]
                for key, path in paths:
                    actual = measured.setdefault(path.as_posix(), file_sha256(path))
                    if actual != row[key]['sha256']: raise ValueError('Measured WAV differs: ' + str(path))
    audit_path = predecessor / 'strong-audit/results.json'
    if audit_path.exists():
        for row in read(audit_path)['rows']:
            paths = [('strong', predecessor / 'development' / row['profile'] / row['utterance'] / 'strong.wav'),
                     ('strong_vc', predecessor / 'strong-audit' / row['utterance'] / (row['content']+'.wav'))]
            for key, path in paths:
                measured[path.as_posix()] = file_sha256(path)
                if measured[path.as_posix()] != row[key]['sha256']: raise ValueError('Audit WAV differs')
    entries = {}
    for directory in (predecessor, evidence.parent if evidence.name == 'final' else evidence):
        for path in directory.rglob('*'):
            if path.is_file() and not path.is_symlink(): entries[path.as_posix()] = path
    for path in Path('outputs/staged_recovery').iterdir():
        if path.is_file() and path.suffix in {'.json', '.log', '.txt'}: entries[path.as_posix()] = path
    for directory in ('vguard', 'scripts', 'tools', 'deploy', 'tests', 'third_party/ada_in_vc'):
        for path in Path(directory).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.sh', '.txt'}:
                entries['executed-source/' + path.as_posix()] = path
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, 'w:gz') as archive:
        for name, path in sorted(entries.items()): archive.add(path, arcname=name, recursive=False)
    receipt = {'archive': args.archive.name, 'bytes': args.archive.stat().st_size, 'sha256': file_sha256(args.archive),
        'measured_wavs_verified_against_scores': len(measured),
        'wav_files': sum(name.endswith('.wav') for name in entries),
        'intermediate_strong_stage_wavs': sum(name.endswith('/strong.wav') for name in entries),
        'scored_strong_stage_wavs': sum(name.endswith('/strong.wav') for name in measured),
        'files_sha256': {name: file_sha256(path) for name, path in sorted(entries.items())},
        'note': 'Strong-stage WAVs are intermediate diagnostics, not final scored successes. Protocol code hashes identify executed bytes; later delivery commits must not be backdated.'}
    write_json(receipt_path, receipt)
    print({k: v for k, v in receipt.items() if k != 'files_sha256'})


if __name__ == '__main__': main()
