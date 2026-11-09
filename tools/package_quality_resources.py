"""Package the additional fixed recovery evaluators without changing the old bundle."""
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def main():
    import resemblyzer
    evidence = Path('docs/evidence/2026-11-09-quality')
    supplement = json.loads((evidence / 'supplemental-protocol.json').read_text())
    package = Path(resemblyzer.__file__).parent
    source = Path('checkpoints/mosnet-original')
    files = {f'checkpoints/mosnet-original/{p.name}': p for p in source.iterdir() if p.is_file()}
    files['whisper-cache/small.en.pt'] = Path.home() / '.cache/whisper/small.en.pt'
    files['resemblyzer/pretrained.pt'] = package / 'pretrained.pt'
    files['outputs/paper_quality/ge2e-registry.npz'] = Path(supplement['asv']['registry'])
    files['deploy/quality-recovery-requirements.txt'] = Path('deploy/quality-recovery-requirements.txt')
    for path in evidence.glob('*.json'):
        if path.name in {'ge2e-resources.json', 'supplemental-protocol.json', 'original-mosnet-source.json'}:
            files['docs/evidence/2026-11-09-quality/' + path.name] = path
    for distribution in ('Resemblyzer', 'openai-whisper'):
        dist = importlib.metadata.distribution(distribution)
        for entry in dist.files or []:
            if 'license' in str(entry).lower() and dist.locate_file(entry).is_file():
                files['licenses/' + distribution + '/' + Path(entry).name] = dist.locate_file(entry)
    hashes = {name: file_sha256(path) for name, path in files.items()}
    if hashes['whisper-cache/small.en.pt'] != supplement['asr']['expected_sha256']:
        raise ValueError('Whisper hash mismatch')
    if hashes['outputs/paper_quality/ge2e-registry.npz'] != supplement['asv']['registry_sha256']:
        raise ValueError('GE2E registry mismatch')
    out = Path('outputs/paper_quality/VoiceGuard-quality-evaluator-resources.zip')
    if out.exists(): raise ValueError('Do not overwrite resource bundle')
    instructions = '''Additional recovery evaluators, not a replacement for the earlier complete model bundle.
Restore checkpoints/, outputs/ and deploy/ under the project root.
Copy whisper-cache/small.en.pt to the current user's ~/.cache/whisper/small.en.pt.
Install deploy/quality-recovery-requirements.txt into the existing deployment venv.
Resemblyzer 0.1.4 ships pretrained.pt; compare its installed file against this copy.
Use the original project model package for AdaIN-VC, ECAPA, historical MOSNet and Whisper base.
Original MOSNet port does not require TensorFlow. Enrollment inputs are identified by SHA256.
No dataset or new license grant is included. Observe the original model/data licenses.
'''
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr('RESTORE.txt', instructions)
        archive.writestr('SHA256.json', json.dumps(hashes, indent=2))
        for name, path in sorted(files.items()): archive.write(path, name)
    receipt = {'archive': out.name, 'bytes': out.stat().st_size, 'sha256': file_sha256(out),
               'files_sha256': hashes, 'uploaded_cloud_share_link': None}
    write_json(out.with_suffix('.zip.receipt.json'), receipt)
    print({k: v for k, v in receipt.items() if k != 'files_sha256'})


if __name__ == '__main__': main()
