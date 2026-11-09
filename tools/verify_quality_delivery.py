"""Verify recovery delivery hashes, enrollment separation and rendered PDF evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-review-complete', action='store_true')
    args = parser.parse_args()
    if not args.visual_review_complete:
        raise ValueError('Inspect the latest rendered pages before finalizing')
    evidence = Path('docs/evidence/2026-11-09-quality')
    outputs = Path('outputs/paper_quality')
    archive_root = outputs / 'remote'
    def read(path): return json.loads(path.read_text(encoding='utf-8'))
    protocol = read(evidence / 'recovery-protocol.json')
    code_verified = {}
    for path, digest in protocol['code_sha256'].items():
        if file_sha256(path) != digest:
            raise ValueError('Frozen execution code changed: ' + path)
        code_verified[path] = digest
    supplement = read(evidence / 'supplemental-protocol.json')
    if file_sha256('tools/evaluate_quality_supplement.py') != supplement['scorer_sha256']:
        raise ValueError('Frozen supplementary scorer changed')
    parity = read(evidence / 'mosnet-parity.json')
    if not parity['passed'] or parity['port_sha256'] != file_sha256('vguard/metrics/mos_original.py'):
        raise ValueError('MOSNet port no longer matches verified parity')
    overlap = {}
    for name in ('registry.npz.manifest.json', 'registry_vc.npz.manifest.json'):
        path = evidence / name
        manifest = read(path)
        paths = {entry['path'] for entries in manifest['enrollment'].values() for entry in entries}
        overlap[name] = [u for u in protocol['validation'] if any(p.endswith('/' + u + '.wav') for p in paths)]
    ge2e = read(evidence / 'ge2e-resources.json')
    overlap['GE2E'] = [u for u in protocol['validation'] if any(p.endswith('/' + u + '.wav') for p in ge2e['enrollment_sha256'])]
    if any(overlap.values()): raise ValueError('Validation overlaps enrollment')
    resource = outputs / 'VoiceGuard-quality-evaluator-resources.zip'
    receipt = read(outputs / (resource.name + '.receipt.json'))
    if resource.stat().st_size != receipt['bytes'] or file_sha256(resource) != receipt['sha256']:
        raise ValueError('Downloaded resource bundle incomplete or corrupted')
    with zipfile.ZipFile(resource) as package:
        for name, expected in receipt['files_sha256'].items():
            hasher = hashlib.sha256()
            with package.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''): hasher.update(chunk)
            if hasher.hexdigest() != expected: raise ValueError('Packaged model differs: ' + name)
    receipt['local_archive_and_members_verified'] = True
    write_json(evidence / 'quality-resources.receipt.json', receipt)
    import pypdfium2 as pdfium
    pdf_path = Path('deliverables/Voice_Guard_论文质量补充报告.pdf')
    pdf = pdfium.PdfDocument(str(pdf_path))
    text = '\n'.join(page.get_textpage().get_text_range() for page in pdf)
    for expected in ('4.063', '4.61%', '16/21', '5/12', 'p227_053'):
        if expected not in text: raise ValueError('Expected result missing from report: ' + expected)
    if 'FINAL_RECOVERY_RESULTS' in text: raise ValueError('Internal Markdown marker in PDF')
    pages = [outputs / f'report-pages/page-{n+1:02d}.png' for n in range(len(pdf))]
    tests = evidence / 'final-tests.log'
    test_output = tests.read_text(encoding='utf-8')
    if 'Ran 30 tests' not in test_output or '\nOK\n' not in test_output:
        raise ValueError('Expected complete test suite result')
    record = {
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_code_sha256_verified': code_verified,
        'supplement_scorer_sha256_verified': supplement['scorer_sha256'],
        'enrollment_overlap': overlap,
        'unit_tests': {'count': 30, 'passed': True, 'log_sha256': file_sha256(tests)},
        'report': {'path': pdf_path.as_posix(), 'sha256': file_sha256(pdf_path), 'pages': len(pdf),
                   'rendered_page_sha256': {p.name: file_sha256(p) for p in pages},
                   'visual_review': 'All latest rendered pages inspected; no clipped text or overlap observed'},
        'resource_bundle_sha256': receipt['sha256'], 'resource_bundle_local_verified': True,
        'experiment_archive': read(evidence / 'recovery-archive-verification.json'),
        'human_mos_ratings': 0, 'automatic_shutdown': False,
        'claim': 'Quality predictions near paper targets, overall paper-level protection not established',
    }
    write_json(evidence / 'final-validation.json', record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
