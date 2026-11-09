"""Check frozen execution bytes, locally verified audio archive, tests and report."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vguard.run_state import file_sha256, write_json
from tools.analyze_staged_recovery import analyze


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-reviewed', action='store_true', required=True,
        help='Use only after inspecting every latest rendered report page.')
    args = parser.parse_args()
    evidence = ROOT / 'docs/evidence/2026-11-09-staged/final'
    remote = ROOT / 'outputs/staged_recovery/remote'
    protocol = read(evidence / 'protocol.json')
    code = {}
    for study in (evidence.parent, evidence):
        for name, digest in read(study / 'protocol.json')['code_sha256'].items():
            executed = remote / 'executed-source' / name
            if file_sha256(executed) != digest: raise ValueError('Archived execution code differs: ' + name)
            current_bytes, executed_bytes = (ROOT / name).read_bytes(), executed.read_bytes()
            if current_bytes.replace(b'\r\n', b'\n') != executed_bytes.replace(b'\r\n', b'\n'):
                raise ValueError('Delivery code differs beyond line endings: ' + name)
            code[name] = {'execution_sha256': digest, 'delivery_sha256': file_sha256(ROOT / name),
                'byte_identical': current_bytes == executed_bytes, 'equal_after_lf_normalization': True}
    run = remote / protocol['output']
    recomputed = analyze(evidence / 'protocol.json', evidence / 'selection.json', run)
    if recomputed != read(evidence / 'analysis.json'): raise ValueError('Recomputed analysis differs')
    native_audit = read(evidence / 'native-validation-audit.json')
    if native_audit['audit_code_sha256'] != file_sha256(ROOT / 'tools/audit_native_validation.py'):
        raise ValueError('Native audit code differs')
    production = read(evidence / 'production-registry-check.json')
    if not production['passed'] or production['builder_sha256'] != file_sha256(ROOT / 'scripts/build_registry.py'):
        raise ValueError('Production enrollment not verified')
    archive = read(evidence / 'archive-verification.json')
    if not archive['local_archive_verified']: raise ValueError('Archive not locally verified')
    bank = remote / protocol['native_target_bank']
    if file_sha256(bank) != protocol['assets'][protocol['native_target_bank']]:
        raise ValueError('Native reference bank differs')
    prior = {'outputs/course/final-complete-evidence.tar.gz':
        '4984c94226caae7eb4602cb804b6217bf94d1eedd34c73e898a944a3ed341b3f',
        'outputs/paper_quality/final-recovery-evidence.tar.gz':
        '9a2aeb9e389b5910c0f5723ff5c87ebc40a13925efbf760c2c10a045f61a3a5d'}
    for name, digest in prior.items():
        if file_sha256(ROOT / name) != digest: raise ValueError('Previous archive changed')
    log = ROOT / 'outputs/staged_recovery/final-tests.log'
    log_text = log.read_text(encoding='utf-8')
    match = re.search(r'Ran (\d+) tests? in', log_text)
    if not match or not re.search(r'^OK\s*$', log_text, re.M): raise ValueError('Tests not passed')
    pdf = ROOT / 'deliverables/Voice_Guard_原生参考与两阶段报告.pdf'
    from pypdf import PdfReader
    page_count = len(PdfReader(pdf).pages)
    pages = sorted((ROOT / 'outputs/staged_recovery/report-pages').glob('page-*.png'))
    if len(pages) != page_count: raise ValueError('Render count differs')
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_code_sha256_verified': code,
        'protocol_sha256': file_sha256(evidence / 'protocol.json'),
        'selection_sha256': file_sha256(evidence / 'selection.json'),
        'independent_analysis_recomputed': True,
        'native_audit_code_sha256_verified': native_audit['audit_code_sha256'],
        'production_enrollment_check': production,
        'unit_tests': {'count': int(match[1]), 'passed': True, 'log_sha256': file_sha256(log)},
        'archive': archive, 'native_reference_bank_sha256': file_sha256(bank),
        'previous_archives_unchanged': prior,
        'report': {'path': pdf.relative_to(ROOT).as_posix(), 'sha256': file_sha256(pdf),
            'pages': page_count, 'rendered_page_sha256': {p.name: file_sha256(p) for p in pages},
            'visual_review': 'Every latest rendered page inspected; no clipping or overlap observed'},
        'human_mos_ratings': 0, 'automatic_shutdown': False,
        'claim': 'Paired exploratory reproduction; paper-scale equivalence not established'}
    write_json(evidence / 'final-verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('frozen_code_sha256_verified', 'archive')}, indent=2))


if __name__ == '__main__': main()
