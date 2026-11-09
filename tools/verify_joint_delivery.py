"""Verify frozen execution, paired analysis, archive, tests and reviewed report."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_joint_recovery import analyze
from vguard.run_state import file_sha256, write_json


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-reviewed', action='store_true', required=True,
        help='Use only after viewing every latest rendered PDF page.')
    args = parser.parse_args()
    evidence = ROOT / 'docs/evidence/2026-11-09-joint'
    archive_root = ROOT / 'outputs/joint_recovery/remote'
    protocol = read(evidence / 'protocol.json')
    code = {}
    for name, digest in protocol['code_sha256'].items():
        executed = archive_root / 'executed-source' / name
        if file_sha256(executed) != digest: raise ValueError('Execution byte digest mismatch: ' + name)
        current_bytes, execution_bytes = (ROOT / name).read_bytes(), executed.read_bytes()
        if current_bytes.replace(b'\r\n', b'\n') != execution_bytes.replace(b'\r\n', b'\n'):
            raise ValueError('Delivery source changed beyond line endings: ' + name)
        code[name] = {'execution_sha256': digest, 'delivery_sha256': file_sha256(ROOT / name),
            'byte_identical': current_bytes == execution_bytes, 'equal_after_lf_normalization': True}
    run = archive_root / protocol['output']
    for split in ('development', 'validation'):
        name = split + '-results.json'
        if (evidence / name).read_bytes() != (run / name).read_bytes():
            raise ValueError('Published raw results differ: ' + name)
    if analyze(evidence, run) != read(evidence / 'analysis.json'):
        raise ValueError('Locally recomputed analysis differs')
    archive = read(evidence / 'archive-verification.json')
    if not archive['local_archive_verified']: raise ValueError('Archive not verified')
    archive_path = ROOT / 'outputs/joint_recovery/joint-evidence.tar.gz'
    if file_sha256(archive_path) != archive['sha256']: raise ValueError('Archive bytes changed')
    for name, digest in protocol['assets'].items():
        if name.startswith('checkpoints/'): continue
        if file_sha256(archive_root / name) != digest: raise ValueError('Archived input changed: ' + name)
    log = archive_root / 'outputs/joint_recovery/tests.log'
    text = log.read_text(encoding='utf-8')
    match = re.search(r'Ran (\d+) tests? in', text)
    if not match or not re.search(r'^OK\s*$', text, re.M): raise ValueError('Tests did not pass')
    prior = {'outputs/course/final-complete-evidence.tar.gz':
        '4984c94226caae7eb4602cb804b6217bf94d1eedd34c73e898a944a3ed341b3f',
        'outputs/paper_quality/final-recovery-evidence.tar.gz':
        '9a2aeb9e389b5910c0f5723ff5c87ebc40a13925efbf760c2c10a045f61a3a5d',
        'outputs/staged_recovery/final-staged-evidence.tar.gz':
        '874865a9d35672143a8cbbac105bb5724189698d805fd4d0efaf8f5ea2134c80'}
    for name, digest in prior.items():
        if file_sha256(ROOT / name) != digest: raise ValueError('Prior archive changed')
    report_inputs = read(evidence / 'report-inputs.json')
    doc = ROOT / 'docs/16_输出波形与联合质量优化.md'
    if file_sha256(doc) != report_inputs['markdown_sha256']: raise ValueError('Report text changed')
    extension = ROOT / 'docs/evidence/2026-11-09-joint-anchored'
    extension_root = ROOT / 'outputs/joint_anchored/remote'
    extension_protocol = read(extension / 'protocol.json')
    extension_code = {}
    for name, digest in extension_protocol['code_sha256'].items():
        executed = extension_root / 'executed-source' / name
        if file_sha256(executed) != digest: raise ValueError('Extension execution digest mismatch: ' + name)
        current_bytes, execution_bytes = (ROOT / name).read_bytes(), executed.read_bytes()
        if current_bytes.replace(b'\r\n', b'\n') != execution_bytes.replace(b'\r\n', b'\n'):
            raise ValueError('Extension source differs beyond line endings: ' + name)
        extension_code[name] = {'execution_sha256': digest, 'delivery_sha256': file_sha256(ROOT / name),
            'byte_identical': current_bytes == execution_bytes, 'equal_after_lf_normalization': True}
    decision = read(extension / 'extension-decision.json')
    selection = read(extension / 'selection.json')
    if decision['protocol_sha256'] != file_sha256(extension / 'protocol.json') or decision['selection_sha256'] != file_sha256(extension / 'selection.json'):
        raise ValueError('Extension decision changed')
    expected_gate = selection['selected']['full_joint'] >= 5 and selection['selected']['protected_quality'] == 3
    if decision['validation_allowed'] != expected_gate: raise ValueError('Preregistered extension gate differs')
    extension_run = extension_root / extension_protocol['output']
    for split in (('development', 'validation') if expected_gate else ('development',)):
        name = split + '-results.json'
        if (extension / name).read_bytes() != (extension_run / name).read_bytes():
            raise ValueError('Published extension raw results differ: ' + name)
    if expected_gate:
        if analyze(extension, extension_run) != read(extension / 'analysis.json'):
            raise ValueError('Extension analysis differs')
    elif (extension_run / 'validation-state.json').exists():
        raise ValueError('Validation ran despite a failed development gate')
    extension_archive = read(extension / 'archive-verification.json')
    if not extension_archive['local_archive_verified'] or file_sha256(ROOT / 'outputs/joint_anchored/anchored-evidence.tar.gz') != extension_archive['sha256']:
        raise ValueError('Extension archive not verified')
    observation = extension_root / 'outputs/joint_anchored/first-joint-partial-observation.json'
    if file_sha256(observation) != extension_protocol['decision_snapshot_sha256']:
        raise ValueError('Extension decision observation changed')
    for name, digest in extension_protocol['assets'].items():
        if not name.startswith('checkpoints/') and file_sha256(extension_root / name) != digest:
            raise ValueError('Extension input changed: ' + name)
    combined_doc = ROOT / 'docs/17_防护保持约束与联合质量验证.md'
    if file_sha256(combined_doc) != read(extension / 'report-inputs.json')['markdown_sha256']:
        raise ValueError('Combined report changed')
    pdf = ROOT / 'deliverables/Voice_Guard_联合质量优化报告.pdf'
    from pypdf import PdfReader
    pdf_pages = len(PdfReader(pdf).pages)
    pages = sorted((ROOT / 'outputs/joint_recovery/report-pages').glob('page-*.png'))
    if len(pages) != pdf_pages: raise ValueError('PDF/render count mismatch')
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': file_sha256(evidence / 'protocol.json'),
        'selection_sha256': file_sha256(evidence / 'selection.json'),
        'execution_source_verified': code, 'independent_analysis_recomputed': True,
        'archive': archive, 'prior_archives_unchanged': prior,
        'extension': {'execution_source_verified': extension_code, 'archive': extension_archive,
            'validation_allowed': expected_gate, 'decision_sha256': file_sha256(extension / 'extension-decision.json'),
            'independent_analysis_recomputed': expected_gate,
            'combined_markdown_sha256': file_sha256(combined_doc)},
        'unit_tests': {'count': int(match[1]), 'passed': True, 'log_sha256': file_sha256(log)},
        'report': {'path': pdf.relative_to(ROOT).as_posix(), 'sha256': file_sha256(pdf),
            'pages': pdf_pages, 'rendered_page_sha256': {p.name: file_sha256(p) for p in pages},
            'visual_review': 'All latest rendered pages inspected; no clipping or overlap observed'},
        'human_mos_ratings': 0, 'ecapa_participates_in_gradient': True,
        'automatic_shutdown': False, 'claim': 'Exploratory paired validation; no paper-scale equivalence claim'}
    write_json(evidence / 'final-verification.json', result)
    write_json(extension / 'final-verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('execution_source_verified', 'archive', 'extension')}, indent=2))


if __name__ == '__main__': main()
