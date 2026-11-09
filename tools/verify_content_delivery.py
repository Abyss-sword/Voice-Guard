"""Verify content-study execution, independent scores, archives and reviewed PDF."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import hashlib
import zipfile
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_content_results import audit
from tools.probe_content_channels import channel_summary
from vguard.metrics.staged import gates
from vguard.run_state import file_sha256, write_json


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def compare_analysis(actual, saved, path=''):
    """Keep discrete results exact; allow only numerical roundoff in floats."""
    if isinstance(actual, dict) and isinstance(saved, dict):
        if actual.keys() != saved.keys(): raise ValueError('Analysis keys differ: ' + path)
        return [item for key in actual for item in
            compare_analysis(actual[key], saved[key], path + '/' + key)]
    if isinstance(actual, list) and isinstance(saved, list):
        if len(actual) != len(saved): raise ValueError('Analysis length differs: ' + path)
        return [item for index, (a, b) in enumerate(zip(actual, saved)) for item in
            compare_analysis(a, b, path + '/' + str(index))]
    if type(actual) is not type(saved): raise ValueError('Analysis type differs: ' + path)
    if actual == saved: return []
    if isinstance(actual, float) and math.isclose(actual, saved, rel_tol=0, abs_tol=1e-12):
        return [{'path': path, 'recomputed': actual, 'saved': saved,
            'absolute_difference': abs(actual - saved)}]
    raise ValueError('Recomputed analysis differs: ' + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-reviewed', action='store_true', required=True)
    parser.parse_args()
    evidence = ROOT / 'docs/evidence/2026-11-09-content-robust'
    extracted = ROOT / 'outputs/content_robust/remote'
    protocol = read(evidence / 'protocol.json'); run = extracted / protocol['output']
    expected = audit(evidence, run)
    roundoff = compare_analysis(expected, read(evidence / 'analysis.json'))
    code = {}
    sources = {**protocol['code_sha256'], 'tools/probe_content_channels.py':
        read(evidence / 'channel-protocol.json')['code_sha256']}
    for name, sha in sources.items():
        actual, delivered = extracted / 'executed-source' / name, ROOT / name
        if file_sha256(actual) != sha: raise ValueError('Execution bytes changed: ' + name)
        if actual.read_bytes().replace(b'\r\n', b'\n') != delivered.read_bytes().replace(b'\r\n', b'\n'):
            raise ValueError('Delivery differs beyond line endings: ' + name)
        code[name] = {'execution_sha256': sha, 'delivery_sha256': file_sha256(delivered),
            'equal_after_lf_normalization': True}
    for name, sha in protocol['assets'].items():
        if not name.startswith('checkpoints/') and file_sha256(extracted / name) != sha:
            raise ValueError('Input/prototype changed')
    for split in expected['splits']:
        name = split + '-results.json'
        if (run / name).read_bytes() != (evidence / name).read_bytes():
            raise ValueError('Published raw scores differ')
    channel_protocol = read(evidence / 'channel-protocol.json')
    rows = read(run / 'channel-results.json')['rows']
    state = read(run / 'channel-state.json')
    tuples = {(u, c, p) for u in channel_protocol['utterances'] for c in channel_protocol['channels']
        for p in channel_protocol['profiles']}
    if state['state'] != 'complete' or state['rows'] != len(tuples) or len(rows) != len(tuples):
        raise ValueError('Channel probe incomplete')
    if {(r['utterance'], r['channel'], r['profile']) for r in rows} != tuples:
        raise ValueError('Missing/duplicate channel observations')
    controls = {}
    for row in rows:
        if row['channel_protocol_sha256'] != file_sha256(evidence / 'channel-protocol.json'):
            raise ValueError('Channel protocol mismatch')
        key = row['utterance'], row['channel']
        clean = {k: row[k] for k in ('original', 'clean_vc')}
        if key in controls and controls[key] != clean: raise ValueError('Channel controls not paired')
        controls[key] = clean
        for label, path in row['paths'].items():
            if file_sha256(extracted / path) != row[label]['sha256']:
                raise ValueError('Channel WAV differs')
        for asv, asr in [('ge2e', 'small_en'), ('ecapa', 'base')]:
            if row['gates_' + asv] != gates(row['utterance'], row['original'], row['clean_vc'],
                    row['defended'], row['protected_vc'], asv=asv, asr=asr):
                raise ValueError('Channel gates differ')
    channel_analysis = read(evidence / 'channel-analysis.json')
    recomputed = {channel: {name: channel_summary([r for r in rows
        if r['channel'] == channel and r['profile'] == name]) for name in channel_protocol['profiles']}
        for channel in channel_protocol['channels']}
    if recomputed != channel_analysis['by_channel']: raise ValueError('Channel summary differs')
    if (evidence / 'channel-results.json').read_bytes() != (run / 'channel-results.json').read_bytes():
        raise ValueError('Published channel rows differ')
    archive = read(evidence / 'archive-verification.json')
    archive_path = ROOT / 'outputs/content_robust/content-evidence.tar.gz'
    if not archive['local_archive_verified'] or file_sha256(archive_path) != archive['sha256']:
        raise ValueError('Archive digest differs')
    old = {'outputs/course/final-complete-evidence.tar.gz':
        '4984c94226caae7eb4602cb804b6217bf94d1eedd34c73e898a944a3ed341b3f',
        'outputs/paper_quality/final-recovery-evidence.tar.gz':
        '9a2aeb9e389b5910c0f5723ff5c87ebc40a13925efbf760c2c10a045f61a3a5d',
        'outputs/staged_recovery/final-staged-evidence.tar.gz':
        '874865a9d35672143a8cbbac105bb5724189698d805fd4d0efaf8f5ea2134c80',
        'outputs/joint_recovery/joint-evidence.tar.gz':
        'b231f74d1f2d7dd8ca9b1a9dbb242d528c33557154bcedb8d8888aa47f09c5f0',
        'outputs/joint_anchored/anchored-evidence.tar.gz':
        '28526b1cedade34a67347d33125e407a5e5c57cd09941a7eb7f6656c599c0229'}
    for name, sha in old.items():
        if file_sha256(ROOT / name) != sha: raise ValueError('Historical archive changed')
    resource = read(evidence / 'ctc-resource-receipt.json')
    resource_path = ROOT / 'outputs/content_robust' / resource['archive']
    if file_sha256(resource_path) != resource['sha256'] or resource_path.stat().st_size != resource['bytes']:
        raise ValueError('CTC resource package differs')
    with zipfile.ZipFile(resource_path) as package:
        if set(package.namelist()) != {resource['checkpoint_path'], 'CONTENT-CTC-README.txt'}:
            raise ValueError('Unexpected CTC resource members')
        digest = hashlib.sha256()
        with package.open(resource['checkpoint_path']) as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != protocol['assets'][resource['checkpoint_path']]:
            raise ValueError('Packaged CTC checkpoint differs')
    log = extracted / 'outputs/content_robust/tests.log'
    test_text = log.read_text(encoding='utf-8')
    count = re.search(r'Ran (\d+) tests? in', test_text)
    if not count or not re.search(r'^OK\s*$', test_text, re.M): raise ValueError('Tests did not pass')
    if int(count[1]) != read(evidence / 'test-verification.json')['count']:
        raise ValueError('Test count differs')
    markdown = ROOT / 'docs/18_内容监督与环境鲁棒优化.md'
    if file_sha256(markdown) != read(evidence / 'report-inputs.json')['markdown_sha256']:
        raise ValueError('Report changed after generation')
    pdf = ROOT / 'deliverables/Voice_Guard_内容监督与环境鲁棒报告.pdf'
    from pypdf import PdfReader
    pages = sorted((ROOT / 'outputs/content_robust/report-pages').glob('page-*.png'))
    if len(pages) != len(PdfReader(pdf).pages): raise ValueError('PDF/preview count differs')
    write_json(evidence / 'final-verification.json', {
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'execution_source_verified': code, 'independent_analysis_recomputed': True,
        'analysis_comparison': {'discrete_values_exact': True,
            'float_absolute_tolerance': 1e-12, 'roundoff_differences': roundoff},
        'channel_gates_and_controls_verified': True, 'archive': archive,
        'ctc_resource_package_verified': resource,
        'historical_archives_unchanged': old, 'unit_tests': int(count[1]),
        'unit_test_log_sha256': file_sha256(log),
        'validation_allowed': expected['validation_allowed'],
        'report': {'path': pdf.relative_to(ROOT).as_posix(), 'sha256': file_sha256(pdf),
            'pages': len(pages), 'markdown_sha256': file_sha256(markdown),
            'rendered_page_sha256': {p.name: file_sha256(p) for p in pages},
            'visual_review': 'All latest rendered pages inspected; no clipping or overlap observed'},
        'human_mos_ratings': 0, 'automatic_shutdown': False,
        'claim': 'Controlled exploratory extension, not a verified Kalypta implementation or paper-scale reproduction'})
    print('Verified:', int(count[1]), 'tests;', len(pages), 'reviewed pages; five prior archives unchanged')


if __name__ == '__main__': main()
