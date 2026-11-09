"""Exercise the evidence auditor on complete records and realistic corruptions."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/2026-11-09-quality'


class RecoveryEvidenceTests(unittest.TestCase):
    def audit(self, mutation=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / 'evidence'
            results = root / 'results'
            evidence.mkdir(); results.mkdir()
            for name in ('recovery-protocol.json', 'recovery-selection.json', 'supplemental-protocol.json'):
                shutil.copyfile(EVIDENCE / name, evidence / name)
            for name in ('validation-state.json', 'validation-results.json', 'validation-supplement.json'):
                shutil.copyfile(EVIDENCE / name, results / name)
            if mutation:
                mutation(results)
            process = subprocess.run([sys.executable, str(ROOT / 'tools/analyze_quality_recovery.py'),
                '--root', str(results), '--evidence', str(evidence)], capture_output=True, text=True)
            report = evidence / 'recovery-analysis.json'
            return process, json.loads(report.read_text()) if report.exists() else None

    def test_complete_frozen_records_are_audited(self):
        process, report = self.audit()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(report['counts_verified'])
        self.assertEqual(report['evaluations']['ECAPA_Whisper_base']['all']['pooled_correlated_pairs']['utterances'], 15)

    def test_duplicate_pair_cannot_hide_missing_validation_pair(self):
        def corrupt(root):
            path = root / 'validation-results.json'
            value = json.loads(path.read_text())
            value['rows'][-1] = value['rows'][0]
            path.write_text(json.dumps(value))
        process, report = self.audit(corrupt)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('pairs missing or duplicated', process.stderr)
        self.assertIsNone(report)

    def test_supplement_must_score_identical_waveforms(self):
        def corrupt(root):
            path = root / 'validation-supplement.json'
            value = json.loads(path.read_text())
            value['rows'][0]['scores']['protected_vc']['sha256'] = '0' * 64
            path.write_text(json.dumps(value))
        process, report = self.audit(corrupt)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('different WAVs', process.stderr)
        self.assertIsNone(report)


if __name__ == '__main__': unittest.main()
