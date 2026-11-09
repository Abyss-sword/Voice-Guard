import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_joint_recovery as runner
from vguard.run_state import file_sha256, write_json


class JointProtocolTests(unittest.TestCase):
    def fixture(self, directory):
        protocol_path = directory / 'protocol.json'
        protocol = {'output': str(directory / 'run'), 'development': ['p225_040'],
            'evaluation_contents': ['p226_002', 'p226_004'],
            'baseline': {'name': 'baseline'},
            'profiles': [{'name': 'output_only'}, {'name': 'guarded'}],
            'selection_rule': 'full joint before identity-only count'}
        write_json(protocol_path, protocol)
        digest = file_sha256(protocol_path)
        run = Path(protocol['output'])
        write_json(run / 'development-state.json', {'state': 'complete', 'protocol_sha256': digest})
        rows = []
        for profile in ['baseline', 'output_only', 'guarded']:
            for content in protocol['evaluation_contents']:
                rows.append({'utterance': 'p225_040', 'content': content, 'profile': profile,
                    'protocol_sha256': digest,
                    'gates_ge2e': {'paper_full_screen': profile == 'guarded' and content == 'p226_004',
                        'strict_joint': profile == 'output_only', 'protected_quality_ok': True},
                    'defended': {'wer_small_en': 0., 'mos_original': 4.}})
        write_json(run / 'development-results.json', {'rows': rows})
        return protocol_path, protocol, rows

    def test_complete_joint_endpoint_drives_global_selection_and_cannot_be_reselected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, protocol, _ = self.fixture(Path(temporary))
            with patch.object(runner, 'PROTOCOL', path):
                runner.select(protocol)
                selection = json.loads(path.with_name('selection.json').read_text())
                self.assertEqual(selection['selected']['profile']['name'], 'guarded')
                self.assertFalse(selection['validation_examined'])
                with self.assertRaisesRegex(ValueError, 'already frozen'):
                    runner.select(protocol)

    def test_duplicate_record_cannot_hide_a_missing_candidate_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, protocol, rows = self.fixture(Path(temporary))
            rows[-1] = rows[0]
            write_json(Path(protocol['output']) / 'development-results.json', {'rows': rows})
            with patch.object(runner, 'PROTOCOL', path), self.assertRaisesRegex(ValueError, 'Missing/duplicate'):
                runner.select(protocol)


if __name__ == '__main__': unittest.main()
