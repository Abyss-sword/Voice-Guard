"""Partial quality evaluations must not become a completed campaign."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import build_cloud_report
from vguard.run_state import ARTIFACTS, write_csv, write_json


class CloudReportTests(unittest.TestCase):
    def test_requires_all_artifacts_and_real_transcript_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "main_vg"
            rows = [{"utterance": f"p{225+i}_040", "defense_ok": True,
                     "conv_clean_hits_X": i != 4} for i in range(5)]
            for row in rows:
                target = directory / row["utterance"]
                write_json(target / "result.json", row)
                for name in ARTIFACTS:
                    (target / name).touch()
            write_csv(directory / "results.csv", rows)
            keys = ("original_trimmed", "defended", "conv_from_clean", "conv_from_defended")
            mos = [{"utterance": row["utterance"], **{f"mos_{key}": 3.1 for key in keys}} for row in rows]
            wer = [{"utterance": row["utterance"], "reference_mode": "whisper",
                    **{f"wer_{key}": .2 for key in keys}} for row in rows]
            def report():
                with patch.object(build_cloud_report, "EXPECTED", ["main_vg"]), \
                     patch("sys.argv", ["report", "--root", str(root)]):
                    build_cloud_report.main()
                return json.loads((root / "campaign-summary.json").read_text())
            self.assertFalse(report()["complete"])
            write_csv(directory / "mos.csv", mos)
            write_csv(directory / "wer.csv", wer)
            self.assertFalse(report()["complete"])
            for row in wer:
                row["reference_mode"] = "transcript"
            write_csv(directory / "wer.csv", wer[:-1])
            self.assertFalse(report()["complete"])
            write_csv(directory / "wer.csv", wer)
            result = report()
            self.assertTrue(result["complete"])
            self.assertEqual(result["groups"][0]["screened"], 4)
            self.assertEqual(result["groups"][0]["screened_successes"], 4)
            (directory / rows[0]["utterance"] / "conv_from_clean.wav").unlink()
            self.assertFalse(report()["complete"])


if __name__ == "__main__":
    unittest.main()
