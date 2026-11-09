"""Regression checks for interrupted runs, denominator errors, and audio sample rates."""
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from vguard.run_state import ARTIFACTS, completed_row, summarize, utterance_seed, write_csv, write_json
from tools.download_models import extract_vctk


class ReproductionTests(unittest.TestCase):
    def test_screened_denominator_excludes_already_failed_clean_conversion(self):
        rows = [{"defense_ok": "False", "conv_clean_hits_X": "True"},
                {"defense_ok": "True", "conv_clean_hits_X": "False"},
                {"asv_error": "missing checkpoint"}]
        stats = summarize(rows)
        self.assertEqual(stats["screened_rate"], 0)
        self.assertEqual(stats["unfiltered_rate"], .5)
        self.assertEqual(stats["asv_missing"], 1)
        self.assertIsNone(summarize([])["screened_rate"])

    def test_partial_attack_is_not_complete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "defended.wav").touch()
            self.assertIsNone(completed_row(root))
            write_json(root / "result.json", {"utterance": "p225_040"})
            self.assertIsNone(completed_row(root))
            for name in ARTIFACTS:
                (root / name).touch()
            self.assertEqual(completed_row(root)["utterance"], "p225_040")
            write_json(root / "result.json", {"asv_error": "failed"})
            self.assertIsNone(completed_row(root))

    def test_csv_keeps_optional_error_fields_and_records(self):
        with tempfile.TemporaryDirectory() as td:
            import csv
            path = Path(td) / "results.csv"
            write_csv(path, [{"utterance": "a", "defense_ok": False},
                             {"utterance": "b", "asv_error": "failure"}])
            with path.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["asv_error"], "failure")

    def test_per_utterance_seed_independent_of_execution_order(self):
        self.assertEqual(utterance_seed(2026, "a"), utterance_seed(2026, "a"))
        self.assertNotEqual(utterance_seed(2026, "a"), utterance_seed(2026, "b"))

    def test_vctk_layout_flattens_audio_and_transcripts(self):
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "sample.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("VCTK/wav22/p225/p225_001.wav", b"wav")
                z.writestr("VCTK/txt/p225/p225_001.txt", "sample text")
            output = Path(td) / "data"
            extract_vctk(archive, output)
            self.assertEqual((output / "p225/p225_001.txt").read_text(), "sample text")

    def test_stereo_22050_reference_is_resampled_to_mono_24000(self):
        import numpy as np
        import soundfile as sf
        from vguard.audio import load_audio
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ref.wav"
            x = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050).astype("float32") * .1
            sf.write(path, np.stack((x, x), axis=1), 22050)
            actual = load_audio(path, 24000)
            self.assertEqual(actual.ndim, 1)
            self.assertLessEqual(abs(len(actual) - 24000), 1)  # librosa rounds fractional lengths up
            self.assertTrue(np.isfinite(actual).all())

    def test_wer_uses_target_reference_for_defended_and_source_for_converted(self):
        from scripts import compute_wer
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            vctk = root / "data"
            for name, text in (("p225/p225_040.txt", "target sentence"),
                               ("p226/p226_002.txt", "source sentence")):
                path = vctk / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            exp = root / "exp/p225_040"
            exp.mkdir(parents=True)
            for name in ("original_trimmed", "defended", "conv_from_clean", "conv_from_defended"):
                (exp / f"{name}.wav").touch()
            references = []
            argv = ["compute_wer", "--exp-dir", str(exp.parent), "--vctk", str(vctk),
                    "--source-speaker", "p226", "--source-utt", "2", "--out", str(root / "wer.csv")]
            with patch("sys.argv", argv), patch.object(compute_wer, "load_whisper", return_value=object()), \
                 patch.object(compute_wer, "transcribe", return_value="hyp"), \
                 patch.object(compute_wer, "cer", return_value=0), \
                 patch.object(compute_wer, "wer", side_effect=lambda hyp, ref: references.append(ref) or 0):
                compute_wer.main()
            self.assertEqual(references, ["target sentence", "target sentence", "source sentence", "source sentence"])


if __name__ == "__main__":
    unittest.main()
