import unittest

from vguard.evaluation import identity_verdict, quality_pass, summarize_pairs, wilson_interval


class EvaluationTests(unittest.TestCase):
    def test_missing_asv_and_already_failed_clean_conversion_are_not_successes(self):
        self.assertFalse(identity_verdict({"defended": {"pred": "p225"}}, "p225")["identity_ok"])
        asv = {"defended": {"pred": "p225"}, "conv_clean": {"pred": "p226"},
               "conv_def": {"pred": "p227"}}
        verdict = identity_verdict(asv, "p225")
        self.assertTrue(verdict["identity_ok"])
        self.assertFalse(verdict["screened_identity_ok"])
        asv["conv_clean"]["pred"] = "p225"
        self.assertTrue(identity_verdict(asv, "p225")["screened_identity_ok"])
        asv["error"] = "partial inference"
        self.assertFalse(identity_verdict(asv, "p225")["identity_ok"])

    def test_zero_denominator_and_small_sample_uncertainty(self):
        self.assertIsNone(wilson_interval(0, 0))
        low, high = wilson_interval(1, 2)
        self.assertLess(low, .10)
        self.assertGreater(high, .90)
        row = {"clean_vc_identity_ok": False, "screened_quality_identity_ok": False,
               "mos_defended": 3.2, "wer_defended": 0}
        stats = summarize_pairs([row])
        self.assertIsNone(stats["joint_rate"])
        self.assertEqual(stats["mean_mos"], 3.2)

    def test_joint_quality_never_accepts_nonfinite_or_changed_protected_identity(self):
        self.assertTrue(quality_pass(3, .1, True))
        self.assertFalse(quality_pass(float("nan"), 0, True))
        self.assertFalse(quality_pass(4, float("inf"), True))
        self.assertFalse(quality_pass(4, 0, False))


if __name__ == "__main__":
    unittest.main()
