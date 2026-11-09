import copy
import unittest
from tools.analyze_staged_recovery import cluster_intervals, summary


class StagedPairedAnalysisTests(unittest.TestCase):
    def records(self, success=False, full=True):
        return [{'utterance': speaker + '_040', 'gates_ge2e': {
            'clean_identity_ok': True, 'clean_full_quality_ok': full,
            'strict_joint': success, 'paper_full_screen': success and full}}
            for speaker in ('p225', 'p226') for _ in range(2)]

    def test_correlated_repetitions_are_resampled_as_two_speaker_clusters(self):
        result = cluster_intervals(self.records(), self.records(True), 'gates_ge2e', samples=100)
        self.assertEqual(result['cluster_count'], 2)
        self.assertEqual(result['strict_joint']['paired_difference_95'], [1., 1.])

    def test_zero_full_clean_denominator_is_na(self):
        result = cluster_intervals(self.records(full=False), self.records(full=False), 'gates_ge2e', samples=100)
        self.assertIsNone(result['paper_full_screen']['staged_95'])
        self.assertEqual(result['paper_full_screen']['zero_denominator_resamples'], 100)

    def test_different_paired_clean_controls_are_rejected(self):
        before = self.records(); after = copy.deepcopy(before)
        after[0]['gates_ge2e']['clean_identity_ok'] = False
        with self.assertRaisesRegex(ValueError, 'denominators differ'):
            cluster_intervals(before, after, 'gates_ge2e', samples=100)

    def test_quality_failure_counts_deduplicate_repeated_content_sources(self):
        row = {'utterance': 'p225_040', 'original': {'mos_original': 4., 'wer_small_en': 0.},
            'defended': {'mos_original': 2.5, 'wer_small_en': .2,
                'wer_english_small_en': .2, 'asv_ge2e': 'p225'},
            'snr_db': 20., 'mask_exceed_fraction': .01, 'mask_excess_mean_db': .1,
            'clean_vc': {'mos_original': 4.}, 'protected_vc': {'mos_original': 3., 'asv_ge2e': 'p225'},
            'gates_ge2e': {'clean_identity_ok': True, 'clean_full_quality_ok': True,
                'protected_quality_ok': False, 'strict_joint': False,
                'paper_identity_screen': False, 'paper_full_screen': False}}
        result = summary([row, copy.deepcopy(row)])
        self.assertEqual(result['utterances'], 1)
        self.assertEqual(result['conversion_pairs'], 2)
        self.assertEqual(result['protected_quality_failure_reasons_nonexclusive'],
            {'identity': 0, 'mos': 1, 'wer': 1})
        self.assertEqual(result['asr_quality_comparison_to_clean_original']['only_protected_exceeds_10pct'], 1)


if __name__ == '__main__': unittest.main()
