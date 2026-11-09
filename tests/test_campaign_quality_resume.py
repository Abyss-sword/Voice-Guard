import csv
from pathlib import Path
import tempfile
import unittest
from deploy.run_course_campaign import saved_quality_digest


class CampaignQualityResumeTests(unittest.TestCase):
    def test_complete_transcript_scores_preserved_but_partial_or_pseudo_scores_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            utterances=[f'p22{i}_040' for i in range(5)]
            with (root/'results.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=['utterance']);writer.writeheader()
                writer.writerows({'utterance':u} for u in utterances)
            command=['python','scripts/compute_wer.py','--exp-dir',str(root),'--out',str(root/'wer.csv')]
            fields=['utterance','reference_mode',*[f'wer_{k}' for k in ('original_trimmed','defended','conv_from_clean','conv_from_defended')]]
            def write(count,reference='transcript',value=0):
                with (root/'wer.csv').open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                    writer.writerows({'utterance':u,'reference_mode':reference,**{k:value for k in fields[2:]}} for u in utterances[:count])
            write(5);digest=saved_quality_digest(command)
            self.assertEqual(len(digest),64)
            write(4);self.assertIsNone(saved_quality_digest(command))
            write(5,'whisper');self.assertIsNone(saved_quality_digest(command))
            write(5,value=float('nan'));self.assertIsNone(saved_quality_digest(command))

    def test_nonquality_stage_never_skipped(self):
        self.assertIsNone(saved_quality_digest(['python','scripts/run_experiment.py']))


if __name__=='__main__':unittest.main()
