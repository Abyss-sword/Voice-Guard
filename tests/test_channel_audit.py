import unittest
from scripts.audit_channels import summarize


class ChannelAuditTests(unittest.TestCase):
    def test_losing_clean_screen_is_not_retained_success_or_removed_from_quality_mean(self):
        base={"utterance":"p225_040","content":"p226_002","mos_defended":3.2,"wer_defended":0}
        rows=[{**base,"channel":"none","clean_vc_identity_ok":True,"screened_quality_identity_ok":True},
              {**base,"channel":"mp3-32k","clean_vc_identity_ok":False,"screened_quality_identity_ok":False}]
        result=summarize(rows)
        self.assertEqual(result["mp3-32k"]["initial_joint_successes"],1)
        self.assertEqual(result["mp3-32k"]["retained_initial_joint_successes"],0)
        self.assertEqual(result["mp3-32k"]["initial_successes_losing_clean_screen"],1)
        self.assertIsNone(result["mp3-32k"]["joint_rate"])
        self.assertEqual(result["mp3-32k"]["mean_mos"],3.2)


if __name__=="__main__":unittest.main()
