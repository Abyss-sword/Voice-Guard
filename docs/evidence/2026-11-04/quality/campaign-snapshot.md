# Huawei ECS campaign

Generated UTC: 2026-11-04T14:32:23.506991+00:00

Complete: False

5 speakers x 1 utterance; 20-speaker enrollment; ECAPA identity screening

| Group | Evaluated | Clean-VC pass | Screened success | MCD | MOS protected | WER protected |
|---|---:|---:|---:|---:|---:|---:|
| main_vg | 5/5 | 4/5 | 2/4 | 35.37 | 1.3972 | None |
| main_vg-nop | 5/5 | 4/5 | 2/4 | 38.2212 | None | None |
| main_emb | 5/5 | 4/5 | 1/4 | 3.2088 | None | None |
| main_fb | 5/5 | 4/5 | 0/4 | 3.2084 | None | None |
| main_e2e | 5/5 | 4/5 | 1/4 | 3.168 | None | None |
| eps_.02 | 5/5 | 4/5 | 3/4 | 23.3646 | None | None |
| eps_.05 | 5/5 | 4/5 | 2/4 | 35.353 | None | None |
| eps_.08 | 2/5 | 2/2 | 1/2 | 38.107 | None | None |
| eps_.10 | 0/5 | 0/0 | 0/0 | None | None | None |

- Not an exact replication of the paper's models, dataset or metrics
- MOS is a third-party predictor; WER uses VCTK transcripts
- Identity success is not joint quality success
- main and eps run concurrently; per-sample times include contention
