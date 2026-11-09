# Huawei ECS campaign

Generated UTC: 2026-11-09T05:11:56.859006+00:00

Complete: True

5 speakers x 1 utterance; 20-speaker enrollment; ECAPA identity screening

| Group | Evaluated | Clean-VC pass | Screened success | MCD | MOS protected | WER protected |
|---|---:|---:|---:|---:|---:|---:|
| main_vg | 5/5 | 4/5 | 2/4 | 35.37 | 1.3972 | 0.6276 |
| main_vg-nop | 5/5 | 4/5 | 2/4 | 38.2212 | 0.7558 | 0.6943 |
| main_emb | 5/5 | 4/5 | 1/4 | 3.2088 | 3.2076 | 0.0571 |
| main_fb | 5/5 | 4/5 | 0/4 | 3.2084 | 3.2428 | 0.0571 |
| main_e2e | 5/5 | 4/5 | 1/4 | 3.168 | 3.209 | 0.0571 |
| eps_.02 | 5/5 | 4/5 | 3/4 | 23.3646 | 2.082 | 0.2971 |
| eps_.05 | 5/5 | 4/5 | 2/4 | 35.353 | 1.394 | 0.8276 |
| eps_.08 | 5/5 | 4/5 | 2/4 | 40.8146 | 1.6038 | 0.96 |
| eps_.10 | 5/5 | 4/5 | 2/4 | 43.1664 | 1.9964 | 1.0267 |

- Not an exact replication of the paper's models, dataset or metrics
- MOS is a third-party predictor; WER uses VCTK transcripts
- Identity success is not joint quality success
- Historical main/eps phases overlapped; resumed stages are sequential; sample times may include GPU contention
