# Quality experiment: PCM16 export verification

Small-sample engineering gates; not paper-equivalent scores.

| Utterance | Baseline MOS | Refined MOS | Baseline WER | Refined WER | Converted identity | Screened joint pass |
|---|---:|---:|---:|---:|---|---|
| p225_040 | 1.049 | 3.275 | 40.0% | 0.0% | p246 | True |
| p226_039 | 0.519 | 3.165 | 100.0% | 0.0% | p226 | False |
| p227_033 | 2.313 | 3.153 | 57.1% | 28.6% | p227 | False |
| p228_035 | 0.673 | 3.409 | 66.7% | 0.0% | p226 | True |
| p229_032 | 2.432 | 2.958 | 50.0% | 0.0% | p231 | False |

Means across all five fixed utterances: {'baseline_mos': 1.3972, 'refined_mos': 3.191929054260254, 'baseline_wer': 0.6276200000000001, 'refined_wer': 0.05714285714285714}

Development: {'total': 2, 'screened': 2, 'joint_passes': 1}; fixed-parameter validation: {'total': 3, 'screened': 2, 'joint_passes': 1}

p229 is excluded because clean conversion already fails identity. All failures remain in quality averages.
Full-context follow-up uses a different input protocol; see its separate results and controls.
