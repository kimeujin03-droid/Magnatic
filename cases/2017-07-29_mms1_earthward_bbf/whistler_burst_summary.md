# Whistler Burst Summary

## Feasibility
- SCM burst sample rate is about `16383.77 Hz`.
- This is sufficient for direct whistler-band analysis in the current interval.
- Operational whistler band was defined dynamically as `0.1-0.5 fce`, with `fce ≈ 28 * Bt[nT]` from FGM.

## BBF Coupling
- Burst files analyzed: `8`
- Whistler segments analyzed: `4264`
- Segments inside provisional BBF bins: `197`
- Strongest whistler segment time: `2017-07-29 15:59:04.471211044`

## Top Whistler Candidates
| Time | Burst File | fce [Hz] | Whistler Band [Hz] | Peak Freq [Hz] | Whistler Power | Ratio | Score | BBF Flag | Direction |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| 2017-07-29 15:59:04.471211044 | mms1_scm_brst_l2_schb_20170729155703_v2.2.0.cdf | 261.5 | 13.1-130.7 | 104.0 | 26000.029 | 0.9948 | 37.845 | False | earthward_candidate |
| 2017-07-29 15:59:42.502719092 | mms1_scm_brst_l2_schb_20170729155703_v2.2.0.cdf | 228.7 | 11.4-114.4 | 68.0 | 14215.547 | 0.9862 | 29.934 | False | earthward_candidate |
| 2017-07-29 16:01:45.255500336 | mms1_scm_brst_l2_schb_20170729160133_v2.2.0.cdf | 225.3 | 11.3-112.6 | 52.0 | 11435.189 | 0.9901 | 28.621 | True | earthward_candidate |
| 2017-07-29 15:59:04.346209391 | mms1_scm_brst_l2_schb_20170729155703_v2.2.0.cdf | 262.8 | 13.1-131.4 | 116.0 | 5710.058 | 0.9825 | 27.530 | False | earthward_candidate |
| 2017-07-29 15:59:34.846429674 | mms1_scm_brst_l2_schb_20170729155703_v2.2.0.cdf | 264.6 | 13.2-132.3 | 76.0 | 6887.150 | 0.9911 | 27.490 | False | earthward_candidate |
| 2017-07-29 15:59:42.627720745 | mms1_scm_brst_l2_schb_20170729155703_v2.2.0.cdf | 229.4 | 11.5-114.7 | 88.0 | 6366.056 | 0.9855 | 26.337 | False | earthward_candidate |
| 2017-07-29 16:02:34.505747090 | mms1_scm_brst_l2_schb_20170729160133_v2.2.0.cdf | 150.0 | 7.5-75.0 | 32.0 | 450.620 | 0.7495 | 25.033 | False | earthward_candidate |
| 2017-07-29 15:59:41.502705863 | mms1_scm_brst_l2_schb_20170729155703_v2.2.0.cdf | 242.6 | 12.1-121.3 | 100.0 | 5753.777 | 0.9629 | 24.293 | False | earthward_candidate |
| 2017-07-29 15:59:34.596426366 | mms1_scm_brst_l2_schb_20170729155703_v2.2.0.cdf | 260.4 | 13.0-130.2 | 76.0 | 3804.956 | 0.9813 | 23.502 | False | earthward_candidate |
| 2017-07-29 15:58:57.439928437 | mms1_scm_brst_l2_schb_20170729155703_v2.2.0.cdf | 271.2 | 13.6-135.6 | 132.0 | 4328.383 | 0.1601 | 21.450 | False | earthward_candidate |

## Interpretation
- `whistler_score` is a candidate-generation score based on dynamic whistler-band power and its fraction of total burst-band power.
- It is not a final event label.
- `bbf_operational_flag` remains the provisional BBF rule from the previous step.

Candidate table: `C:\Magnetic\cases\2017-07-29_mms1_earthward_bbf\whistler_burst_candidates.csv`
Overview plot: `C:\Magnetic\cases\2017-07-29_mms1_earthward_bbf\whistler_burst_overview.svg`