# MMS Analysis Summary

## Operational Definition
- Bin size: `5` seconds.
- Speed rule: `|Vx| > 300 km/s`.
- Persistence rule: at least `2` consecutive bins.
- Magnetic support rule: `|delta Bz| > 3 nT` in the same bin.
- Candidate ranking is separate from event flagging. `candidate_score` ranks unusual windows; `bbf_operational_flag` is the provisional event rule.

## Direction Convention
- `Vx > 300`: provisional earthward BBF candidate.
- `Vx < -300`: provisional tailward BBF candidate.
- Current strongest candidates in this dataset are tailward because the largest-magnitude Vx values are negative.

## Dataset Check
| Dataset | Time Range | Rows | Median Cadence | Missing Values | Variables | BBF Label |
|---|---|---:|---:|---:|---|---|
| FPI ion velocity | 2026-01-13 02:49:43.661955 to 2026-01-13 05:59:55.779139 | 2537 | 4.500045s | 0 | timestamp + Vx/Vy/Vz in GSE | no BBF label; *_label_fast fields are component names only |
| FGM magnetic field | 2026-01-13 00:00:05.948595659 to 2026-01-14 00:00:07.211378273 | 1382912 | 0.062501s | 0 | timestamp + Bx/By/Bz/Bt in GSE | no BBF label; label_* fields are axis descriptors only |
| SCM wave | 2026-01-13 00:00:01.429405566 to 2026-01-14 00:00:02.723439582 | 2762497 | 0.031250s | 4 | timestamp + raw waveform vectors; derived low/mid/high band powers in observable 0.5-16 Hz range | no BBF label |

## Raw vs Feature Table
- These files are raw or near-raw time-series products, not a prebuilt BBF feature table.
- `timestamp` exists across all products.
- `Vx/Vy/Vz` exists in FPI.
- `Bx/By/Bz` exists in FGM.
- Wave content exists in SCM, but only as survey-rate waveform. There is no BBF label column.

## Whistler Feasibility
- SCM survey sample rate is about `32.00 Hz`, so Nyquist is about `16.00 Hz`.
- FGM total field implies `fce ≈ 28 * Bt[nT]`, which is hundreds of Hz here.
- Therefore the classic whistler band is not observable in this SCM survey product. The script derives observable low/mid/high band powers in `0.5-16 Hz`, but that is not a true whistler-band measurement.

## Candidate Generation
- Operational BBF bins found: `16`
- Earthward candidate bins: `1`
- Tailward candidate bins: `39`

| Time | Vx | Bz | High-Band Power | candidate_score | operational_flag | direction |
|---|---:|---:|---:|---:|---|---|
| 2026-01-13 02:50:15 | -393.108 | 18.750 | 105.787 | 151.161 | True | tailward_candidate |
| 2026-01-13 02:50:05 | -331.838 | 4.767 | 23.470 | 151.095 | True | tailward_candidate |
| 2026-01-13 02:49:55 | -349.249 | -0.040 | 9.658 | 146.548 | True | tailward_candidate |
| 2026-01-13 02:54:05 | -480.143 | 15.502 | 43.000 | 113.331 | True | tailward_candidate |
| 2026-01-13 02:59:05 | -141.372 | 17.422 | 5.888 | 112.728 | False | none |
| 2026-01-13 02:53:55 | -385.631 | 16.540 | 20.728 | 112.609 | True | tailward_candidate |
| 2026-01-13 02:49:50 | -414.572 | 12.300 | 26.715 | 106.737 | True | tailward_candidate |
| 2026-01-13 02:55:05 | -392.794 | 14.158 | 6.158 | 105.284 | True | tailward_candidate |
| 2026-01-13 02:54:00 | -462.522 | 22.152 | 79.226 | 103.886 | True | tailward_candidate |
| 2026-01-13 02:50:00 | -302.318 | -8.056 | 7.904 | 103.472 | True | tailward_candidate |

## Label Separation
- `candidate_score`: unsupervised ranking for candidate generation.
- `bbf_operational_flag`: rule-based provisional event flag.
- `actual_bbf_label`: unavailable in these files and must come from later labeling or an external event list.

Primary candidate center time: `2026-01-13 02:50:15`
Candidate table: `C:\Magnetic\candidate_table.csv`
Candidate plot: `C:\Magnetic\candidate_interval.svg`