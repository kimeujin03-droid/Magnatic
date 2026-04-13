# Early BBF Prediction Pilot Dataset

This folder contains a small pilot dataset for early BBF prediction from the MMS1
2017-07-29 case.

## Purpose

The framing is early prediction:

- Input: observations from `t - 15 min` to `t`
- Target: whether a BBF start occurs from `t` to `t + 5 min`
- Label source: BBF event start times derived from speed criteria
- Leakage control: velocity and event-derived whistler flags are excluded from model inputs

This dataset is intended as a pipeline sanity check before scaling the same
framing to longer THEMIS data.

## Generated Files

- `causal_features.parquet`: resampled causal feature table
- `sequence_index.csv`: metadata for each sequence sample
- `feature_schema.json`: input feature list and excluded leakage columns
- `early_bbf_dataset_summary.json`: compact build summary
- `sequences/early_bbf_*.npz`: LSTM-ready sequence arrays

## Pilot Build Summary

```text
case_id: 2017-07-29_mms1_earthward_bbf
tabular_rows: 3338
sequence_count: 14
positive_sequences: 8
sequence_timesteps: 901
sequence_features: 18
input_window: t-900s to t
prediction_window: t to t+300s
```

Each `.npz` file contains:

- `X`: input sequence with shape `(901, 18)`
- `relative_time_s`: sequence time axis from `-900` to `0`
- `y_bbf_within_5min`: binary target label
- `anchor_time`: prediction time `t`

## Input Features

The sequence feature columns are:

```text
relative_time_s
Bz
abs_Bz
dBz_dt
Bz_delta_60s
Bz_rolling_std_60s
Bz_sign_change_rate_60s
fce_hz
whistler_band_power
wave_total_power_10_4000hz
whistler_ratio
whistler_activity_score
wave_power_delta_60s
wave_power_increase_rate_60s
wave_power_rolling_mean_60s
wave_power_rolling_max_60s
wave_power_rolling_var_60s
whistler_feature_observed
```

The wave inputs are continuous activity statistics. They are not whistler event
flags.

## Excluded From Inputs

The following are intentionally excluded from the model input:

```text
Vx
bbf_event_label
bbf_event_id
inside_bbf_at_time
strict_whistler_segment_label
strict_whistler_event_label
baseline_pass
nearest_whistler
future_whistler_*
```

Velocity is used only to define BBF labels. It is not used as a predictor.

## Rebuild Command

From `C:\Magnetic`:

```powershell
.\venv\Scripts\python.exe build_early_bbf_dataset.py --anchor-stride-seconds 180 --max-sequences 14
```

To create feature sanity-check tables and aggregate-feature baseline inputs:

```powershell
.\venv\Scripts\python.exe evaluate_early_bbf_pilot.py
.\venv\Scripts\python.exe evaluate_early_bbf_pilot.py --drop-inside-bbf
```

The report outputs are written under `reports/`:

- `feature_sanity_check.csv`
- `feature_sanity_check_clean.csv`
- `window_aggregate_features.csv`
- `window_aggregate_features_clean.csv`
- `baseline_results.csv`
- `baseline_results_clean.csv`

The sanity-check tables compare each feature by `target_bbf_within_5min` and
include:

- `mean`
- `std`
- `min`
- `max`
- `missing_rate`

Baseline model execution requires `scikit-learn`; `xgboost` is used only if
installed. Both were installed in the local venv for the current pilot run.

Current baseline results on all 14 pilot samples:

```text
model                 accuracy  balanced_accuracy  f1       roc_auc
majority              0.571429  0.500000           0.727273 0.500000
logistic_regression   0.785714  0.770833           0.823529 0.875000
random_forest         0.785714  0.770833           0.823529 0.947917
xgboost               0.785714  0.770833           0.823529 0.583333
```

Current baseline results on the clean subset
(`inside_bbf_at_anchor == 0`, 11 samples):

```text
model                 accuracy  balanced_accuracy  f1       roc_auc
majority              0.000000  0.000000           0.000000 0.000000
logistic_regression   0.818182  0.816667           0.833333 0.800000
random_forest         0.818182  0.816667           0.833333 0.900000
xgboost               0.727273  0.716667           0.769231 0.833333
```

These scores are not evidence of predictive performance because the dataset is
only 11-14 samples. They are only a check that the feature table, labels, and
baseline evaluation path are wired correctly.

The clean sanity check found no missing values and no fully constant feature.
The largest standardized mean differences in this small pilot were in `fce_hz`,
`wave_power_rolling_mean_60s`, `whistler_activity_score`,
`whistler_band_power`, and `whistler_ratio`.

The default 5-minute anchor stride yields 12 valid samples for this short case
because the script requires a full 15-minute history and full 5-minute prediction
horizon. The command above uses a 3-minute stride to create 14 pilot sequences.

## Important Caveat

This 14-sample dataset is only a sanity check. Because the case is short and BBF
events are close together, three anchors have `inside_bbf_at_anchor = 1` in
`sequence_index.csv`.

For a clean early-prediction training set, filter those rows out:

```text
inside_bbf_at_anchor == 0
```

The clean version should then be scaled to longer data and split by time, not by
random sample.
