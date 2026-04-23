import os

import numpy as np
import pandas as pd

from analyze_whistler_baseline import CONFIG_PATH as BASELINE_CONFIG_PATH
from analyze_whistler_baseline import SEGMENTS_CSV as BASELINE_SEGMENTS_CSV
from analyze_whistler_baseline import build_segments as build_baseline_segments
from analyze_whistler_baseline import load_config as load_baseline_config
from analyze_whistler_burst import (
    BIN_SECONDS,
    MIN_BACKGROUND_EXCESS,
    MIN_FCE_HZ,
    MIN_CONSECUTIVE_BINS,
    VX_THRESHOLD,
    BZ_VAR_THRESHOLD,
    build_bbf_table,
    build_whistler_table,
)
from path_utils import resolve_case_dir

DEFAULT_CASE_ID = "2017-07-29_mms1_earthward_bbf"
BASE_DIR = resolve_case_dir(default_case=DEFAULT_CASE_ID)
REPORT_PATH = BASE_DIR / "event_coupling_summary.md"
BBF_EVENTS_CSV = BASE_DIR / "bbf_events.csv"
WHISTLER_EVENTS_CSV = BASE_DIR / "whistler_events.csv"
COUPLING_CSV = BASE_DIR / "event_coupling.csv"
THRESHOLD_SWEEP_CSV = BASE_DIR / "threshold_sweep.csv"
MODEL_FEATURES_CSV = BASE_DIR / "whistler_model_features.csv"

WHISTLER_EVENT_GAP_SECONDS = 1.0
LEADING_MIN_SECONDS = -30.0
LEADING_MAX_SECONDS = -5.0
COINCIDENT_WINDOW_SECONDS = 5.0
LAGGING_MIN_SECONDS = 5.0
LAGGING_MAX_SECONDS = 60.0
MC_ITERATIONS = 2000
CONDITIONAL_WINDOWS_SECONDS = [10.0, 30.0, 60.0]

WHISTLER_SEGMENT_SCORE_Q = 0.75
WHISTLER_SEGMENT_RATIO_Q = 0.75
WHISTLER_MIN_DURATION_SECONDS = 0.25
WHISTLER_MIN_BAND_OCCUPANCY = 0.60
WHISTLER_MAX_PEAK_FREQ_CV = 0.35


def load_or_build_baseline_segments() -> pd.DataFrame:
    """Load baseline labels or regenerate them from the baseline config."""
    if os.path.exists(BASELINE_SEGMENTS_CSV):
        return pd.read_csv(BASELINE_SEGMENTS_CSV, parse_dates=["time"])
    if os.path.exists(BASELINE_CONFIG_PATH):
        cfg = load_baseline_config()
        return build_baseline_segments(cfg)
    raise FileNotFoundError(f"Baseline config not found: {BASELINE_CONFIG_PATH}")


def build_model_feature_table(raw_wh: pd.DataFrame, bbf: pd.DataFrame) -> pd.DataFrame:
    """Merge raw whistler features with strict baseline labels and BBF context."""
    baseline_segments = load_or_build_baseline_segments().copy()
    baseline_segments["time"] = pd.to_datetime(baseline_segments["time"])
    baseline_segments = baseline_segments.sort_values("time")

    wh = raw_wh.reset_index().rename(columns={"index": "time"}).sort_values("time")
    wh["time"] = pd.to_datetime(wh["time"])

    tol = pd.Timedelta(seconds=0.2)
    feature_table = pd.merge_asof(
        wh,
        baseline_segments[
            [
                "time",
                "burst_file",
                "fce_hz",
                "peak_freq_hz",
                "freq_fraction_of_fce",
                "ellipticity",
                "planarity",
                "psd_nt2_per_hz",
                "baseline_pass",
            ]
        ].rename(
            columns={
                "fce_hz": "strict_fce_hz",
                "peak_freq_hz": "strict_peak_freq_hz",
                "freq_fraction_of_fce": "strict_freq_fraction_of_fce",
                "psd_nt2_per_hz": "strict_psd_nt2_per_hz",
                "baseline_pass": "strict_whistler_segment_label",
            }
        ),
        on="time",
        by="burst_file",
        direction="nearest",
        tolerance=tol,
    )

    bbf_for_join = (
        bbf[["Vx", "Bz", "bbf_operational_flag", "bbf_direction"]]
        .copy()
        .reset_index()
        .sort_values("time")
        .rename(columns={"bbf_operational_flag": "bbf_event_label"})
    )
    feature_table = pd.merge_asof(
        feature_table,
        bbf_for_join,
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=BIN_SECONDS / 2),
    )
    feature_table["strict_whistler_segment_label"] = feature_table["strict_whistler_segment_label"].fillna(False).astype(bool)
    feature_table["bbf_event_label"] = feature_table["bbf_event_label"].fillna(False).astype(bool)
    feature_table["strict_whistler_event_label"] = False

    strict_mask = feature_table["strict_whistler_segment_label"]
    if strict_mask.any():
        group = (feature_table.loc[strict_mask, "time"].diff() > pd.Timedelta(seconds=WHISTLER_EVENT_GAP_SECONDS)).cumsum()
        for _, event in feature_table.loc[strict_mask].groupby(group):
            start = event["time"].min()
            end = event["time"].max() + pd.Timedelta(seconds=0.125)
            in_event = (feature_table["time"] >= start) & (feature_table["time"] <= end)
            feature_table.loc[in_event, "strict_whistler_event_label"] = True

    feature_table["strict_whistler_segment_label"] = feature_table["strict_whistler_segment_label"].astype(int)
    feature_table["strict_whistler_event_label"] = feature_table["strict_whistler_event_label"].astype(int)
    feature_table["bbf_event_label"] = feature_table["bbf_event_label"].astype(int)
    feature_table["whistler_feature_valid"] = (
        feature_table["fce_valid"]
        & feature_table["whistler_ratio"].notna()
        & feature_table["background_excess"].notna()
        & feature_table["whistler_power_z"].notna()
        & feature_table["whistler_ratio_z"].notna()
        & feature_table["background_excess_z"].notna()
    ).astype(int)
    feature_table = feature_table[
        [
            "time",
            "burst_file",
            "fce_hz",
            "whistler_band_low_hz",
            "whistler_band_high_hz",
            "whistler_peak_freq_hz",
            "whistler_band_power",
            "wave_total_power_10_4000hz",
            "whistler_ratio",
            "background_excess",
            "whistler_power_z",
            "whistler_ratio_z",
            "background_excess_z",
            "whistler_activity_score",
            "whistler_score",
            "fce_valid",
            "whistler_feature_valid",
            "strict_whistler_segment_label",
            "strict_whistler_event_label",
            "strict_freq_fraction_of_fce",
            "ellipticity",
            "planarity",
            "strict_psd_nt2_per_hz",
            "Vx",
            "Bz",
            "bbf_event_label",
            "bbf_direction",
        ]
    ]
    feature_table.to_csv(MODEL_FEATURES_CSV, index=False)
    return feature_table


def build_runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    runs = []
    in_run = False
    start = None
    prev_t = None
    for t, flag in mask.items():
        if bool(flag) and not in_run:
            in_run = True
            start = t
        if bool(flag):
            prev_t = t
        if in_run and not bool(flag):
            runs.append((start, prev_t))
            in_run = False
            start = None
            prev_t = None
    if in_run and start is not None and prev_t is not None:
        runs.append((start, prev_t))
    return runs


def classify_lag(lag_onset_seconds: float) -> str:
    if LEADING_MIN_SECONDS <= lag_onset_seconds <= LEADING_MAX_SECONDS:
        return "leading"
    if -COINCIDENT_WINDOW_SECONDS <= lag_onset_seconds <= COINCIDENT_WINDOW_SECONDS:
        return "coincident"
    if LAGGING_MIN_SECONDS <= lag_onset_seconds <= LAGGING_MAX_SECONDS:
        return "lagging"
    return "uncorrelated"


def empty_bbf_events_frame() -> pd.DataFrame:
    """Return the canonical empty BBF-event table."""
    return pd.DataFrame(
        columns=[
            "bbf_event_id",
            "start_time",
            "end_time",
            "duration_s",
            "peak_time",
            "peak_abs_vx",
            "peak_vx",
            "mean_bz",
            "max_bz_delta",
            "max_bz_delta_time",
            "front_time",
            "front_dbz",
            "flow_accel_onset_time",
            "magnetic_support",
            "direction",
            "num_bins",
            "bbf_operational_event",
        ]
    )


def summarize_bbf_run(idx: int, event: pd.DataFrame) -> dict:
    """Convert one speed-defined run into an event-level BBF summary."""
    start = event.index.min()
    end = event.index.max()
    event_max_bz_delta = float(event["bz_delta"].fillna(0.0).max())
    magnetic_support = event_max_bz_delta > BZ_VAR_THRESHOLD
    peak_time = event["abs_vx"].idxmax()
    max_bz_delta_time = event["bz_delta"].fillna(0.0).idxmax()
    positive_dbz = event["dBz"].clip(lower=0.0).fillna(0.0)
    front_time = positive_dbz.idxmax()
    accel_onset_mask = event["dVx"].fillna(0.0) > 0.0
    flow_accel_onset_time = accel_onset_mask[accel_onset_mask].index.min() if accel_onset_mask.any() else start
    direction_values = event["bbf_direction"].replace("none", pd.NA).dropna()
    return {
        "bbf_event_id": idx,
        "start_time": start,
        "end_time": end + pd.Timedelta(seconds=BIN_SECONDS),
        "duration_s": (end - start).total_seconds() + BIN_SECONDS,
        "peak_time": peak_time,
        "peak_abs_vx": float(event.loc[peak_time, "abs_vx"]),
        "peak_vx": float(event.loc[peak_time, "Vx"]),
        "mean_bz": float(event["Bz"].mean()),
        "max_bz_delta": event_max_bz_delta,
        "max_bz_delta_time": max_bz_delta_time,
        "front_time": front_time,
        "front_dbz": float(positive_dbz.max()),
        "flow_accel_onset_time": flow_accel_onset_time,
        "magnetic_support": magnetic_support,
        "direction": direction_values.mode().iloc[0] if not direction_values.empty else "none",
        "num_bins": int(len(event)),
    }


def reindex_operational_events(events: pd.DataFrame, bbf: pd.DataFrame, event_ids: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only magnetically supported BBF events and remap event ids."""
    filtered = events[events["magnetic_support"]].copy().reset_index(drop=True)
    bbf = bbf.copy()
    bbf["bbf_event_id"] = event_ids
    bbf["bbf_operational_flag"] = bbf["bbf_event_id"].notna()
    if filtered.empty:
        return filtered, bbf
    filtered["bbf_event_id"] = np.arange(1, len(filtered) + 1)
    id_map = dict(zip(events.loc[events["magnetic_support"], "bbf_event_id"], filtered["bbf_event_id"]))
    bbf["bbf_event_id"] = bbf["bbf_event_id"].map(id_map)
    return filtered, bbf


def build_bbf_events(bbf: pd.DataFrame, output_csv: str | None = BBF_EVENTS_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate speed-defined BBF bins into event-level summaries."""
    runs = build_runs(bbf["bbf_speed_run_flag"])
    rows = []
    event_ids = pd.Series(np.nan, index=bbf.index, dtype="float64")
    for idx, (start, end) in enumerate(runs, start=1):
        event = bbf.loc[start:end].copy()
        summary = summarize_bbf_run(idx, event)
        rows.append(summary)
        if summary["magnetic_support"]:
            event_ids.loc[start:end] = idx
    events = pd.DataFrame(rows)
    if events.empty:
        empty = empty_bbf_events_frame()
        bbf = bbf.copy()
        bbf["bbf_event_id"] = event_ids
        bbf["bbf_operational_flag"] = False
        if output_csv is not None:
            empty.to_csv(output_csv, index=False)
        return empty, bbf
    events["bbf_operational_event"] = events["magnetic_support"]
    filtered, bbf = reindex_operational_events(events, bbf, event_ids)
    if output_csv is not None:
        filtered.to_csv(output_csv, index=False)
    return filtered, bbf


def build_whistler_events(wh: pd.DataFrame, min_background_excess: float = MIN_BACKGROUND_EXCESS) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Merge segment candidates into event-level whistler detections."""
    score_threshold = float(wh["whistler_score"].quantile(WHISTLER_SEGMENT_SCORE_Q))
    ratio_threshold = float(wh["whistler_ratio"].quantile(WHISTLER_SEGMENT_RATIO_Q))
    background_threshold = float(wh["background_excess"].quantile(WHISTLER_SEGMENT_SCORE_Q))

    wh = wh.copy()
    wh["segment_candidate"] = (
        (wh["whistler_score"] >= score_threshold)
        & (wh["fce_hz"] >= MIN_FCE_HZ)
        & (wh["background_excess"] >= max(background_threshold, min_background_excess))
        & wh["whistler_peak_freq_hz"].notna()
    )
    candidate = wh[wh["segment_candidate"]].copy().sort_index()
    segment_step_seconds = float(candidate.index.to_series().diff().dropna().dt.total_seconds().median()) if len(candidate) > 1 else 0.125

    if candidate.empty:
        out = pd.DataFrame(
            columns=[
                "whistler_event_id",
                "start_time",
                "end_time",
                "duration_s",
                "peak_time",
                "peak_score",
                "peak_freq_hz",
                "mean_fce_hz",
                "mean_ratio",
                "mean_background_excess",
                "peak_freq_cv",
                "band_occupancy",
                "num_segments",
            ]
        )
        out.to_csv(WHISTLER_EVENTS_CSV, index=False)
        thresholds = {
            "score_threshold": score_threshold,
            "ratio_threshold": ratio_threshold,
            "background_threshold": background_threshold,
            "background_floor": min_background_excess,
            "segment_step_seconds": segment_step_seconds,
        }
        return out, wh, thresholds

    group = (candidate.index.to_series().diff() > pd.Timedelta(seconds=WHISTLER_EVENT_GAP_SECONDS)).cumsum()
    rows = []
    event_id_series = pd.Series(np.nan, index=wh.index, dtype="float64")
    kept_segments = pd.Series(False, index=wh.index)
    kept_id = 1
    for _, event in candidate.groupby(group):
        start = event.index.min()
        end = event.index.max()
        duration_s = (end - start).total_seconds() + segment_step_seconds
        full_window = wh.loc[start:end].copy()
        occupancy = float(full_window["segment_candidate"].mean()) if len(full_window) else 0.0
        peak_freq_mean = float(event["whistler_peak_freq_hz"].mean())
        peak_freq_std = float(event["whistler_peak_freq_hz"].std(ddof=0)) if len(event) > 1 else 0.0
        peak_freq_cv = peak_freq_std / peak_freq_mean if peak_freq_mean > 0 else np.nan
        keep = (
            duration_s >= WHISTLER_MIN_DURATION_SECONDS
            and occupancy >= WHISTLER_MIN_BAND_OCCUPANCY
            and (peak_freq_cv <= WHISTLER_MAX_PEAK_FREQ_CV if peak_freq_cv == peak_freq_cv else False)
        )
        if not keep:
            continue
        peak_time = event["whistler_score"].idxmax()
        rows.append(
            {
                "whistler_event_id": kept_id,
                "start_time": start,
                "end_time": end,
                "duration_s": duration_s,
                "peak_time": peak_time,
                "peak_score": float(event.loc[peak_time, "whistler_score"]),
                "peak_freq_hz": float(event.loc[peak_time, "whistler_peak_freq_hz"]),
                "mean_fce_hz": float(event["fce_hz"].mean()),
                "mean_ratio": float(event["whistler_ratio"].mean()),
                "mean_background_excess": float(event["background_excess"].mean()),
                "peak_freq_cv": peak_freq_cv,
                "band_occupancy": occupancy,
                "num_segments": int(len(event)),
            }
        )
        event_id_series.loc[start:end] = kept_id
        kept_segments.loc[start:end] = full_window["segment_candidate"]
        kept_id += 1

    events = pd.DataFrame(rows)
    events.to_csv(WHISTLER_EVENTS_CSV, index=False)
    wh["whistler_event_id"] = event_id_series
    wh["whistler_event_segment"] = kept_segments
    thresholds = {
        "score_threshold": score_threshold,
        "ratio_threshold": ratio_threshold,
        "background_threshold": background_threshold,
        "background_floor": min_background_excess,
        "segment_step_seconds": segment_step_seconds,
    }
    return events, wh, thresholds


def nearest_whistler_lag(bbf_events: pd.DataFrame, whistler_events: pd.DataFrame) -> pd.DataFrame:
    """Attach the nearest whistler event to each BBF event and compute lags."""
    rows = []
    for _, bbf in bbf_events.iterrows():
        bbf_start = pd.Timestamp(bbf["start_time"])
        bbf_end = pd.Timestamp(bbf["end_time"])
        bbf_peak = pd.Timestamp(bbf["peak_time"])
        bbf_bz_peak = pd.Timestamp(bbf["max_bz_delta_time"])
        bbf_front = pd.Timestamp(bbf["front_time"])
        bbf_accel_onset = pd.Timestamp(bbf["flow_accel_onset_time"])
        duration_s = max((bbf_end - bbf_start).total_seconds(), 1e-9)
        if whistler_events.empty:
            rows.append(
                {
                    **bbf.to_dict(),
                    "nearest_whistler_event_id": np.nan,
                    "nearest_whistler_start_time": pd.NaT,
                    "nearest_whistler_peak_time": pd.NaT,
                    "lag_onset_seconds": np.nan,
                    "lag_peak_seconds": np.nan,
                    "lag_to_vx_peak_seconds": np.nan,
                    "lag_to_bz_peak_seconds": np.nan,
                    "lag_to_front_seconds": np.nan,
                    "lag_to_flow_accel_seconds": np.nan,
                    "normalized_event_phase": np.nan,
                    "phase_region": "no_whistler_event",
                    "lag_class": "no_whistler_event",
                    "front_lag_class": "no_whistler_event",
                    "overlap": False,
                }
            )
            continue
        onset_times = pd.to_datetime(whistler_events["start_time"])
        onset_delta = (onset_times - bbf_start).dt.total_seconds()
        nearest_idx = onset_delta.abs().idxmin()
        nearest = whistler_events.loc[nearest_idx]
        whistler_onset = pd.Timestamp(nearest["start_time"])
        lag_onset = float(onset_delta.loc[nearest_idx])
        lag_peak = float((pd.Timestamp(nearest["peak_time"]) - bbf_start).total_seconds())
        lag_to_vx_peak = float((whistler_onset - bbf_peak).total_seconds())
        lag_to_bz_peak = float((whistler_onset - bbf_bz_peak).total_seconds())
        lag_to_front = float((whistler_onset - bbf_front).total_seconds())
        lag_to_flow_accel = float((whistler_onset - bbf_accel_onset).total_seconds())
        phase = float((whistler_onset - bbf_start).total_seconds() / duration_s)
        if phase < 0.0:
            phase_region = "pre_event"
        elif phase < (1.0 / 3.0):
            phase_region = "early"
        elif phase < (2.0 / 3.0):
            phase_region = "mid"
        elif phase <= 1.0:
            phase_region = "late"
        else:
            phase_region = "post_event"
        overlap = not (
            pd.Timestamp(nearest["end_time"]) < pd.Timestamp(bbf["start_time"])
            or pd.Timestamp(nearest["start_time"]) > pd.Timestamp(bbf["end_time"])
        )
        rows.append(
            {
                **bbf.to_dict(),
                "nearest_whistler_event_id": int(nearest["whistler_event_id"]),
                "nearest_whistler_start_time": nearest["start_time"],
                "nearest_whistler_peak_time": nearest["peak_time"],
                "lag_onset_seconds": lag_onset,
                "lag_peak_seconds": lag_peak,
                "lag_to_vx_peak_seconds": lag_to_vx_peak,
                "lag_to_bz_peak_seconds": lag_to_bz_peak,
                "lag_to_front_seconds": lag_to_front,
                "lag_to_flow_accel_seconds": lag_to_flow_accel,
                "normalized_event_phase": phase,
                "phase_region": phase_region,
                "lag_class": classify_lag(lag_onset),
                "front_lag_class": classify_lag(lag_to_front),
                "overlap": overlap,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(COUPLING_CSV, index=False)
    return df


def monte_carlo_overlap(common_index: pd.DatetimeIndex, bbf_flags: pd.Series, whistler_events: pd.DataFrame, actual_overlap: int) -> dict:
    if whistler_events.empty or len(common_index) == 0:
        return {"expected_overlap": np.nan, "p_value_ge": np.nan}
    rng = np.random.default_rng(42)
    start_ns = int(common_index.min().value)
    end_ns = int(common_index.max().value)
    bin_ns = int(pd.Timedelta(seconds=BIN_SECONDS).value)
    durations_ns = (
        pd.to_timedelta(pd.to_datetime(whistler_events["end_time"]) - pd.to_datetime(whistler_events["start_time"]))
        .astype("int64")
        .to_numpy()
    )
    index_ns = common_index.view("int64")
    overlaps = []
    for _ in range(MC_ITERATIONS):
        sim = np.zeros(len(common_index), dtype=bool)
        for duration_ns in durations_ns:
            max_start = max(start_ns, end_ns - int(duration_ns))
            sim_start = start_ns if max_start <= start_ns else int(rng.integers(start_ns, max_start + 1))
            sim_end = sim_start + int(duration_ns) + bin_ns
            sim |= (index_ns >= sim_start) & (index_ns <= sim_end)
        overlaps.append(int((bbf_flags.to_numpy(dtype=bool) & sim).sum()))
    overlaps_arr = np.asarray(overlaps, dtype=float)
    return {
        "expected_overlap": float(np.mean(overlaps_arr)),
        "p_value_ge": float((np.count_nonzero(overlaps_arr >= actual_overlap) + 1) / (len(overlaps_arr) + 1)),
    }


def compute_conditional_probabilities(bbf_events: pd.DataFrame, whistler_events: pd.DataFrame, common_duration_s: float) -> list[dict]:
    onsets = pd.to_datetime(whistler_events["start_time"]) if not whistler_events.empty else pd.DatetimeIndex([])
    overall_rate = len(whistler_events) / common_duration_s if common_duration_s > 0 else np.nan
    rows = []
    for window_s in CONDITIONAL_WINDOWS_SECONDS:
        hits = 0
        for _, bbf in bbf_events.iterrows():
            start = pd.Timestamp(bbf["start_time"])
            if ((onsets >= start) & (onsets <= start + pd.Timedelta(seconds=window_s))).any():
                hits += 1
        observed = hits / len(bbf_events) if len(bbf_events) else np.nan
        expected = 1.0 - np.exp(-overall_rate * window_s) if overall_rate == overall_rate else np.nan
        rows.append(
            {
                "window_s": window_s,
                "observed_probability": observed,
                "expected_probability": expected,
                "enrichment": observed / expected if expected and expected > 0 else np.nan,
            }
        )
    return rows


def compute_baseline(bbf: pd.DataFrame, wh: pd.DataFrame, whistler_events: pd.DataFrame) -> dict:
    time_start = max(bbf.index.min(), wh.index.min())
    time_end = min(bbf.index.max(), wh.index.max())
    bbf_overlap = bbf.loc[time_start:time_end].copy()
    wh_overlap = wh.loc[time_start:time_end].copy()

    common = pd.concat(
        [
            bbf_overlap["bbf_operational_flag"].rename("bbf_flag"),
            wh_overlap["whistler_event_segment"].resample(f"{BIN_SECONDS}s").max().fillna(False).rename("whistler_flag"),
        ],
        axis=1,
    ).fillna(False)
    common["bbf_flag"] = common["bbf_flag"].astype(bool)
    common["whistler_flag"] = common["whistler_flag"].astype(bool)
    bbf_true_bins = int(common["bbf_flag"].sum())
    wh_true_bins = int(common["whistler_flag"].sum())
    bbf_fraction = bbf_true_bins / len(common) if len(common) else np.nan
    wh_fraction = wh_true_bins / len(common) if len(common) else np.nan
    actual_overlap = int((common["bbf_flag"] & common["whistler_flag"]).sum())
    expected_overlap = float(bbf_fraction * wh_fraction * len(common)) if len(common) else np.nan
    mc_stats = monte_carlo_overlap(common.index, common["bbf_flag"], whistler_events, actual_overlap)
    mc_expected = mc_stats["expected_overlap"]
    conditional = compute_conditional_probabilities(build_bbf_events(bbf_overlap, output_csv=None)[0], whistler_events, (time_end - time_start).total_seconds())
    return {
        "time_start": time_start,
        "time_end": time_end,
        "total_common_bins": int(len(common)),
        "bbf_true_bins": bbf_true_bins,
        "bbf_fraction": float(bbf_fraction),
        "whistler_true_bins": wh_true_bins,
        "whistler_fraction": float(wh_fraction),
        "actual_overlap_bins": actual_overlap,
        "expected_random_overlap_bins": expected_overlap,
        "expected_mc_overlap_bins": mc_expected,
        "permutation_p_value": mc_stats["p_value_ge"],
        "overlap_enrichment": float(actual_overlap / expected_overlap) if expected_overlap and expected_overlap > 0 else np.nan,
        "overlap_enrichment_mc": float(actual_overlap / mc_expected) if mc_expected and mc_expected > 0 else np.nan,
        "conditional": conditional,
    }


def summarize_sweep_point(raw_wh: pd.DataFrame, bbf_events: pd.DataFrame, bbf: pd.DataFrame, min_background_excess: float) -> dict:
    whistler_events, wh, thresholds = build_whistler_events(raw_wh, min_background_excess=min_background_excess)
    coupling = nearest_whistler_lag(bbf_events, whistler_events)
    baseline = compute_baseline(bbf, wh, whistler_events)
    return {
        "background_excess_floor": float(min_background_excess),
        "effective_background_threshold": float(max(thresholds["background_threshold"], min_background_excess)),
        "whistler_event_count": int(len(whistler_events)),
        "overlap_enrichment_mc": float(baseline["overlap_enrichment_mc"]) if not np.isnan(baseline["overlap_enrichment_mc"]) else np.nan,
        "permutation_p_value": float(baseline["permutation_p_value"]) if not np.isnan(baseline["permutation_p_value"]) else np.nan,
        "median_phase": float(coupling["normalized_event_phase"].median()) if not coupling.empty else np.nan,
        "mean_lag_to_vx_peak_s": float(coupling["lag_to_vx_peak_seconds"].mean()) if not coupling.empty else np.nan,
        "coincident_count": int((coupling["lag_class"] == "coincident").sum()) if not coupling.empty else 0,
        "front_coincident_count": int((coupling["front_lag_class"] == "coincident").sum()) if not coupling.empty else 0,
    }


def write_report(lines: list[str]) -> None:
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def build_rules_section(wh_thresholds: dict) -> list[str]:
    return [
        "# Event Coupling Summary",
        "",
        "## Analysis Rules",
        f"- BBF speed rule: `Vx > +{VX_THRESHOLD:.0f} km/s`, at least `{MIN_CONSECUTIVE_BINS}` consecutive `{BIN_SECONDS}s` bins.",
        f"- Tailward fast flow is tracked separately as `Vx < -{VX_THRESHOLD:.0f} km/s` and is excluded from the BBF set.",
        f"- BBF magnetic support is evaluated at event level: `max |delta Bz| > {BZ_VAR_THRESHOLD:.0f} nT` somewhere inside the speed-defined event.",
        f"- Whistler seed segment rule: `whistler_score >= q{int(WHISTLER_SEGMENT_SCORE_Q*100)}` (`{wh_thresholds['score_threshold']:.3f}`), `fce >= {MIN_FCE_HZ:.0f} Hz`, and `background_excess >= max(q{int(WHISTLER_SEGMENT_SCORE_Q*100)}, {MIN_BACKGROUND_EXCESS:.1f})` (`{wh_thresholds['background_threshold']:.3f}`).",
        f"- Whistler event filters: minimum duration `{WHISTLER_MIN_DURATION_SECONDS:.2f}s`, band occupancy `>= {WHISTLER_MIN_BAND_OCCUPANCY:.2f}`, peak-frequency CV `<= {WHISTLER_MAX_PEAK_FREQ_CV:.2f}`, merge gaps up to `{WHISTLER_EVENT_GAP_SECONDS:.1f}s`.",
        "- Model features are exported separately: continuous features are preserved without thresholding, and strict Santolik-style labels are attached as boolean targets.",
        f"- Leading: `{LEADING_MIN_SECONDS:.0f}s` to `{LEADING_MAX_SECONDS:.0f}s` before BBF start.",
        f"- Coincident: within `+/-{COINCIDENT_WINDOW_SECONDS:.0f}s` of BBF start.",
        f"- Lagging: `+{LAGGING_MIN_SECONDS:.0f}s` to `+{LAGGING_MAX_SECONDS:.0f}s` after BBF start.",
        "- Other offsets are `uncorrelated`.",
        "",
    ]


def build_event_counts_section(bbf_events: pd.DataFrame, whistler_events: pd.DataFrame, model_features: pd.DataFrame, raw_bbf: pd.DataFrame) -> list[str]:
    direction_counts = bbf_events["direction"].value_counts().to_dict() if not bbf_events.empty else {}
    tailward_run_count = len(build_runs(raw_bbf["tailward_speed_run_flag"]))
    return [
        "## Event Counts",
        f"- Total earthward BBF events: `{len(bbf_events)}`",
        f"- Total whistler events: `{len(whistler_events)}`",
        f"- Model-feature rows: `{len(model_features)}`",
        f"- Strict whistler segment labels: `{int(model_features['strict_whistler_segment_label'].sum())}`",
        f"- Strict whistler event labels: `{int(model_features['strict_whistler_event_label'].sum())}`",
        f"- Tailward fast-flow runs excluded from BBF set: `{tailward_run_count}`",
        f"- Tailward events surviving BBF filter: `{direction_counts.get('tailward_fastflow_candidate', 0)}`",
        f"- Earthward BBF events: `{direction_counts.get('earthward_candidate', 0)}`",
        "",
    ]


def write_empty_coupling_csv() -> None:
    pd.DataFrame(
        columns=[
            "bbf_event_id",
            "start_time",
            "end_time",
            "direction",
            "peak_vx",
            "nearest_whistler_start_time",
            "lag_onset_seconds",
            "lag_peak_seconds",
            "lag_class",
            "overlap",
        ]
    ).to_csv(COUPLING_CSV, index=False)


def write_no_bbf_report(lines: list[str]) -> None:
    lines.extend(
        [
            "## Result",
            "- No earthward BBF events are present in this burst interval under the current BBF definition.",
            "- This interval contains tailward fast flow and whistler activity, but it does not provide BBF-whistler coupling evidence.",
            "",
            f"BBF events CSV: `{BBF_EVENTS_CSV}`",
            f"Whistler events CSV: `{WHISTLER_EVENTS_CSV}`",
            f"Coupling CSV: `{COUPLING_CSV}`",
            f"Model features CSV: `{MODEL_FEATURES_CSV}`",
        ]
    )
    write_report(lines)


def build_lag_statistics_section(coupling: pd.DataFrame) -> list[str]:
    lag_counts = coupling["lag_class"].value_counts().to_dict() if not coupling.empty else {}
    front_lag_counts = coupling["front_lag_class"].value_counts().to_dict() if not coupling.empty else {}
    mean_lag = coupling["lag_onset_seconds"].mean() if not coupling["lag_onset_seconds"].dropna().empty else np.nan
    median_lag = coupling["lag_onset_seconds"].median() if not coupling["lag_onset_seconds"].dropna().empty else np.nan
    mean_front_lag = coupling["lag_to_front_seconds"].mean() if not coupling["lag_to_front_seconds"].dropna().empty else np.nan
    median_front_lag = coupling["lag_to_front_seconds"].median() if not coupling["lag_to_front_seconds"].dropna().empty else np.nan
    mean_vx_peak_lag = coupling["lag_to_vx_peak_seconds"].mean() if not coupling["lag_to_vx_peak_seconds"].dropna().empty else np.nan
    mean_bz_peak_lag = coupling["lag_to_bz_peak_seconds"].mean() if not coupling["lag_to_bz_peak_seconds"].dropna().empty else np.nan
    return [
        "## Lag Statistics",
        f"- Mean onset lag: `{mean_lag:.3f}s`" if not np.isnan(mean_lag) else "- Mean onset lag: `NaN`",
        f"- Median onset lag: `{median_lag:.3f}s`" if not np.isnan(median_lag) else "- Median onset lag: `NaN`",
        f"- Mean lag relative to BBF front time: `{mean_front_lag:.3f}s`" if not np.isnan(mean_front_lag) else "- Mean lag relative to BBF front time: `NaN`",
        f"- Median lag relative to BBF front time: `{median_front_lag:.3f}s`" if not np.isnan(median_front_lag) else "- Median lag relative to BBF front time: `NaN`",
        f"- Mean lag relative to BBF peak Vx: `{mean_vx_peak_lag:.3f}s`" if not np.isnan(mean_vx_peak_lag) else "- Mean lag relative to BBF peak Vx: `NaN`",
        f"- Mean lag relative to max |dBz| time: `{mean_bz_peak_lag:.3f}s`" if not np.isnan(mean_bz_peak_lag) else "- Mean lag relative to max |dBz| time: `NaN`",
        f"- Leading: `{lag_counts.get('leading', 0)}`",
        f"- Coincident: `{lag_counts.get('coincident', 0)}`",
        f"- Lagging: `{lag_counts.get('lagging', 0)}`",
        f"- Uncorrelated: `{lag_counts.get('uncorrelated', 0)}`",
        f"- Front-referenced coincident: `{front_lag_counts.get('coincident', 0)}`",
        "",
    ]


def build_phase_section(coupling: pd.DataFrame) -> list[str]:
    phase_median = coupling["normalized_event_phase"].median() if not coupling["normalized_event_phase"].dropna().empty else np.nan
    phase_counts = coupling["phase_region"].value_counts().to_dict() if not coupling.empty else {}
    return [
        "## Event Phase",
        f"- Median normalized phase `phi`: `{phase_median:.3f}`" if not np.isnan(phase_median) else "- Median normalized phase `phi`: `NaN`",
        f"- Pre-event (`phi < 0`): `{phase_counts.get('pre_event', 0)}`",
        f"- Early (`0 <= phi < 1/3`): `{phase_counts.get('early', 0)}`",
        f"- Mid (`1/3 <= phi < 2/3`): `{phase_counts.get('mid', 0)}`",
        f"- Late (`2/3 <= phi <= 1`): `{phase_counts.get('late', 0)}`",
        f"- Post-event (`phi > 1`): `{phase_counts.get('post_event', 0)}`",
        "",
    ]


def build_overlap_section(baseline: dict) -> list[str]:
    lines = [
        "## Overlap And Baseline",
        f"- Analysis interval: `{baseline['time_start']}` to `{baseline['time_end']}` ({baseline['total_common_bins']} bins)",
        f"- BBF occupancy: `{baseline['bbf_true_bins']}/{baseline['total_common_bins']}` (`{baseline['bbf_fraction']:.4f}`)",
        f"- Whistler occupancy: `{baseline['whistler_true_bins']}/{baseline['total_common_bins']}` (`{baseline['whistler_fraction']:.4f}`)",
        f"- Actual overlap bins: `{baseline['actual_overlap_bins']}`",
        f"- Expected random overlap: `{baseline['expected_random_overlap_bins']:.3f}`",
        f"- Monte Carlo expected overlap: `{baseline['expected_mc_overlap_bins']:.3f}`" if not np.isnan(baseline["expected_mc_overlap_bins"]) else "- Monte Carlo expected overlap: `NaN`",
        f"- Permutation p-value: `{baseline['permutation_p_value']:.4f}`" if not np.isnan(baseline["permutation_p_value"]) else "- Permutation p-value: `NaN`",
        f"- Overlap enrichment: `{baseline['overlap_enrichment']:.3f}x` / `{baseline['overlap_enrichment_mc']:.3f}x` (Monte Carlo)"
        if not np.isnan(baseline["overlap_enrichment"]) and not np.isnan(baseline["overlap_enrichment_mc"])
        else "- Overlap enrichment: `NaN`",
        "",
        "## Conditional Probability",
    ]
    for row in baseline["conditional"]:
        if np.isnan(row["observed_probability"]) or np.isnan(row["expected_probability"]):
            lines.append(f"- Within {int(row['window_s'])} s: `NaN`")
        else:
            lines.append(
                f"- Within {int(row['window_s'])} s: observed `{row['observed_probability']:.3f}` vs baseline `{row['expected_probability']:.3f}` -> `{row['enrichment']:.3f}x`"
            )
    lines.append("")
    return lines


def build_bbf_to_whistler_table_section(coupling: pd.DataFrame) -> list[str]:
    lines = [
        "## BBF To Nearest Whistler",
        "| BBF Event | Start | End | Direction | Peak Vx | Max |dBz| | Nearest Whistler Onset | Start Lag [s] | Front Lag [s] | Vx Peak Lag [s] | |dBz| Peak Lag [s] | Phi | Phase | Class | Front Class | Overlap |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for _, row in coupling.iterrows():
        lines.append(
            f"| {int(row['bbf_event_id'])} | {row['start_time']} | {row['end_time']} | {str(row['direction']).replace('_candidate','')} | "
            f"{row['peak_vx']:.1f} | {row['max_bz_delta']:.2f} | {row['nearest_whistler_start_time']} | "
            f"{row['lag_onset_seconds']:.2f} | {row['lag_to_front_seconds']:.2f} | {row['lag_to_vx_peak_seconds']:.2f} | {row['lag_to_bz_peak_seconds']:.2f} | "
            f"{row['normalized_event_phase']:.3f} | {row['phase_region']} | {row['lag_class']} | {row['front_lag_class']} | {bool(row['overlap'])} |"
        )
    lines.append("")
    return lines


def build_output_section() -> list[str]:
    return [
        "## Threshold Sweep",
        f"- Background-excess sweep CSV: `{THRESHOLD_SWEEP_CSV}`",
        "",
        "## Model-Ready Features",
        "- Continuous predictors are kept as raw/continuous features: `whistler_ratio`, `background_excess`, `whistler_power_z`, `whistler_ratio_z`, `background_excess_z`, and `whistler_activity_score`.",
        "- Strict labels are attached separately using the Santolik-style rule: `strict_whistler_segment_label` and `strict_whistler_event_label`.",
        "",
        f"BBF events CSV: `{BBF_EVENTS_CSV}`",
        f"Whistler events CSV: `{WHISTLER_EVENTS_CSV}`",
        f"Coupling CSV: `{COUPLING_CSV}`",
        f"Model features CSV: `{MODEL_FEATURES_CSV}`",
    ]


def main() -> None:
    raw_bbf = build_bbf_table()
    bbf_events, bbf = build_bbf_events(raw_bbf)
    raw_wh = build_whistler_table()
    model_features = build_model_feature_table(raw_wh, bbf)
    whistler_events, wh, wh_thresholds = build_whistler_events(raw_wh)
    sweep_rows = []
    for floor in [2.0, 2.5, 3.0, 3.5, 4.0]:
        sweep_rows.append(summarize_sweep_point(raw_wh, bbf_events, bbf, min_background_excess=floor))
    pd.DataFrame(sweep_rows).to_csv(THRESHOLD_SWEEP_CSV, index=False)
    lines = build_rules_section(wh_thresholds) + build_event_counts_section(bbf_events, whistler_events, model_features, raw_bbf)

    if bbf_events.empty:
        write_empty_coupling_csv()
        write_no_bbf_report(lines)
        print(f"report: {REPORT_PATH}")
        print(f"bbf events: {BBF_EVENTS_CSV}")
        print(f"whistler events: {WHISTLER_EVENTS_CSV}")
        print(f"coupling: {COUPLING_CSV}")
        return

    coupling = nearest_whistler_lag(bbf_events, whistler_events)
    baseline = compute_baseline(bbf, wh, whistler_events)
    lines.extend(build_lag_statistics_section(coupling))
    lines.extend(build_phase_section(coupling))
    lines.extend(build_overlap_section(baseline))
    lines.extend(build_bbf_to_whistler_table_section(coupling))
    lines.extend(build_output_section())
    write_report(lines)

    print(f"report: {REPORT_PATH}")
    print(f"bbf events: {BBF_EVENTS_CSV}")
    print(f"whistler events: {WHISTLER_EVENTS_CSV}")
    print(f"coupling: {COUPLING_CSV}")
    print(f"model features: {MODEL_FEATURES_CSV}")


if __name__ == "__main__":
    main()
