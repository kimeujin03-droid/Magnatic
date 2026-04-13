import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CASE_DIR = Path(r"C:\Magnetic\cases\2017-07-29_mms1_earthward_bbf")

NUMERIC_FEATURES = [
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
    "strict_freq_fraction_of_fce",
    "ellipticity",
    "planarity",
    "strict_psd_nt2_per_hz",
    "Vx",
    "Bz",
]

BOOLEAN_FEATURES = [
    "fce_valid",
    "whistler_feature_valid",
    "bbf_event_label",
]

LABEL_COLUMNS = [
    "strict_whistler_segment_label",
    "strict_whistler_event_label",
    "future_whistler_within_10s",
    "future_whistler_within_30s",
    "future_whistler_within_60s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Random Forest tabular features and LSTM sequence arrays from case-level MMS outputs."
    )
    parser.add_argument(
        "--case-dir",
        default=os.environ.get("MMS_CASE_DIR", str(DEFAULT_CASE_DIR)),
        help="Case directory containing whistler_model_features.csv, bbf_events.csv, and whistler_events.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <case-dir>/ml_dataset.",
    )
    parser.add_argument(
        "--resample-seconds",
        type=float,
        default=1.0,
        help="Regular time step for tabular and sequence datasets.",
    )
    parser.add_argument(
        "--pre-seconds",
        type=float,
        default=120.0,
        help="LSTM sequence start offset relative to BBF start.",
    )
    parser.add_argument(
        "--post-seconds",
        type=float,
        default=300.0,
        help="LSTM sequence stop offset relative to BBF start.",
    )
    parser.add_argument(
        "--regular-anchor-stride-seconds",
        type=float,
        default=300.0,
        help="Stride for additional regular time-anchor sequences used to create negative examples.",
    )
    parser.add_argument(
        "--skip-regular-anchors",
        action="store_true",
        help="Only write BBF-event anchored LSTM sequences.",
    )
    return parser.parse_args()


def load_case_config(case_dir: Path) -> dict:
    path = case_dir / "case_config.json"
    if not path.exists():
        return {"case_id": case_dir.name}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_inputs(case_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_path = case_dir / "whistler_model_features.csv"
    bbf_path = case_dir / "bbf_events.csv"
    whistler_path = case_dir / "whistler_events.csv"

    missing = [str(path) for path in [feature_path, bbf_path, whistler_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    features = pd.read_csv(feature_path, parse_dates=["time"]).sort_values("time")
    bbf_events = pd.read_csv(
        bbf_path,
        parse_dates=[
            "start_time",
            "end_time",
            "peak_time",
            "max_bz_delta_time",
            "front_time",
            "flow_accel_onset_time",
        ],
    )
    whistler_events = pd.read_csv(
        whistler_path,
        parse_dates=["start_time", "end_time", "peak_time"],
    )
    return features, bbf_events, whistler_events


def coerce_feature_columns(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    for col in NUMERIC_FEATURES:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BOOLEAN_FEATURES + ["strict_whistler_segment_label", "strict_whistler_event_label"]:
        if col not in df:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool).astype(np.float32)

    df["abs_Bz"] = df["Bz"].abs()
    df["abs_Vx"] = df["Vx"].abs()
    df["whistler_to_fce"] = df["whistler_peak_freq_hz"] / df["fce_hz"].replace(0.0, np.nan)
    return df


def build_regular_grid(features: pd.DataFrame, resample_seconds: float) -> pd.DataFrame:
    feature_cols = NUMERIC_FEATURES + ["abs_Bz", "abs_Vx", "whistler_to_fce"] + BOOLEAN_FEATURES
    label_cols = ["strict_whistler_segment_label", "strict_whistler_event_label"]

    df = features.set_index("time").sort_index()
    numeric = df[feature_cols].resample(f"{resample_seconds}s").mean()
    labels = df[label_cols].resample(f"{resample_seconds}s").max()

    out = pd.concat([numeric, labels], axis=1)
    out["feature_observed"] = df["whistler_feature_valid"].resample(f"{resample_seconds}s").max().fillna(0.0)
    out[feature_cols] = out[feature_cols].ffill().bfill()
    out = out.fillna(0.0).reset_index()
    return out


def add_future_whistler_labels(grid: pd.DataFrame, whistler_events: pd.DataFrame, windows: list[float]) -> pd.DataFrame:
    out = grid.copy()
    times_ns = out["time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    starts_ns = (
        whistler_events["start_time"].dropna().sort_values().to_numpy(dtype="datetime64[ns]").astype("int64")
    )

    for window in windows:
        col = f"future_whistler_within_{int(window)}s"
        if len(starts_ns) == 0:
            out[col] = 0
            continue
        pos = np.searchsorted(starts_ns, times_ns, side="left")
        has_next = pos < len(starts_ns)
        delta = np.full(len(times_ns), np.inf)
        delta[has_next] = (starts_ns[pos[has_next]] - times_ns[has_next]) / 1_000_000_000.0
        out[col] = ((delta >= 0.0) & (delta <= window)).astype(np.int8)
    return out


def attach_event_context(grid: pd.DataFrame, bbf_events: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    out["bbf_event_id"] = 0
    out["relative_to_bbf_start_s"] = np.nan
    out["inside_bbf_event"] = 0

    for row in bbf_events.itertuples(index=False):
        mask = (out["time"] >= row.start_time) & (out["time"] <= row.end_time)
        out.loc[mask, "bbf_event_id"] = int(row.bbf_event_id)
        out.loc[mask, "relative_to_bbf_start_s"] = (
            out.loc[mask, "time"] - row.start_time
        ).dt.total_seconds()
        out.loc[mask, "inside_bbf_event"] = 1
    out["relative_to_bbf_start_s"] = out["relative_to_bbf_start_s"].fillna(0.0)
    return out


def save_tabular(tabular: pd.DataFrame, output_dir: Path) -> str:
    parquet_path = output_dir / "tabular_features.parquet"
    csv_path = output_dir / "tabular_features.csv"
    try:
        tabular.to_parquet(parquet_path, index=False)
        if csv_path.exists():
            csv_path.unlink()
        return parquet_path.name
    except Exception as exc:
        tabular.to_csv(csv_path, index=False)
        return f"{csv_path.name} (parquet unavailable: {exc.__class__.__name__})"


def sequence_feature_columns() -> list[str]:
    return [
        "relative_time_s",
        *NUMERIC_FEATURES,
        "abs_Bz",
        "abs_Vx",
        "whistler_to_fce",
        *BOOLEAN_FEATURES,
        "feature_observed",
        "inside_bbf_event",
    ]


def make_sequence_for_event(
    grid: pd.DataFrame,
    anchor_time: pd.Timestamp,
    feature_cols: list[str],
    pre_seconds: float,
    post_seconds: float,
    resample_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    window_start = anchor_time - pd.Timedelta(seconds=pre_seconds)
    window_stop = anchor_time + pd.Timedelta(seconds=post_seconds)
    expected_times = pd.date_range(window_start, window_stop, freq=f"{resample_seconds}s")

    seq = grid[(grid["time"] >= window_start) & (grid["time"] <= window_stop)].copy()
    seq = seq.set_index("time").reindex(expected_times)
    seq["relative_time_s"] = (seq.index - anchor_time).total_seconds()
    seq[feature_cols] = seq[feature_cols].ffill().bfill().fillna(0.0)

    x = seq[feature_cols].to_numpy(dtype=np.float32)
    rel_time = seq["relative_time_s"].to_numpy(dtype=np.float32)
    return x, rel_time


def target_values_for_anchor(
    grid: pd.DataFrame,
    anchor_time: pd.Timestamp,
    event_end_time: pd.Timestamp | None = None,
) -> tuple[int, int]:
    future_mask = (grid["time"] >= anchor_time) & (grid["time"] <= anchor_time + pd.Timedelta(seconds=60))
    y_future_60 = int(grid.loc[future_mask, "future_whistler_within_60s"].max()) if future_mask.any() else 0

    end_time = event_end_time if event_end_time is not None else anchor_time + pd.Timedelta(seconds=60)
    strict_mask = (grid["time"] >= anchor_time) & (grid["time"] <= end_time)
    y_strict_event = int(grid.loc[strict_mask, "strict_whistler_event_label"].max()) if strict_mask.any() else 0
    return y_future_60, y_strict_event


def build_sequences(
    grid: pd.DataFrame,
    bbf_events: pd.DataFrame,
    output_dir: Path,
    case_id: str,
    pre_seconds: float,
    post_seconds: float,
    resample_seconds: float,
    regular_anchor_stride_seconds: float,
    include_regular_anchors: bool,
) -> pd.DataFrame:
    sequence_dir = output_dir / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for old_sequence in sequence_dir.glob("seq_*.npz"):
        old_sequence.unlink()

    feature_cols = sequence_feature_columns()
    rows = []
    for event in bbf_events.sort_values("start_time").to_dict("records"):
        x, rel_time = make_sequence_for_event(
            grid, event["start_time"], feature_cols, pre_seconds, post_seconds, resample_seconds
        )
        y_future_60, y_strict_event = target_values_for_anchor(grid, event["start_time"], event["end_time"])

        filename = f"seq_bbf_{int(event['bbf_event_id']):06d}.npz"
        np.savez_compressed(
            sequence_dir / filename,
            X=x,
            relative_time_s=rel_time,
            y_future_whistler_within_60s=np.asarray(y_future_60, dtype=np.int8),
            y_strict_whistler_event_label=np.asarray(y_strict_event, dtype=np.int8),
            bbf_event_id=np.asarray(int(event["bbf_event_id"]), dtype=np.int32),
        )
        rows.append(
            {
                "case_id": case_id,
                "anchor_type": "bbf_event",
                "bbf_event_id": int(event["bbf_event_id"]),
                "anchor_time": event["start_time"],
                "path": f"sequences/{filename}",
                "start_time": event["start_time"],
                "end_time": event["end_time"],
                "timesteps": x.shape[0],
                "features": x.shape[1],
                "target_future_whistler_within_60s": y_future_60,
                "target_strict_whistler_event_label": y_strict_event,
            }
        )
    if include_regular_anchors:
        regular_start = grid["time"].min() + pd.Timedelta(seconds=pre_seconds)
        regular_stop = grid["time"].max() - pd.Timedelta(seconds=post_seconds)
        if regular_start <= regular_stop:
            anchors = pd.date_range(
                regular_start,
                regular_stop,
                freq=f"{regular_anchor_stride_seconds}s",
            )
            bbf_starts = set(pd.to_datetime(bbf_events["start_time"]).astype("int64"))
            for idx, anchor_time in enumerate(anchors, start=1):
                if pd.Timestamp(anchor_time).value in bbf_starts:
                    continue
                x, rel_time = make_sequence_for_event(
                    grid, pd.Timestamp(anchor_time), feature_cols, pre_seconds, post_seconds, resample_seconds
                )
                y_future_60, y_strict_event = target_values_for_anchor(grid, pd.Timestamp(anchor_time))
                filename = f"seq_time_{idx:06d}.npz"
                np.savez_compressed(
                    sequence_dir / filename,
                    X=x,
                    relative_time_s=rel_time,
                    y_future_whistler_within_60s=np.asarray(y_future_60, dtype=np.int8),
                    y_strict_whistler_event_label=np.asarray(y_strict_event, dtype=np.int8),
                    bbf_event_id=np.asarray(0, dtype=np.int32),
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "anchor_type": "regular_time",
                        "bbf_event_id": 0,
                        "anchor_time": pd.Timestamp(anchor_time),
                        "path": f"sequences/{filename}",
                        "start_time": pd.NaT,
                        "end_time": pd.NaT,
                        "timesteps": x.shape[0],
                        "features": x.shape[1],
                        "target_future_whistler_within_60s": y_future_60,
                        "target_strict_whistler_event_label": y_strict_event,
                    }
                )
    return pd.DataFrame(rows)


def write_schema(output_dir: Path, case_config: dict, tabular_name: str, resample_seconds: float) -> None:
    schema = {
        "case_id": case_config.get("case_id"),
        "spacecraft": case_config.get("spacecraft"),
        "resample_seconds": resample_seconds,
        "tabular_dataset": tabular_name,
        "tabular_feature_columns": [
            *NUMERIC_FEATURES,
            "abs_Bz",
            "abs_Vx",
            "whistler_to_fce",
            *BOOLEAN_FEATURES,
            "feature_observed",
            "bbf_event_id",
            "relative_to_bbf_start_s",
            "inside_bbf_event",
        ],
        "sequence_feature_columns": sequence_feature_columns(),
        "label_columns": LABEL_COLUMNS,
        "notes": [
            "Tabular rows are regular time bins intended for Random Forest or other baseline models.",
            "Sequence .npz files are fixed windows around BBF starts intended for LSTM models.",
            "Raw CDF files are not copied into this dataset.",
        ],
    }
    with (output_dir / "feature_schema.json").open("w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2)


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir)
    output_dir = Path(args.output_dir) if args.output_dir else case_dir / "ml_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    case_config = load_case_config(case_dir)
    case_id = case_config.get("case_id", case_dir.name)

    features, bbf_events, whistler_events = load_inputs(case_dir)
    features = coerce_feature_columns(features)
    grid = build_regular_grid(features, args.resample_seconds)
    grid = add_future_whistler_labels(grid, whistler_events, [10.0, 30.0, 60.0])
    grid = attach_event_context(grid, bbf_events)
    grid.insert(0, "case_id", case_id)

    tabular_name = save_tabular(grid, output_dir)
    sequence_index = build_sequences(
        grid,
        bbf_events,
        output_dir,
        case_id,
        args.pre_seconds,
        args.post_seconds,
        args.resample_seconds,
        args.regular_anchor_stride_seconds,
        not args.skip_regular_anchors,
    )
    sequence_index.to_csv(output_dir / "sequence_index.csv", index=False)
    write_schema(output_dir, case_config, tabular_name, args.resample_seconds)

    summary = {
        "case_id": case_id,
        "tabular_rows": int(len(grid)),
        "sequence_count": int(len(sequence_index)),
        "sequence_timesteps": int(sequence_index["timesteps"].iloc[0]) if len(sequence_index) else 0,
        "sequence_features": int(sequence_index["features"].iloc[0]) if len(sequence_index) else 0,
        "tabular_dataset": tabular_name,
    }
    with (output_dir / "ml_dataset_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
