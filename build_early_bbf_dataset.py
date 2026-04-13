import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CASE_DIR = Path(r"C:\Magnetic\cases\2017-07-29_mms1_earthward_bbf")

RAW_COLUMNS = [
    "Bz",
    "fce_hz",
    "whistler_band_power",
    "wave_total_power_10_4000hz",
    "whistler_ratio",
    "whistler_activity_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a causal pilot dataset for early BBF prediction from one MMS case."
    )
    parser.add_argument(
        "--case-dir",
        default=os.environ.get("MMS_CASE_DIR", str(DEFAULT_CASE_DIR)),
        help="Case directory containing whistler_model_features.csv and bbf_events.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <case-dir>/early_bbf_dataset.",
    )
    parser.add_argument("--resample-seconds", type=float, default=1.0)
    parser.add_argument("--history-seconds", type=float, default=900.0)
    parser.add_argument("--horizon-seconds", type=float, default=300.0)
    parser.add_argument("--anchor-stride-seconds", type=float, default=300.0)
    parser.add_argument("--max-sequences", type=int, default=14)
    return parser.parse_args()


def load_case_config(case_dir: Path) -> dict:
    path = case_dir / "case_config.json"
    if not path.exists():
        return {"case_id": case_dir.name}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_inputs(case_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = case_dir / "whistler_model_features.csv"
    bbf_path = case_dir / "bbf_events.csv"
    missing = [str(path) for path in [feature_path, bbf_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    features = pd.read_csv(feature_path, parse_dates=["time"]).sort_values("time")
    bbf_events = pd.read_csv(bbf_path, parse_dates=["start_time", "end_time", "peak_time"])
    return features, bbf_events


def build_causal_grid(features: pd.DataFrame, resample_seconds: float) -> pd.DataFrame:
    df = features.copy()
    for col in RAW_COLUMNS:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "whistler_feature_valid" not in df:
        df["whistler_feature_valid"] = 0.0
    df["whistler_feature_valid"] = pd.to_numeric(df["whistler_feature_valid"], errors="coerce").fillna(0.0)

    grid = df.set_index("time").sort_index()[RAW_COLUMNS + ["whistler_feature_valid"]]
    grid = grid.resample(f"{resample_seconds}s").mean().ffill().fillna(0.0)

    dt = float(resample_seconds)
    window_60 = max(1, int(round(60.0 / dt)))
    wave = grid["whistler_band_power"]
    bz = grid["Bz"]

    out = pd.DataFrame(index=grid.index)
    out["Bz"] = bz
    out["abs_Bz"] = bz.abs()
    out["dBz_dt"] = bz.diff() / dt
    out["Bz_delta_60s"] = bz - bz.shift(window_60)
    out["Bz_rolling_std_60s"] = bz.rolling(window_60, min_periods=1).std()
    out["Bz_sign_change_rate_60s"] = (
        (np.sign(bz).diff().fillna(0.0) != 0.0).astype(float).rolling(window_60, min_periods=1).mean()
    )
    out["fce_hz"] = grid["fce_hz"]
    out["whistler_band_power"] = wave
    out["wave_total_power_10_4000hz"] = grid["wave_total_power_10_4000hz"]
    out["whistler_ratio"] = grid["whistler_ratio"]
    out["whistler_activity_score"] = grid["whistler_activity_score"]
    out["wave_power_delta_60s"] = wave - wave.shift(window_60)
    out["wave_power_increase_rate_60s"] = out["wave_power_delta_60s"] / 60.0
    out["wave_power_rolling_mean_60s"] = wave.rolling(window_60, min_periods=1).mean()
    out["wave_power_rolling_max_60s"] = wave.rolling(window_60, min_periods=1).max()
    out["wave_power_rolling_var_60s"] = wave.rolling(window_60, min_periods=1).var()
    out["whistler_feature_observed"] = grid["whistler_feature_valid"].clip(lower=0.0, upper=1.0)

    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).reset_index(names="time")


def add_bbf_context(grid: pd.DataFrame, bbf_events: pd.DataFrame, horizon_seconds: float) -> pd.DataFrame:
    out = grid.copy()
    out["target_bbf_within_5min"] = 0
    out["inside_bbf_at_time"] = 0

    starts = pd.to_datetime(bbf_events["start_time"]).dropna().sort_values()
    starts_ns = starts.to_numpy(dtype="datetime64[ns]").astype("int64")
    times_ns = out["time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    horizon_ns = int(horizon_seconds * 1_000_000_000)

    if len(starts_ns):
        pos = np.searchsorted(starts_ns, times_ns, side="right")
        has_next = pos < len(starts_ns)
        delta = np.full(len(times_ns), np.iinfo(np.int64).max, dtype=np.int64)
        delta[has_next] = starts_ns[pos[has_next]] - times_ns[has_next]
        out["target_bbf_within_5min"] = ((delta > 0) & (delta <= horizon_ns)).astype(np.int8)

    for event in bbf_events.itertuples(index=False):
        mask = (out["time"] >= event.start_time) & (out["time"] <= event.end_time)
        out.loc[mask, "inside_bbf_at_time"] = 1

    return out


def sequence_feature_columns() -> list[str]:
    return [
        "relative_time_s",
        "Bz",
        "abs_Bz",
        "dBz_dt",
        "Bz_delta_60s",
        "Bz_rolling_std_60s",
        "Bz_sign_change_rate_60s",
        "fce_hz",
        "whistler_band_power",
        "wave_total_power_10_4000hz",
        "whistler_ratio",
        "whistler_activity_score",
        "wave_power_delta_60s",
        "wave_power_increase_rate_60s",
        "wave_power_rolling_mean_60s",
        "wave_power_rolling_max_60s",
        "wave_power_rolling_var_60s",
        "whistler_feature_observed",
    ]


def choose_anchors(
    grid: pd.DataFrame,
    bbf_events: pd.DataFrame,
    history_seconds: float,
    horizon_seconds: float,
    anchor_stride_seconds: float,
    max_sequences: int,
) -> list[pd.Timestamp]:
    start = grid["time"].min() + pd.Timedelta(seconds=history_seconds)
    stop = grid["time"].max() - pd.Timedelta(seconds=horizon_seconds)
    regular = []
    if start <= stop:
        regular = list(pd.date_range(start, stop, freq=f"{anchor_stride_seconds}s"))

    positive = [
        pd.Timestamp(ts) - pd.Timedelta(seconds=horizon_seconds)
        for ts in pd.to_datetime(bbf_events["start_time"]).dropna()
    ]
    anchors = sorted({pd.Timestamp(anchor) for anchor in [*regular, *positive] if start <= anchor <= stop})
    if max_sequences > 0:
        return anchors[:max_sequences]
    return anchors


def make_sequence(
    grid: pd.DataFrame,
    anchor_time: pd.Timestamp,
    feature_cols: list[str],
    history_seconds: float,
    resample_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    window_start = anchor_time - pd.Timedelta(seconds=history_seconds)
    expected_times = pd.date_range(window_start, anchor_time, freq=f"{resample_seconds}s")
    seq = grid[(grid["time"] >= window_start) & (grid["time"] <= anchor_time)].set_index("time")
    seq = seq.reindex(expected_times).ffill().bfill().fillna(0.0)
    seq["relative_time_s"] = (seq.index - anchor_time).total_seconds()
    x = seq[feature_cols].to_numpy(dtype=np.float32)
    rel_time = seq["relative_time_s"].to_numpy(dtype=np.float32)
    return x, rel_time


def write_sequences(
    grid: pd.DataFrame,
    anchors: list[pd.Timestamp],
    output_dir: Path,
    case_id: str,
    history_seconds: float,
    horizon_seconds: float,
    resample_seconds: float,
) -> pd.DataFrame:
    sequence_dir = output_dir / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for old_sequence in sequence_dir.glob("early_bbf_*.npz"):
        old_sequence.unlink()

    feature_cols = sequence_feature_columns()
    rows = []
    by_time = grid.set_index("time")
    for idx, anchor_time in enumerate(anchors, start=1):
        x, rel_time = make_sequence(grid, anchor_time, feature_cols, history_seconds, resample_seconds)
        row_at_anchor = by_time.loc[anchor_time] if anchor_time in by_time.index else by_time.iloc[by_time.index.searchsorted(anchor_time)]
        y = int(row_at_anchor["target_bbf_within_5min"])
        inside = int(row_at_anchor["inside_bbf_at_time"])
        filename = f"early_bbf_{idx:06d}.npz"
        np.savez_compressed(
            sequence_dir / filename,
            X=x,
            relative_time_s=rel_time,
            y_bbf_within_5min=np.asarray(y, dtype=np.int8),
            anchor_time=np.asarray(str(anchor_time)),
        )
        rows.append(
            {
                "case_id": case_id,
                "anchor_time": anchor_time,
                "path": f"sequences/{filename}",
                "timesteps": x.shape[0],
                "features": x.shape[1],
                "target_bbf_within_5min": y,
                "inside_bbf_at_anchor": inside,
                "history_seconds": history_seconds,
                "horizon_seconds": horizon_seconds,
            }
        )
    return pd.DataFrame(rows)


def save_tabular(grid: pd.DataFrame, output_dir: Path) -> str:
    parquet_path = output_dir / "causal_features.parquet"
    csv_path = output_dir / "causal_features.csv"
    try:
        grid.to_parquet(parquet_path, index=False)
        if csv_path.exists():
            csv_path.unlink()
        return parquet_path.name
    except Exception as exc:
        grid.to_csv(csv_path, index=False)
        return f"{csv_path.name} (parquet unavailable: {exc.__class__.__name__})"


def write_schema(output_dir: Path, tabular_name: str, args: argparse.Namespace) -> None:
    schema = {
        "task": "early_bbf_prediction",
        "input_window": f"t-{int(args.history_seconds)}s to t",
        "prediction_window": f"t to t+{int(args.horizon_seconds)}s",
        "target": "BBF start strictly after t and within the prediction window.",
        "tabular_dataset": tabular_name,
        "sequence_feature_columns": sequence_feature_columns(),
        "excluded_from_inputs": [
            "Vx",
            "bbf_event_label",
            "bbf_event_id",
            "inside_bbf_at_time",
            "strict_whistler_segment_label",
            "strict_whistler_event_label",
            "baseline_pass",
            "nearest_whistler",
            "future_whistler_*",
        ],
        "notes": [
            "All rolling/delta features are computed on the resampled grid using current-or-past samples only.",
            "Wave inputs are continuous activity statistics, not whistler event flags.",
        ],
    }
    with (output_dir / "feature_schema.json").open("w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2)


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir)
    output_dir = Path(args.output_dir) if args.output_dir else case_dir / "early_bbf_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_case_config(case_dir)
    case_id = cfg.get("case_id", case_dir.name)
    features, bbf_events = load_inputs(case_dir)
    grid = build_causal_grid(features, args.resample_seconds)
    grid = add_bbf_context(grid, bbf_events, args.horizon_seconds)
    grid.insert(0, "case_id", case_id)

    anchors = choose_anchors(
        grid,
        bbf_events,
        args.history_seconds,
        args.horizon_seconds,
        args.anchor_stride_seconds,
        args.max_sequences,
    )
    sequence_index = write_sequences(
        grid,
        anchors,
        output_dir,
        case_id,
        args.history_seconds,
        args.horizon_seconds,
        args.resample_seconds,
    )
    sequence_index.to_csv(output_dir / "sequence_index.csv", index=False)

    tabular_name = save_tabular(grid, output_dir)
    write_schema(output_dir, tabular_name, args)

    summary = {
        "case_id": case_id,
        "tabular_rows": int(len(grid)),
        "sequence_count": int(len(sequence_index)),
        "positive_sequences": int(sequence_index["target_bbf_within_5min"].sum()) if len(sequence_index) else 0,
        "sequence_timesteps": int(sequence_index["timesteps"].iloc[0]) if len(sequence_index) else 0,
        "sequence_features": int(sequence_index["features"].iloc[0]) if len(sequence_index) else 0,
        "output_dir": str(output_dir),
        "tabular_dataset": tabular_name,
    }
    with (output_dir / "early_bbf_dataset_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
