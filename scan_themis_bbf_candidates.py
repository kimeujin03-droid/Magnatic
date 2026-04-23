import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from cdasws import CdasWs


DATASETS = {
    "fgm": "THA_L2_FGM",
    "fbk": "THA_L2_FBK",
    "mom": "THA_L2_MOM",
    "orbit": "THA_OR_SSC",
}

VARIABLES = {
    "fgm": ["tha_fgs_gsmQ"],
    "fbk": ["tha_fb_scm1"],
    "mom": ["tha_peim_velocity_gsm", "tha_peim_data_quality"],
    "orbit": ["XYZ_GSM"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan THEMIS-A tail coverage and BBF label thresholds.")
    parser.add_argument("--start-date", default="2017-01-01", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--output-dir", default=r"C:\Magnetic\datasets\tha_bbf_scan_2017-01-01_14d")
    parser.add_argument("--bin-seconds", type=int, default=5)
    parser.add_argument("--v-thresholds", type=float, nargs="+", default=[200, 250, 300, 350, 400])
    parser.add_argument("--durations", type=int, nargs="+", default=[20, 40, 60, 80])
    parser.add_argument("--input-window-seconds", type=int, default=900)
    parser.add_argument("--prediction-window-seconds", type=int, default=300)
    parser.add_argument("--anchor-stride-seconds", type=int, default=60)
    parser.add_argument("--tail-min-x", type=float, default=-9.0)
    return parser.parse_args()


def bounds(start_date: str, days: int) -> list[tuple[datetime, datetime]]:
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    return [(start + timedelta(days=i), start + timedelta(days=i + 1)) for i in range(days)]


def fetch(client: CdasWs, kind: str, start: datetime, stop: datetime):
    start_s = start.isoformat().replace("+00:00", "Z")
    stop_s = stop.isoformat().replace("+00:00", "Z")
    last_error = None
    for attempt in range(1, 4):
        try:
            _, data = client.get_data(DATASETS[kind], VARIABLES[kind], start_s, stop_s)
            if data is None:
                raise RuntimeError("no data returned")
            return data
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(2 * attempt)
    raise RuntimeError(last_error)


def time_coord_name(data_array) -> str | None:
    for dim in data_array.dims:
        coord = data_array.coords.get(dim)
        if coord is not None and np.issubdtype(coord.dtype, np.datetime64):
            return dim
    for dim in data_array.dims:
        if "epoch" in dim.lower() or dim.lower() == "epoch":
            return dim
    return None


def clean(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr[np.abs(arr) > 1e30] = np.nan
    return arr


def vector_frame(dataset, var_name: str, columns: list[str]) -> pd.DataFrame:
    if var_name not in dataset:
        return pd.DataFrame(columns=columns)
    da = dataset[var_name]
    tdim = time_coord_name(da)
    if tdim is None:
        return pd.DataFrame(columns=columns)
    time_index = pd.to_datetime(da.coords[tdim].values)
    values = clean(da.values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[1] < len(columns):
        return pd.DataFrame(index=time_index)
    return pd.DataFrame(values[:, : len(columns)], index=time_index, columns=columns)


def scalar_frame(dataset, var_name: str, column: str) -> pd.DataFrame:
    if var_name not in dataset:
        return pd.DataFrame(columns=[column])
    da = dataset[var_name]
    tdim = time_coord_name(da)
    if tdim is None:
        return pd.DataFrame(columns=[column])
    time_index = pd.to_datetime(da.coords[tdim].values)
    return pd.DataFrame({column: clean(da.values).reshape(-1)}, index=time_index)


def resample(df: pd.DataFrame, bin_seconds: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_index().resample(f"{bin_seconds}s").mean()


def valid_from_dataset(dataset, var_name: str, column: str, bin_seconds: int) -> pd.DataFrame:
    if var_name not in dataset:
        return pd.DataFrame(columns=[column])
    da = dataset[var_name]
    tdim = time_coord_name(da)
    if tdim is None:
        return pd.DataFrame(columns=[column])
    time_index = pd.to_datetime(da.coords[tdim].values)
    values = clean(da.values)
    if values.ndim > 1:
        valid = np.isfinite(values).any(axis=tuple(range(1, values.ndim)))
    else:
        valid = np.isfinite(values)
    frame = pd.DataFrame({column: valid.astype(float)}, index=time_index)
    return frame.sort_index().resample(f"{bin_seconds}s").max()


def build_daily_frame(client: CdasWs, start: datetime, stop: datetime, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    print(f"{start.date()}: fetching orbit")
    orbit = resample(vector_frame(fetch(client, "orbit", start, stop), "XYZ_GSM", ["GSM_x", "GSM_y", "GSM_z"]), args.bin_seconds)
    print(f"{start.date()}: fetching MOM")
    mom_data = fetch(client, "mom", start, stop)
    mom = resample(vector_frame(mom_data, "tha_peim_velocity_gsm", ["Vx", "Vy", "Vz"]), args.bin_seconds)
    quality = resample(scalar_frame(mom_data, "tha_peim_data_quality", "mom_quality"), args.bin_seconds)
    mom_valid = valid_from_dataset(mom_data, "tha_peim_velocity_gsm", "mom_valid", args.bin_seconds)

    print(f"{start.date()}: fetching FGM")
    fgm_data = fetch(client, "fgm", start, stop)
    fgm = resample(vector_frame(fgm_data, "tha_fgs_gsmQ", ["Bx", "By", "Bz"]), args.bin_seconds)
    fgm_valid = valid_from_dataset(fgm_data, "tha_fgs_gsmQ", "fgm_valid", args.bin_seconds)
    print(f"{start.date()}: fetching FBK coverage")
    fbk_valid = valid_from_dataset(fetch(client, "fbk", start, stop), "tha_fb_scm1", "fbk_valid", args.bin_seconds)

    frame = pd.concat([orbit, mom, quality, mom_valid, fgm, fgm_valid, fbk_valid], axis=1).sort_index()
    frame = frame.groupby(level=0).mean(numeric_only=True).sort_index()
    frame = frame.ffill(limit=max(1, 90 // args.bin_seconds))
    frame["tail"] = (frame["GSM_x"] < args.tail_min_x) & (frame["GSM_y"].abs() < frame["GSM_x"].abs())
    frame["mom_valid"] = frame["mom_valid"].fillna(0).gt(0)
    frame["fgm_valid"] = frame["fgm_valid"].fillna(0).gt(0)
    frame["fbk_valid"] = frame["fbk_valid"].fillna(0).gt(0)
    frame["usable"] = frame["tail"] & frame["mom_valid"] & frame["fgm_valid"] & frame["fbk_valid"]

    v = frame[["Vx", "Vy", "Vz"]].to_numpy(dtype=np.float64)
    b = frame[["Bx", "By", "Bz"]].to_numpy(dtype=np.float64)
    b_norm = np.linalg.norm(b, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    dot = np.einsum("ij,ij->i", v, b)
    v_parallel = dot / np.where(b_norm > 0, b_norm, np.nan)
    frame["V_abs"] = np.linalg.norm(v, axis=1)
    frame["V_perp"] = np.sqrt(np.clip(v_norm**2 - v_parallel**2, 0, None))

    day = start.date().isoformat()
    usable_v = frame.loc[frame["usable"], "V_perp"].dropna()
    stats = {
        "date": day,
        "tail_total_seconds": int(frame["tail"].sum() * args.bin_seconds),
        "tail_fraction": float(frame["tail"].mean()),
        "mom_valid_seconds": int((frame["tail"] & frame["mom_valid"]).sum() * args.bin_seconds),
        "mom_valid_fraction": float(frame.loc[frame["tail"], "mom_valid"].mean()) if frame["tail"].any() else 0.0,
        "fgm_valid_seconds": int((frame["tail"] & frame["fgm_valid"]).sum() * args.bin_seconds),
        "fbk_valid_seconds": int((frame["tail"] & frame["fbk_valid"]).sum() * args.bin_seconds),
        "usable_overlap_seconds": int(frame["usable"].sum() * args.bin_seconds),
        "max_abs_vperp": float(usable_v.max()) if not usable_v.empty else np.nan,
        "p95_abs_vperp": float(usable_v.quantile(0.95)) if not usable_v.empty else np.nan,
        "quality_good_fraction": float((frame.loc[frame["usable"], "mom_quality"].fillna(np.inf) <= 0).mean())
        if frame["usable"].any()
        else 0.0,
    }
    return frame, stats


def event_runs(mask: pd.Series, min_bins: int) -> list[pd.Index]:
    runs = []
    group_id = mask.ne(mask.shift()).cumsum()
    for _, group in mask.groupby(group_id):
        if bool(group.iloc[0]) and len(group) >= min_bins:
            runs.append(group.index)
    return runs


def scan_thresholds(frames: dict[str, pd.DataFrame], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    event_rows = []
    pred_bins = max(1, args.prediction_window_seconds // args.bin_seconds)
    anchor_step = max(1, args.anchor_stride_seconds // args.bin_seconds)
    input_bins = max(1, args.input_window_seconds // args.bin_seconds)
    for threshold in args.v_thresholds:
        for duration in args.durations:
            min_bins = max(1, int(np.ceil(duration / args.bin_seconds)))
            event_count = 0
            positive_seconds = 0
            event_durations = []
            positive_anchor_count = 0
            inside_bbf_anchor_count = 0
            clean_positive_anchor_count = 0
            days_with_event = 0

            for date, frame in frames.items():
                base = frame["usable"] & frame["V_perp"].ge(threshold).fillna(False)
                label = pd.Series(False, index=frame.index)
                runs = event_runs(base, min_bins)
                if runs:
                    days_with_event += 1
                event_count += len(runs)
                for run in runs:
                    label.loc[run] = True
                    duration_s = len(run) * args.bin_seconds
                    event_durations.append(duration_s)
                    positive_seconds += duration_s
                    event_slice = frame.loc[run]
                    event_rows.append(
                        {
                            "date": date,
                            "v_threshold": threshold,
                            "min_duration_s": duration,
                            "start_time": run[0],
                            "end_time": run[-1],
                            "duration_s": duration_s,
                            "max_vperp": float(event_slice["V_perp"].max()),
                            "median_vperp": float(event_slice["V_perp"].median()),
                            "max_vabs": float(event_slice["V_abs"].max()),
                            "median_mom_quality": float(event_slice["mom_quality"].median()),
                        }
                    )

                anchor_positions = np.arange(input_bins, max(input_bins, len(frame) - pred_bins), anchor_step)
                label_values = label.to_numpy()
                for anchor in anchor_positions:
                    future_positive = label_values[anchor : anchor + pred_bins].any()
                    if future_positive:
                        positive_anchor_count += 1
                        if label_values[anchor]:
                            inside_bbf_anchor_count += 1
                        else:
                            clean_positive_anchor_count += 1

            total_usable_seconds = sum(int(frame["usable"].sum() * args.bin_seconds) for frame in frames.values())
            rows.append(
                {
                    "v_threshold": threshold,
                    "min_duration_s": duration,
                    "bbf_event_count": event_count,
                    "total_positive_seconds": positive_seconds,
                    "positive_fraction": positive_seconds / total_usable_seconds if total_usable_seconds else 0.0,
                    "max_event_duration_s": max(event_durations) if event_durations else 0,
                    "median_event_duration_s": float(np.median(event_durations)) if event_durations else 0.0,
                    "positive_anchor_count": positive_anchor_count,
                    "inside_bbf_anchor_count": inside_bbf_anchor_count,
                    "clean_positive_anchor_count": clean_positive_anchor_count,
                    "days_with_at_least_one_event": days_with_event,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(event_rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = CdasWs()
    frames = {}
    orbit_rows = []
    failures = []
    for start, stop in bounds(args.start_date, args.days):
        date = start.date().isoformat()
        try:
            frame, stats = build_daily_frame(client, start, stop, args)
            frames[date] = frame
            orbit_rows.append(stats)
            print(
                f"{date}: tail={stats['tail_total_seconds']}s usable={stats['usable_overlap_seconds']}s "
                f"max_v={stats['max_abs_vperp']:.1f}"
            )
        except Exception as exc:
            failure = {"date": date, "error": f"{exc.__class__.__name__}: {exc}"}
            failures.append(failure)
            print(f"{date}: failed: {failure['error']}")

    orbit_df = pd.DataFrame(orbit_rows)
    orbit_df.to_csv(output_dir / "orbit_scan.csv", index=False)

    usable_dates = orbit_df[
        (orbit_df["tail_total_seconds"] >= 3600)
        & (orbit_df["usable_overlap_seconds"] >= 1800)
        & (orbit_df["mom_valid_fraction"] >= 0.3)
        & (orbit_df["max_abs_vperp"] >= 250)
    ]["date"].tolist()
    threshold_frames = {date: frame for date, frame in frames.items() if date in usable_dates}
    threshold_df, event_df = scan_thresholds(threshold_frames, args)
    threshold_df.to_csv(output_dir / "threshold_scan.csv", index=False)
    event_df.to_csv(output_dir / "threshold_events.csv", index=False)

    summary = {
        "spacecraft": "THA",
        "datasets": DATASETS,
        "variables": VARIABLES,
        "start_date": args.start_date,
        "days": args.days,
        "bin_seconds": args.bin_seconds,
        "tail_filter": f"GSM_x < {args.tail_min_x} and abs(GSM_y) < abs(GSM_x)",
        "usable_date_criteria": {
            "tail_total_seconds": ">= 3600",
            "usable_overlap_seconds": ">= 1800",
            "mom_valid_fraction": ">= 0.3",
            "max_abs_vperp": ">= 250",
        },
        "usable_dates": usable_dates,
        "failures": failures,
        "orbit_scan": str(output_dir / "orbit_scan.csv"),
        "threshold_scan": str(output_dir / "threshold_scan.csv"),
        "threshold_events": str(output_dir / "threshold_events.csv"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
