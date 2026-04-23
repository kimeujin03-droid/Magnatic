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
    "fgm": ["tha_fgs_gsmQ", "tha_fgs_btotalQ"],
    "fbk": ["tha_fb_scm1", "tha_fb_scm2", "tha_fb_scm3", "tha_fb_edc12"],
    "mom": ["tha_peim_velocity_gsm", "tha_peim_velocity_gsmQ", "tha_peim_data_quality"],
    "orbit": ["XYZ_GSM"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lightweight THEMIS-A BBF early-prediction pilot dataset."
    )
    parser.add_argument("--start-date", default="2017-07-29", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--output-dir", default=r"C:\Magnetic\datasets\tha_2017-07-29_7d")
    parser.add_argument("--bin-seconds", type=int, default=5)
    parser.add_argument("--bbf-threshold-km-s", type=float, default=400.0)
    parser.add_argument("--min-duration-seconds", type=int, default=80)
    parser.add_argument("--no-tail-filter", action="store_true")
    return parser.parse_args()


def day_bounds(start_date: str, days: int) -> list[tuple[datetime, datetime]]:
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    return [(start + timedelta(days=i), start + timedelta(days=i + 1)) for i in range(days)]


def fetch_dataset(client: CdasWs, kind: str, start: datetime, stop: datetime):
    last_error = None
    for attempt in range(1, 4):
        try:
            status, data = client.get_data(
                DATASETS[kind],
                VARIABLES[kind],
                start.isoformat().replace("+00:00", "Z"),
                stop.isoformat().replace("+00:00", "Z"),
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(2 * attempt)
    else:
        raise RuntimeError(last_error)
    if data is None:
        raise RuntimeError(f"No data returned for {DATASETS[kind]}: {status}")
    return data


def clean_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr[np.abs(arr) > 1e30] = np.nan
    return arr


def time_coord_name(data_array) -> str | None:
    for dim in data_array.dims:
        coord = data_array.coords.get(dim)
        if coord is not None and np.issubdtype(coord.dtype, np.datetime64):
            return dim
    for dim in data_array.dims:
        if "epoch" in dim.lower() or dim.lower() == "epoch":
            return dim
    return None


def resample_numeric(df: pd.DataFrame, bin_seconds: int, how: str = "mean") -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_index()
    rule = f"{bin_seconds}s"
    if how == "max":
        return df.resample(rule).max()
    return df.resample(rule).mean()


def vector_frame(dataset, var_name: str, columns: list[str]) -> pd.DataFrame:
    da = dataset[var_name]
    tdim = time_coord_name(da)
    if tdim is None:
        return pd.DataFrame(columns=columns)
    time = pd.to_datetime(da.coords[tdim].values)
    values = clean_values(da.values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[1] < len(columns):
        return pd.DataFrame(index=time)
    return pd.DataFrame(values[:, : len(columns)], index=time, columns=columns)


def scalar_frame(dataset, var_name: str, column: str) -> pd.DataFrame:
    da = dataset[var_name]
    tdim = time_coord_name(da)
    if tdim is None:
        return pd.DataFrame(columns=[column])
    time = pd.to_datetime(da.coords[tdim].values)
    values = clean_values(da.values).reshape(-1)
    return pd.DataFrame({column: values}, index=time)


def build_fgm_features(dataset, bin_seconds: int) -> pd.DataFrame:
    vec = vector_frame(dataset, "tha_fgs_gsmQ", ["Bx", "By", "Bz"])
    total = scalar_frame(dataset, "tha_fgs_btotalQ", "B_total")
    frame = resample_numeric(vec.join(total, how="outer"), bin_seconds)
    if frame.empty:
        return frame
    frame["B_std_1m"] = frame["B_total"].rolling(12, min_periods=3).std()
    frame["Bz_std_1m"] = frame["Bz"].rolling(12, min_periods=3).std()
    sign = np.sign(frame["Bz"])
    frame["Bz_sign_change_1m"] = sign.ne(sign.shift()).rolling(12, min_periods=3).mean()
    frame["B_total_slope_1m"] = frame["B_total"].diff(12) / 60.0
    return frame


def build_mom_features(dataset, bin_seconds: int) -> pd.DataFrame:
    frame = vector_frame(dataset, "tha_peim_velocity_gsm", ["Vx", "Vy", "Vz"])
    good_quality = vector_frame(dataset, "tha_peim_velocity_gsmQ", ["Vx_good", "Vy_good", "Vz_good"])
    quality = scalar_frame(dataset, "tha_peim_data_quality", "mom_quality")
    frame = frame.join(good_quality, how="outer").join(quality, how="outer")
    frame = resample_numeric(frame, bin_seconds)
    if frame.empty:
        return frame
    frame["V_abs"] = np.sqrt(frame["Vx"] ** 2 + frame["Vy"] ** 2 + frame["Vz"] ** 2)
    return frame


def fbk_variable_frame(dataset, var_name: str, prefix: str, bin_seconds: int, channels: int = 3) -> pd.DataFrame:
    if var_name not in dataset:
        return pd.DataFrame()
    da = dataset[var_name]
    tdim = time_coord_name(da)
    if tdim is None:
        return pd.DataFrame()
    time = pd.to_datetime(da.coords[tdim].values)
    values = clean_values(da.values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    cols = {}
    n_channels = min(channels, values.shape[1])
    for idx in range(n_channels):
        cols[f"{prefix}_ch{idx}"] = np.log10(np.clip(values[:, idx], 1e-30, None))
    cols[f"{prefix}_mean"] = np.log10(np.clip(np.nanmean(values, axis=1), 1e-30, None))
    cols[f"{prefix}_max"] = np.log10(np.clip(np.nanmax(values, axis=1), 1e-30, None))
    if values.shape[1] >= 2:
        low = np.nanmean(values[:, max(values.shape[1] // 2, 1) :], axis=1)
        high = np.nanmean(values[:, : max(values.shape[1] // 2, 1)], axis=1)
        cols[f"{prefix}_low_high_ratio"] = np.log10(np.clip(low / np.clip(high, 1e-30, None), 1e-30, None))
    frame = pd.DataFrame(cols, index=time)
    frame = resample_numeric(frame, bin_seconds)
    for col in [c for c in frame.columns if c.endswith("_mean") or c.endswith("_max")]:
        frame[f"{col}_slope_1m"] = frame[col].diff(max(1, 60 // bin_seconds)) / 60.0
    return frame


def build_fbk_features(dataset, bin_seconds: int) -> pd.DataFrame:
    frames = [
        fbk_variable_frame(dataset, "tha_fb_scm1", "scm1", bin_seconds),
        fbk_variable_frame(dataset, "tha_fb_scm2", "scm2", bin_seconds),
        fbk_variable_frame(dataset, "tha_fb_scm3", "scm3", bin_seconds),
        fbk_variable_frame(dataset, "tha_fb_edc12", "edc12", bin_seconds),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    for col in [c for c in out.columns if c.endswith("_max")]:
        out[f"{col}_rollmax_1m"] = out[col].rolling(max(1, 60 // bin_seconds), min_periods=3).max()
    return out


def build_orbit_features(dataset, bin_seconds: int) -> pd.DataFrame:
    frame = vector_frame(dataset, "XYZ_GSM", ["GSM_x", "GSM_y", "GSM_z"])
    return resample_numeric(frame, bin_seconds)


def add_labels(frame: pd.DataFrame, threshold: float, min_duration_seconds: int, bin_seconds: int) -> pd.DataFrame:
    b = frame[["Bx", "By", "Bz"]].to_numpy(dtype=np.float64)
    v = frame[["Vx", "Vy", "Vz"]].to_numpy(dtype=np.float64)
    b_norm = np.linalg.norm(b, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    dot = np.einsum("ij,ij->i", v, b)
    parallel = dot / np.where(b_norm > 0, b_norm, np.nan)
    v_perp = np.sqrt(np.clip(v_norm**2 - parallel**2, 0, None))
    frame["V_perp"] = v_perp

    raw = frame["V_perp"].gt(threshold).fillna(False)
    min_bins = max(1, int(np.ceil(min_duration_seconds / bin_seconds)))
    group_id = raw.ne(raw.shift()).cumsum()
    label = pd.Series(False, index=frame.index)
    for _, group in raw.groupby(group_id):
        if bool(group.iloc[0]) and len(group) >= min_bins:
            label.loc[group.index] = True
    frame["bbf_label"] = label.astype(int)
    frame["future_bbf_5m"] = (
        frame["bbf_label"].rolling(max(1, 300 // bin_seconds), min_periods=1).max().shift(-max(1, 300 // bin_seconds) + 1)
    )
    frame["future_bbf_5m"] = frame["future_bbf_5m"].fillna(0).astype(int)
    return frame


def process_day(client: CdasWs, start: datetime, stop: datetime, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    day = start.date().isoformat()
    print(f"{day}: fetching FGM")
    fgm = build_fgm_features(fetch_dataset(client, "fgm", start, stop), args.bin_seconds)
    print(f"{day}: fetching FBK")
    fbk = build_fbk_features(fetch_dataset(client, "fbk", start, stop), args.bin_seconds)
    print(f"{day}: fetching MOM")
    mom = build_mom_features(fetch_dataset(client, "mom", start, stop), args.bin_seconds)
    print(f"{day}: fetching orbit")
    orbit = build_orbit_features(fetch_dataset(client, "orbit", start, stop), args.bin_seconds)

    frame = pd.concat([fgm, fbk, mom, orbit], axis=1).sort_index()
    frame = frame.groupby(level=0).mean(numeric_only=True).sort_index()
    frame = frame.ffill(limit=max(1, 90 // args.bin_seconds))
    before_filter = len(frame)
    if not args.no_tail_filter:
        frame = frame[(frame["GSM_x"] < -9.0) & (frame["GSM_y"].abs() < frame["GSM_x"].abs())]
    frame = add_labels(frame, args.bbf_threshold_km_s, args.min_duration_seconds, args.bin_seconds)
    frame.insert(0, "time", frame.index)
    frame.insert(1, "date", day)
    stats = {
        "date": day,
        "rows_before_tail_filter": int(before_filter),
        "rows_after_tail_filter": int(len(frame)),
        "positive_timestep_labels": int(frame["bbf_label"].sum()) if not frame.empty else 0,
        "positive_future_windows": int(frame["future_bbf_5m"].sum()) if not frame.empty else 0,
    }
    return frame.reset_index(drop=True), stats


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = CdasWs()
    all_frames = []
    day_stats = []
    failures = []
    for start, stop in day_bounds(args.start_date, args.days):
        try:
            frame, stats = process_day(client, start, stop, args)
            all_frames.append(frame)
            day_stats.append(stats)
            print(f"{stats['date']}: rows={stats['rows_after_tail_filter']} positives={stats['positive_timestep_labels']}")
        except Exception as exc:
            failure = {"date": start.date().isoformat(), "error": f"{exc.__class__.__name__}: {exc}"}
            failures.append(failure)
            print(f"{failure['date']}: failed: {failure['error']}")

    data = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    output_path = output_dir / "tha_early_features.parquet"
    data.to_parquet(output_path, index=False)

    summary = {
        "spacecraft": "THA",
        "datasets": DATASETS,
        "variables": VARIABLES,
        "start_date": args.start_date,
        "days": args.days,
        "bin_seconds": args.bin_seconds,
        "tail_filter": None if args.no_tail_filter else "GSM_x < -9 and abs(GSM_y) < abs(GSM_x)",
        "label": f"V_perp > {args.bbf_threshold_km_s} km/s for >= {args.min_duration_seconds} s",
        "rows": int(len(data)),
        "positive_timestep_labels": int(data["bbf_label"].sum()) if "bbf_label" in data else 0,
        "positive_future_windows": int(data["future_bbf_5m"].sum()) if "future_bbf_5m" in data else 0,
        "day_stats": day_stats,
        "failures": failures,
        "output": str(output_path),
        "raw_policy": "CDAWeb data fetched into memory; no raw CDF files are intentionally stored.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
