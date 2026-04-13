import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlretrieve

import cdflib
import numpy as np
import pandas as pd


API_BASE = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1"
DEFAULT_MANIFEST = Path(r"C:\Magnetic\manifests\mms1_2017-01-01_to_2018-01-01_manifest.csv")
DEFAULT_OUTPUT_DIR = Path(r"C:\Magnetic\datasets\yearly_2017_mms1")
DEFAULT_TMP_DIR = Path(r"C:\Magnetic\tmp_mms_download")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream MMS1 FPI fast ion moments by day and export yearly BBF speed candidates."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tmp-dir", default=str(DEFAULT_TMP_DIR))
    parser.add_argument("--bin-seconds", type=float, default=5.0)
    parser.add_argument("--vx-threshold", type=float, default=300.0)
    parser.add_argument("--min-consecutive-bins", type=int, default=2)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Append to existing output and skip completed dates.")
    parser.add_argument("--keep-raw", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timetag"])
    df = df[df["product"] == "fpi_fast_dis_moms"].copy()
    df["date"] = df["timetag"].dt.strftime("%Y-%m-%d")
    return df.sort_values(["date", "file_name"])


def download_file(file_name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / file_name
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"{API_BASE}/download/science?{urlencode({'file': file_name})}"
    urlretrieve(url, target)
    return target


def load_epoch(cdf: cdflib.CDF) -> pd.DatetimeIndex:
    return pd.to_datetime(cdflib.cdfepoch.to_datetime(cdf.varget("Epoch")))


def load_fpi_fast(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        cdf = cdflib.CDF(str(path))
        variables = cdf.cdf_info().zVariables
        bulk_var = next(v for v in variables if "mms1_dis_bulkv_gse_" in v and "_fast" in v)
        epoch = load_epoch(cdf)
        velocity = np.asarray(cdf.varget(bulk_var), dtype=np.float64)
        frames.append(
            pd.DataFrame(
                {
                    "time": epoch,
                    "Vx": velocity[:, 0],
                    "Vy": velocity[:, 1],
                    "Vz": velocity[:, 2],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["Vx", "Vy", "Vz"])
    return pd.concat(frames, ignore_index=True).sort_values("time").set_index("time")


def find_runs(mask: pd.Series) -> list[pd.Index]:
    if mask.empty:
        return []
    group_id = mask.ne(mask.shift()).cumsum()
    runs = []
    for _, group in mask.groupby(group_id):
        if bool(group.iloc[0]):
            runs.append(group.index)
    return runs


def build_events(day: str, fpi: pd.DataFrame, bin_seconds: float, threshold: float, min_bins: int) -> list[dict]:
    binned = fpi.resample(f"{bin_seconds}s").mean().dropna(subset=["Vx"])
    if binned.empty:
        return []

    rows = []
    for direction, mask in [
        ("earthward_candidate", binned["Vx"] >= threshold),
        ("tailward_fastflow_candidate", binned["Vx"] <= -threshold),
    ]:
        for run in find_runs(mask):
            if len(run) < min_bins:
                continue
            event = binned.loc[run]
            peak_idx = event["Vx"].idxmax() if direction == "earthward_candidate" else event["Vx"].idxmin()
            rows.append(
                {
                    "date": day,
                    "direction": direction,
                    "start_time": run[0],
                    "end_time": run[-1],
                    "duration_s": float((run[-1] - run[0]).total_seconds() + bin_seconds),
                    "num_bins": int(len(run)),
                    "peak_time": peak_idx,
                    "peak_vx": float(event.loc[peak_idx, "Vx"]),
                    "peak_abs_vx": float(event["Vx"].abs().max()),
                    "mean_vx": float(event["Vx"].mean()),
                    "mean_vy": float(event["Vy"].mean()),
                    "mean_vz": float(event["Vz"].mean()),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "date",
        "direction",
        "start_time",
        "end_time",
        "duration_s",
        "num_bins",
        "peak_time",
        "peak_vx",
        "peak_abs_vx",
        "mean_vx",
        "mean_vy",
        "mean_vz",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.to_dict("records")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tmp_dir = Path(args.tmp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(Path(args.manifest))
    days = list(manifest["date"].drop_duplicates())
    if args.max_days is not None:
        days = days[: args.max_days]

    event_path = output_dir / "bbf_speed_candidates_2017.csv"
    all_events = load_existing_events(event_path) if args.resume else []
    completed_days = {str(row["date"]) for row in all_events} if args.resume else set()
    failures = []
    for index, day in enumerate(days, start=1):
        if day in completed_days:
            print(f"[{index}/{len(days)}] {day}: already completed, skipping")
            continue
        day_rows = manifest[manifest["date"] == day]
        day_dir = tmp_dir / day
        paths = []
        print(f"[{index}/{len(days)}] {day}: downloading {len(day_rows)} FPI fast files")
        try:
            for row in day_rows.itertuples(index=False):
                paths.append(download_file(row.file_name, day_dir))
            fpi = load_fpi_fast(paths)
            events = build_events(
                day,
                fpi,
                args.bin_seconds,
                args.vx_threshold,
                args.min_consecutive_bins,
            )
            all_events.extend(events)
            print(f"[{index}/{len(days)}] {day}: {len(events)} candidate events")
        except Exception as exc:
            failures.append({"date": day, "error": f"{exc.__class__.__name__}: {exc}"})
            print(f"[{index}/{len(days)}] {day}: failed: {exc}")
        finally:
            if not args.keep_raw and day_dir.exists():
                for file_path in day_dir.glob("*.cdf"):
                    file_path.unlink()
                try:
                    day_dir.rmdir()
                except OSError:
                    pass
        write_csv(event_path, all_events)

    write_csv(event_path, all_events)
    summary = {
        "manifest": str(args.manifest),
        "days_requested": len(days),
        "events": len(all_events),
        "earthward_events": int(sum(row["direction"] == "earthward_candidate" for row in all_events)),
        "tailward_events": int(sum(row["direction"] == "tailward_fastflow_candidate" for row in all_events)),
        "failures": failures,
        "output": str(event_path),
        "raw_policy": "delete after each day" if not args.keep_raw else "keep downloaded raw CDF files",
    }
    with (output_dir / "bbf_speed_candidates_2017_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
