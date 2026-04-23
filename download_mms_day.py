import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlretrieve

import pandas as pd


API_BASE = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1"
DEFAULT_MANIFEST = Path(r"C:\Magnetic\manifests\mms1_2017-01-01_to_2018-01-01_manifest.csv")
DEFAULT_CASES_DIR = Path(r"C:\Magnetic\cases")
DEFAULT_PRODUCTS = ["fgm_srvy", "fgm_brst", "fpi_brst_dis_moms", "scm_brst_schb"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one day of MMS files from an existing manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format.")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--cases-dir", default=str(DEFAULT_CASES_DIR))
    parser.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS)
    parser.add_argument("--max-download-gib", type=float, default=2.0)
    return parser.parse_args()


def download_file(file_name: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / file_name
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"{API_BASE}/download/science?{urlencode({'file': file_name})}"
    urlretrieve(url, target)
    return target


def write_case_configs(case_dir: Path, case_id: str, date: str) -> None:
    case_cfg = {
        "case_id": case_id,
        "spacecraft": "MMS1",
        "time_start_utc": f"{date}T00:00:00Z",
        "time_stop_utc": f"{date}T23:59:59Z",
        "bbf_definition": {
            "direction": "earthward_only",
            "vx_threshold_km_s": 300.0,
            "min_consecutive_bins": 2,
            "bin_size_seconds": 5,
            "event_level_bz_support_nt": 3.0,
        },
        "whistler_definition": {
            "band": "0.1-0.5 fce",
            "minimum_duration_seconds": 0.25,
            "minimum_band_occupancy": 0.6,
            "maximum_peak_frequency_cv": 0.35,
            "merge_gap_seconds": 1.0,
        },
    }
    with (case_dir / "case_config.json").open("w", encoding="utf-8") as fh:
        json.dump(case_cfg, fh, indent=2)

    baseline_dir = case_dir / "baseline_santolik"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_cfg = {
        "case_name": case_id,
        "detector_name": "baseline_santolik_style",
        "spacecraft": "MMS1",
        "time_range_utc": {"start": f"{date}T00:00:00Z", "stop": f"{date}T23:59:59Z"},
        "frequency_gate": {"low_fraction_of_fce": 0.1, "high_fraction_of_fce": 0.5},
        "quality_gate": {
            "ellipticity_min": 0.7,
            "planarity_min": 0.7,
            "magnetic_psd_min_nt2_per_hz": 1e-7,
        },
        "event_merge": {"minimum_duration_s": 0.25, "maximum_gap_s": 1.0},
        "notes": ["Full-day pilot download generated from manifest."],
    }
    with (baseline_dir / "baseline_config.json").open("w", encoding="utf-8") as fh:
        json.dump(baseline_cfg, fh, indent=2)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, parse_dates=["timetag"])
    rows = manifest[
        manifest["timetag"].dt.strftime("%Y-%m-%d").eq(args.date)
        & manifest["product"].isin(args.products)
    ].sort_values(["product", "file_name"])
    if rows.empty:
        raise ValueError(f"No manifest rows found for {args.date} and products {args.products}")

    expected_gib = rows["file_size"].sum() / 1024**3
    if expected_gib > args.max_download_gib:
        raise RuntimeError(
            f"Expected download is {expected_gib:.3f} GiB, above limit {args.max_download_gib:.3f} GiB"
        )

    case_id = args.case_id or f"{args.date}_mms1_full_day"
    case_dir = Path(args.cases_dir) / case_id
    data_dir = case_dir / "data"
    case_dir.mkdir(parents=True, exist_ok=True)
    write_case_configs(case_dir, case_id, args.date)

    downloaded = []
    for idx, row in enumerate(rows.itertuples(index=False), start=1):
        print(f"[{idx}/{len(rows)}] {row.product}: {row.file_name}")
        path = download_file(row.file_name, data_dir)
        downloaded.append({"product": row.product, "file_name": row.file_name, "path": str(path)})

    summary = {
        "case_id": case_id,
        "date": args.date,
        "products": args.products,
        "file_count": len(downloaded),
        "expected_gib": round(expected_gib, 3),
        "case_dir": str(case_dir),
        "data_dir": str(data_dir),
    }
    with (case_dir / "download_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
