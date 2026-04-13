import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from build_ml_dataset import main as build_single_case_main


DEFAULT_CASES_DIR = Path(r"C:\Magnetic\cases")
DEFAULT_OUTPUT_DIR = Path(r"C:\Magnetic\datasets\yearly_2017_mms1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine case-level ML datasets into a year-range dataset without copying raw CDF files."
    )
    parser.add_argument("--cases-dir", default=str(DEFAULT_CASES_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start", default="2017-01-01", help="Inclusive start date.")
    parser.add_argument("--stop", default="2018-01-01", help="Exclusive stop date.")
    parser.add_argument("--rebuild-cases", action="store_true", help="Rebuild each case ml_dataset before merging.")
    return parser.parse_args()


def discover_case_dirs(cases_dir: Path, start: pd.Timestamp, stop: pd.Timestamp) -> list[Path]:
    case_dirs = []
    for path in sorted(cases_dir.iterdir()):
        if not path.is_dir():
            continue
        config_path = path / "case_config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            case_start = pd.to_datetime(cfg.get("time_start_utc"), utc=True, errors="coerce")
            if pd.notna(case_start) and start <= case_start < stop:
                case_dirs.append(path)
        elif (path / "whistler_model_features.csv").exists():
            case_dirs.append(path)
    return case_dirs


def rebuild_case(case_dir: Path) -> None:
    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = ["build_ml_dataset.py", "--case-dir", str(case_dir)]
        build_single_case_main()
    finally:
        sys.argv = old_argv


def load_case_tabular(ml_dir: Path) -> pd.DataFrame | None:
    parquet_path = ml_dir / "tabular_features.parquet"
    csv_path = ml_dir / "tabular_features.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["time"])
    return None


def copy_sequences(case_dir: Path, output_sequence_dir: Path) -> pd.DataFrame | None:
    ml_dir = case_dir / "ml_dataset"
    index_path = ml_dir / "sequence_index.csv"
    if not index_path.exists():
        return None

    sequence_index = pd.read_csv(index_path)
    rows = []
    for row in sequence_index.to_dict("records"):
        source = ml_dir / row["path"]
        if not source.exists():
            continue
        target_name = f"{case_dir.name}_{source.name}"
        target = output_sequence_dir / target_name
        shutil.copy2(source, target)
        row["source_case_dir"] = case_dir.name
        row["path"] = f"sequences/{target_name}"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cases_dir = Path(args.cases_dir)
    output_dir = Path(args.output_dir)
    output_sequence_dir = output_dir / "sequences"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_sequence_dir.mkdir(parents=True, exist_ok=True)

    start = pd.to_datetime(args.start, utc=True)
    stop = pd.to_datetime(args.stop, utc=True)
    case_dirs = discover_case_dirs(cases_dir, start, stop)

    tabular_frames = []
    sequence_frames = []
    for case_dir in case_dirs:
        if args.rebuild_cases:
            rebuild_case(case_dir)
        tabular = load_case_tabular(case_dir / "ml_dataset")
        if tabular is not None:
            tabular_frames.append(tabular)
        sequence_index = copy_sequences(case_dir, output_sequence_dir)
        if sequence_index is not None and len(sequence_index):
            sequence_frames.append(sequence_index)

    if tabular_frames:
        tabular_all = pd.concat(tabular_frames, ignore_index=True).sort_values(["case_id", "time"])
        tabular_all.to_parquet(output_dir / "tabular_features.parquet", index=False)
    else:
        tabular_all = pd.DataFrame()

    if sequence_frames:
        sequence_all = pd.concat(sequence_frames, ignore_index=True)
        sequence_all.to_csv(output_dir / "sequence_index.csv", index=False)
    else:
        sequence_all = pd.DataFrame()
        sequence_all.to_csv(output_dir / "sequence_index.csv", index=False)

    summary = {
        "start": args.start,
        "stop": args.stop,
        "case_count": len(case_dirs),
        "cases": [path.name for path in case_dirs],
        "tabular_rows": int(len(tabular_all)),
        "sequence_count": int(len(sequence_all)),
        "output_dir": str(output_dir),
        "note": "This merges available case-level feature datasets only; it does not download or retain raw CDF files.",
    }
    with (output_dir / "yearly_ml_dataset_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
