import argparse
import csv
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


API_BASE = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1"

PRODUCTS = {
    "fgm_srvy": {
        "sc_id": "mms1",
        "instrument_id": "fgm",
        "data_rate_mode": "srvy",
        "data_level": "l2",
    },
    "fpi_fast_dis_moms": {
        "sc_id": "mms1",
        "instrument_id": "fpi",
        "data_rate_mode": "fast",
        "data_level": "l2",
        "descriptor": "dis-moms",
    },
    "fpi_brst_dis_moms": {
        "sc_id": "mms1",
        "instrument_id": "fpi",
        "data_rate_mode": "brst",
        "data_level": "l2",
        "descriptor": "dis-moms",
    },
    "fgm_brst": {
        "sc_id": "mms1",
        "instrument_id": "fgm",
        "data_rate_mode": "brst",
        "data_level": "l2",
    },
    "scm_brst_schb": {
        "sc_id": "mms1",
        "instrument_id": "scm",
        "data_rate_mode": "brst",
        "data_level": "l2",
        "descriptor": "schb",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an MMS SDC file manifest for a date range.")
    parser.add_argument("--start", default="2017-01-01", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--stop", default="2018-01-01", help="Exclusive stop date, YYYY-MM-DD.")
    parser.add_argument("--output-dir", default=r"C:\Magnetic\manifests")
    parser.add_argument(
        "--products",
        nargs="+",
        default=list(PRODUCTS.keys()),
        choices=sorted(PRODUCTS.keys()),
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to wait between API calls.")
    return parser.parse_args()


def query_file_info(product_name: str, start: str, stop: str) -> list[dict]:
    params = dict(PRODUCTS[product_name])
    params["start_date"] = start
    params["end_date"] = stop
    url = f"{API_BASE}/file_info/science?{urlencode(params)}"
    with urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = []
    for file_info in payload.get("files", []):
        rows.append(
            {
                "product": product_name,
                "file_name": file_info.get("file_name"),
                "file_size": int(file_info.get("file_size", 0) or 0),
                "timetag": file_info.get("timetag"),
                "modified_date": file_info.get("modified_date"),
                "query_start": start,
                "query_stop": stop,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "product",
        "file_name",
        "file_size",
        "timetag",
        "modified_date",
        "query_start",
        "query_stop",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary = []
    for product in args.products:
        rows = query_file_info(product, args.start, args.stop)
        all_rows.extend(rows)
        total_bytes = sum(row["file_size"] for row in rows)
        summary.append(
            {
                "product": product,
                "files": len(rows),
                "total_bytes": total_bytes,
                "total_gib": round(total_bytes / (1024**3), 3),
            }
        )
        print(f"{product}: {len(rows)} files, {total_bytes / (1024**3):.3f} GiB")
        time.sleep(args.sleep)

    manifest_path = output_dir / f"mms1_{args.start}_to_{args.stop}_manifest.csv"
    summary_path = output_dir / f"mms1_{args.start}_to_{args.stop}_summary.json"
    write_csv(manifest_path, all_rows)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "start": args.start,
                "stop": args.stop,
                "products": args.products,
                "summary": summary,
                "manifest": str(manifest_path),
            },
            fh,
            indent=2,
        )
    print(json.dumps({"manifest": str(manifest_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
