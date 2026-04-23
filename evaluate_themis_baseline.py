import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DATASET = Path(r"C:\Magnetic\datasets\tha_2017-01-01_60d_vperp_200_20_filled\tha_early_features.parquet")
DEFAULT_OUTPUT_DIR = Path(r"C:\Magnetic\datasets\tha_2017-01-01_60d_vperp_200_20_filled\baseline")

LEAKAGE_COLUMNS = {
    "time",
    "date",
    "Vx",
    "Vy",
    "Vz",
    "Vx_good",
    "Vy_good",
    "Vz_good",
    "V_abs",
    "V_perp",
    "mom_quality",
    "bbf_label",
    "future_bbf_5m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight THEMIS BBF early-prediction baselines.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--anchor-stride-seconds", type=int, default=60)
    parser.add_argument("--bin-seconds", type=int, default=5)
    return parser.parse_args()


def split_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["date"].unique())
    n = len(dates)
    train_dates = set(dates[: int(n * 0.70)])
    val_dates = set(dates[int(n * 0.70) : int(n * 0.85)])
    test_dates = set(dates[int(n * 0.85) :])
    return (
        df[df["date"].isin(train_dates)].copy(),
        df[df["date"].isin(val_dates)].copy(),
        df[df["date"].isin(test_dates)].copy(),
    )


def sample_clean_anchors(df: pd.DataFrame, stride_rows: int) -> pd.DataFrame:
    anchors = df.iloc[::stride_rows].copy()
    return anchors[anchors["bbf_label"].eq(0)].copy()


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    fgm = [c for c in columns if c.startswith("B")]
    fbk = [c for c in columns if c.startswith("scm") or c.startswith("edc")]
    allowed = [c for c in columns if c not in LEAKAGE_COLUMNS]
    return {
        "fgm_only": [c for c in fgm if c in allowed],
        "fgm_fbk": [c for c in fgm + fbk if c in allowed],
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


def metric_row(split: str, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    row = {
        "split": split,
        "samples": int(len(y_true)),
        "positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "pr_auc": float(average_precision_score(y_true, scores)) if y_true.sum() else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, scores)) if 0 < y_true.sum() < len(y_true) else float("nan"),
    }
    return row


def run_model(name: str, model, features: list[str], train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    x_train = train[features]
    y_train = train["future_bbf_5m"].to_numpy(dtype=int)
    x_val = val[features]
    y_val = val["future_bbf_5m"].to_numpy(dtype=int)
    x_test = test[features]
    y_test = test["future_bbf_5m"].to_numpy(dtype=int)

    model.fit(x_train, y_train)
    val_scores = model.predict_proba(x_val)[:, 1]
    threshold = choose_threshold(y_val, val_scores)
    test_scores = model.predict_proba(x_test)[:, 1]

    rows = []
    for split, y, scores in [("val", y_val, val_scores), ("test", y_test, test_scores)]:
        row = metric_row(split, y, scores, threshold)
        row.update({"model": name, "feature_set": features[0].split("_")[0] if features else "none", "num_features": len(features)})
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.dataset).sort_values("time").reset_index(drop=True)
    stride_rows = max(1, args.anchor_stride_seconds // args.bin_seconds)
    anchors = sample_clean_anchors(df, stride_rows)
    train, val, test = split_by_date(anchors)

    sets = feature_sets(list(df.columns))
    rows = []
    for set_name, features in sets.items():
        models = {
            "logistic_regression": make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs"),
            ),
            "random_forest": make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        }
        for model_name, model in models.items():
            for row in run_model(model_name, model, features, train, val, test):
                row["feature_set"] = set_name
                rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "baseline_metrics.csv", index=False)
    summary = {
        "dataset": str(args.dataset),
        "anchor_stride_seconds": args.anchor_stride_seconds,
        "samples": {
            "all_clean_anchors": int(len(anchors)),
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
            "train_positives": int(train["future_bbf_5m"].sum()),
            "val_positives": int(val["future_bbf_5m"].sum()),
            "test_positives": int(test["future_bbf_5m"].sum()),
        },
        "feature_sets": {name: cols for name, cols in sets.items()},
        "metrics": str(output_dir / "baseline_metrics.csv"),
    }
    with (output_dir / "baseline_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(results.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
