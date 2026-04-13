import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET_DIR = Path(r"C:\Magnetic\cases\2017-07-29_mms1_earthward_bbf\early_bbf_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create feature sanity checks and simple aggregate baselines for the early BBF pilot dataset."
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--drop-inside-bbf",
        action="store_true",
        help="Drop anchors where a BBF is already active at prediction time.",
    )
    return parser.parse_args()


def load_schema(dataset_dir: Path) -> list[str]:
    with (dataset_dir / "feature_schema.json").open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    return list(schema["sequence_feature_columns"])


def load_sequences(dataset_dir: Path, drop_inside_bbf: bool) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    index = pd.read_csv(dataset_dir / "sequence_index.csv", parse_dates=["anchor_time"])
    if drop_inside_bbf and "inside_bbf_at_anchor" in index:
        index = index[index["inside_bbf_at_anchor"] == 0].copy()

    arrays = {}
    for row in index.itertuples(index=False):
        data = np.load(dataset_dir / row.path)
        arrays[row.path] = np.asarray(data["X"], dtype=np.float64)
    return index.reset_index(drop=True), arrays


def feature_sanity_table(index: pd.DataFrame, arrays: dict[str, np.ndarray], feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feature_idx, feature in enumerate(feature_cols):
        for label_value, label_group in index.groupby("target_bbf_within_5min", dropna=False):
            values = []
            missing = 0
            total = 0
            for path in label_group["path"]:
                x = arrays[path][:, feature_idx]
                missing += int(np.isnan(x).sum())
                total += int(x.size)
                values.append(x)
            if values:
                flat = np.concatenate(values)
                finite = flat[np.isfinite(flat)]
            else:
                finite = np.asarray([], dtype=np.float64)
            rows.append(
                {
                    "feature": feature,
                    "target_bbf_within_5min": int(label_value),
                    "samples": int(len(label_group)),
                    "mean": float(np.mean(finite)) if finite.size else np.nan,
                    "std": float(np.std(finite)) if finite.size else np.nan,
                    "min": float(np.min(finite)) if finite.size else np.nan,
                    "max": float(np.max(finite)) if finite.size else np.nan,
                    "missing_rate": float(missing / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def aggregate_features(index: pd.DataFrame, arrays: dict[str, np.ndarray], feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for row in index.itertuples(index=False):
        x = arrays[row.path]
        out = {
            "path": row.path,
            "anchor_time": row.anchor_time,
            "target_bbf_within_5min": int(row.target_bbf_within_5min),
            "inside_bbf_at_anchor": int(getattr(row, "inside_bbf_at_anchor", 0)),
        }
        for feature_idx, feature in enumerate(feature_cols):
            values = x[:, feature_idx]
            out[f"{feature}__mean"] = float(np.nanmean(values))
            out[f"{feature}__std"] = float(np.nanstd(values))
            out[f"{feature}__min"] = float(np.nanmin(values))
            out[f"{feature}__max"] = float(np.nanmax(values))
            out[f"{feature}__last"] = float(values[-1])
            out[f"{feature}__delta"] = float(values[-1] - values[0])
        rows.append(out)
    return pd.DataFrame(rows)


def evaluate_baselines(agg: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in agg.columns if "__" in col]
    y = agg["target_bbf_within_5min"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2 or len(y) < 4:
        return pd.DataFrame(
            [
                {
                    "model": "not_run",
                    "reason": "Need at least four samples and both classes.",
                    "samples": int(len(y)),
                    "positive_samples": int(y.sum()),
                }
            ]
        )

    try:
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "model": "sklearn_baselines_not_run",
                    "reason": f"{exc.__class__.__name__}: {exc}",
                    "samples": int(len(y)),
                    "positive_samples": int(y.sum()),
                }
            ]
        )

    models = [
        ("majority", DummyClassifier(strategy="most_frequent")),
        (
            "logistic_regression",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42, class_weight="balanced"),
        ),
    ]

    try:
        from xgboost import XGBClassifier

        models.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=30,
                    max_depth=2,
                    learning_rate=0.1,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                ),
            )
        )
    except Exception as exc:
        models.append(("xgboost_not_run", f"{exc.__class__.__name__}: {exc}"))

    x = agg[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    cv = LeaveOneOut()
    rows = []
    for name, model in models:
        if isinstance(model, str):
            rows.append({"model": name, "reason": model, "samples": int(len(y)), "positive_samples": int(y.sum())})
            continue
        try:
            pred = cross_val_predict(model, x, y, cv=cv, method="predict")
            row = {
                "model": name,
                "cv": "leave_one_out",
                "samples": int(len(y)),
                "positive_samples": int(y.sum()),
                "accuracy": float(accuracy_score(y, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "f1": float(f1_score(y, pred, zero_division=0)),
            }
            try:
                proba = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
                row["roc_auc"] = float(roc_auc_score(y, proba))
            except Exception:
                row["roc_auc"] = np.nan
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "model": name,
                    "reason": f"{exc.__class__.__name__}: {exc}",
                    "samples": int(len(y)),
                    "positive_samples": int(y.sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = load_schema(dataset_dir)
    index, arrays = load_sequences(dataset_dir, args.drop_inside_bbf)
    sanity = feature_sanity_table(index, arrays, feature_cols)
    agg = aggregate_features(index, arrays, feature_cols)
    baseline = evaluate_baselines(agg)

    suffix = "_clean" if args.drop_inside_bbf else ""
    sanity.to_csv(output_dir / f"feature_sanity_check{suffix}.csv", index=False)
    agg.to_csv(output_dir / f"window_aggregate_features{suffix}.csv", index=False)
    baseline.to_csv(output_dir / f"baseline_results{suffix}.csv", index=False)

    summary = {
        "dataset_dir": str(dataset_dir),
        "drop_inside_bbf": bool(args.drop_inside_bbf),
        "samples": int(len(index)),
        "positive_samples": int(index["target_bbf_within_5min"].sum()) if len(index) else 0,
        "feature_count": int(len(feature_cols)),
        "outputs": [
            str(output_dir / f"feature_sanity_check{suffix}.csv"),
            str(output_dir / f"window_aggregate_features{suffix}.csv"),
            str(output_dir / f"baseline_results{suffix}.csv"),
        ],
    }
    with (output_dir / f"evaluation_summary{suffix}.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(baseline.to_string(index=False))


if __name__ == "__main__":
    main()
