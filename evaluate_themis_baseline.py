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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


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

RAW_FGM_FEATURES = {
    "Bx",
    "By",
    "Bz",
    "B_total",
    "B_std_1m",
    "Bz_std_1m",
    "Bz_sign_change_1m",
    "B_total_slope_1m",
}

PHYSICS_FGM_FEATURES = {
    "dip_angle_deg",
    "stretching_index",
    "stretching_index_log",
    "magnetic_pressure_proxy",
    "sheet_proximity",
    "Bz_trend_1m",
    "Bz_trend_5m",
    "Bz_trend_15m",
    "Bx_var_3m",
    "Bz_var_3m",
    "B_total_var_3m",
    "P_mag_ratio_30m",
}

FBK_FEATURES = {
    "scm1_ch0",
    "scm1_ch1",
    "scm1_ch2",
    "scm1_mean",
    "scm1_max",
    "scm1_low_high_ratio",
    "scm1_mean_slope_1m",
    "scm1_max_slope_1m",
    "scm1_max_rollmax_1m",
    "scm2_ch0",
    "scm2_ch1",
    "scm2_ch2",
    "scm2_mean",
    "scm2_max",
    "scm2_low_high_ratio",
    "scm2_mean_slope_1m",
    "scm2_max_slope_1m",
    "scm2_max_rollmax_1m",
    "scm3_ch0",
    "scm3_ch1",
    "scm3_ch2",
    "scm3_mean",
    "scm3_max",
    "scm3_low_high_ratio",
    "scm3_mean_slope_1m",
    "scm3_max_slope_1m",
    "scm3_max_rollmax_1m",
    "edc12_ch0",
    "edc12_ch1",
    "edc12_ch2",
    "edc12_mean",
    "edc12_max",
    "edc12_low_high_ratio",
    "edc12_mean_slope_1m",
    "edc12_max_slope_1m",
    "edc12_max_rollmax_1m",
}

CLIP_LIMITS = {
    "stretching_index": 100.0,
    "stretching_index_log": np.log10(101.0),
    "Bz_trend_1m": 50.0,
    "Bz_trend_5m": 100.0,
    "Bz_trend_15m": 150.0,
    "Bx_var_3m": 1_000.0,
    "Bz_var_3m": 1_000.0,
    "B_total_var_3m": 1_000.0,
    "P_mag_ratio_30m": 1.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run THEMIS BBF early-prediction ablation baselines.")
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


def apply_feature_clipping(df: pd.DataFrame) -> pd.DataFrame:
    """Clip ratio and trend features at physically plausible upper bounds."""
    out = df.copy()
    for column, limit in CLIP_LIMITS.items():
        if column in out.columns:
            out[column] = out[column].clip(lower=-limit if "trend" in column else None, upper=limit)
    return out


def build_feature_sets(columns: list[str]) -> dict[str, list[str]]:
    allowed = {c for c in columns if c not in LEAKAGE_COLUMNS}

    raw_fgm = sorted(RAW_FGM_FEATURES & allowed)
    physics_fgm = sorted(PHYSICS_FGM_FEATURES & allowed)
    fbk = sorted(FBK_FEATURES & allowed)

    return {
        "raw_fgm": raw_fgm,
        "raw_fgm_fbk": raw_fgm + fbk,
        "physics_only": physics_fgm,
        "physics_fbk": physics_fgm + fbk,
        "hybrid_fgm": raw_fgm + physics_fgm,
        "hybrid_fgm_fbk": raw_fgm + physics_fgm + fbk,
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


def metric_row(split: str, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    return {
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


def positive_weight(y: np.ndarray) -> float:
    positives = max(int(y.sum()), 1)
    negatives = max(len(y) - positives, 1)
    return negatives / positives


def build_models(y_train: np.ndarray) -> dict[str, Pipeline]:
    models = {
        "logistic_l2": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=3000, solver="lbfgs")),
            ]
        ),
        "logistic_l1": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", penalty="l1", C=0.15, max_iter=4000, solver="liblinear")),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.0,
                        min_child_weight=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=positive_weight(y_train),
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    return models


def extract_importance_rows(model_name: str, fitted_pipeline: Pipeline, features: list[str], feature_set: str) -> list[dict]:
    estimator = fitted_pipeline.named_steps["model"]
    rows = []
    if model_name == "logistic_l1":
        coefs = estimator.coef_[0]
        for feature, value in zip(features, coefs):
            rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "feature": feature,
                    "importance": float(abs(value)),
                    "signed_value": float(value),
                }
            )
    elif hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        for feature, value in zip(features, importances):
            rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "feature": feature,
                    "importance": float(value),
                    "signed_value": float(value),
                }
            )
    return rows


def run_model(name: str, model: Pipeline, features: list[str], train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[list[dict], list[dict]]:
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

    metrics = []
    for split, y, scores in [("val", y_val, val_scores), ("test", y_test, test_scores)]:
        row = metric_row(split, y, scores, threshold)
        row.update({"model": name, "feature_set": feature_set, "num_features": len(features)})
        metrics.append(row)

    importance_rows = extract_importance_rows(name, model, features, feature_set)
    return metrics, importance_rows


def summarize_top_features(importance_df: pd.DataFrame) -> dict:
    summary = {}
    if importance_df.empty:
        return summary
    for (model_name, feature_set), chunk in importance_df.groupby(["model", "feature_set"]):
        top = chunk.sort_values("importance", ascending=False).head(10)
        summary[f"{model_name}:{feature_set}"] = [
            {"feature": row["feature"], "importance": round(float(row["importance"]), 6)}
            for _, row in top.iterrows()
        ]
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.dataset).sort_values("time").reset_index(drop=True)
    df = apply_feature_clipping(df)

    stride_rows = max(1, args.anchor_stride_seconds // args.bin_seconds)
    anchors = sample_clean_anchors(df, stride_rows)
    train, val, test = split_by_date(anchors)

    feature_sets = build_feature_sets(list(df.columns))
    metric_rows = []
    importance_rows = []

    for feature_set_name, features in feature_sets.items():
        if not features:
            continue
        models = build_models(train["future_bbf_5m"].to_numpy(dtype=int))
        for model_name, model in models.items():
            rows, importances = run_model(model_name, model, features, train, val, test, feature_set_name)
            metric_rows.extend(rows)
            importance_rows.extend(importances)

    results = pd.DataFrame(metric_rows)
    importance_df = pd.DataFrame(importance_rows)

    results.to_csv(output_dir / "baseline_metrics.csv", index=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

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
        "feature_sets": feature_sets,
        "clip_limits": CLIP_LIMITS,
        "models_run": sorted(results["model"].unique()) if not results.empty else [],
        "metrics": str(output_dir / "baseline_metrics.csv"),
        "feature_importance": str(output_dir / "feature_importance.csv"),
        "top_features": summarize_top_features(importance_df),
    }
    with (output_dir / "baseline_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(results.to_string(index=False))
    if not importance_df.empty:
        print("\nTop feature importance by model/set:")
        for key, value in summary["top_features"].items():
            print(f"{key}: {value}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
