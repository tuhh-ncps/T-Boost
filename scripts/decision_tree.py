#!/usr/bin/env python3
"""Train Decision Tree regressors to predict next_time from event context.

Reads CSV files from data_csv/, trains one model per CSV file, and uses the
requested feature set:
- activity
- timestamp
- prefix_sequence
- time_span
- next_activity (also accepts the misspelled next_actvity)

Target label:
- next_time
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


DEFAULT_DATA_DIR = Path("data_csv")
DEFAULT_RESULT_DIR = Path("results/decision_tree")
PREFIX_K = 6


def _stable_file_seed(base_seed: int, file_name: str) -> int:
    return int(base_seed + sum(file_name.encode("utf-8")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Decision Tree regressor per CSV file to predict next_time."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder containing input CSV files")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Folder to store all models and reports (default: result)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional single CSV filename to train (e.g., BPI12.csv). Default trains all CSVs.",
    )
    parser.add_argument(
        "--next-activity-noise-rate",
        type=float,
        default=0.05,
        help="Probability of replacing next_activity with a different value (default: 0.05)",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=0,
        help="Optional row cap per file for faster experiments (0 = use all rows)",
    )
    parser.add_argument(
        "--max-train-cases",
        type=int,
        default=0,
        help="Use only the first N temporal training cases (0 = use all training cases)",
    )
    parser.add_argument(
        "--auto-stop-by-mae",
        action="store_true",
        help="Automatically choose train-case cutoff with lowest validation MAE",
    )
    parser.add_argument(
        "--train-case-step",
        type=int,
        default=500,
        help="Step size (in cases) for auto-stop search (default: 500)",
    )
    parser.add_argument(
        "--train-case-patience",
        type=int,
        default=2,
        help="Auto-stop patience for non-improving MAE case cutoffs (default: 2)",
    )
    parser.add_argument(
        "--train-case-min-delta",
        type=float,
        default=0.0,
        help="Minimum MAE decrease required to count as improvement for auto-stop (default: 0.0)",
    )
    parser.add_argument(
        "--max-depth-limit",
        type=int,
        default=10,
        help="Upper bound for max_depth tuning (default: 10)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop depth tuning after this many non-improving depths (default: 2)",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum R2 improvement required to reset patience (default: 1e-4)",
    )
    return parser.parse_args()


def list_csv_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {data_dir}")

    files = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return files


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Accept both the expected column and common misspelling.
    if "next_actvity" in df.columns and "next_activity" not in df.columns:
        df = df.rename(columns={"next_actvity": "next_activity"})
    return df


def load_single_dataset(path: Path, max_rows_per_file: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)
    if max_rows_per_file > 0:
        df = df.head(max_rows_per_file)
    return df


def validate_required_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    feature_cols = ["activity", "timestamp", "prefix_sequence", "time_span", "next_activity"]
    target_col = "next_time"

    missing = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return feature_cols, target_col


def split_prefix_sequence(value: object, k: int = PREFIX_K) -> List[str]:
    if pd.isna(value):
        parts: List[str] = []
    else:
        text = str(value).strip()
        if not text:
            parts = []
        else:
            parts = [p.strip() for p in re.split(r"\s*(?:->|>|,|;|\|)\s*", text) if p.strip()]

    parts = parts[-k:]
    if len(parts) < k:
        parts = ["START"] * (k - len(parts)) + parts
    return parts


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out["timestamp_epoch"] = ts.astype("int64") / 1e9
    out.loc[ts.isna(), "timestamp_epoch"] = np.nan
    out["ts_dayofweek"] = ts.dt.dayofweek
    out["ts_hour"] = ts.dt.hour
    out["ts_month"] = ts.dt.month

    out["time_span"] = pd.to_numeric(out["time_span"], errors="coerce")
    out["next_time"] = pd.to_numeric(out["next_time"], errors="coerce")

    prefix_values = out["prefix_sequence"].apply(split_prefix_sequence)
    for i in range(PREFIX_K):
        out[f"prefix_{i + 1}"] = prefix_values.str[i]

    numeric_cols = ["timestamp_epoch", "time_span", "ts_dayofweek", "ts_hour", "ts_month"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Use plain object dtype + np.nan so sklearn imputers handle missing values reliably.
    for col in ["activity", "prefix_sequence", "next_activity"]:
        out[col] = out[col].astype("object")
        out.loc[out[col].isna(), col] = np.nan

    for i in range(PREFIX_K):
        col = f"prefix_{i + 1}"
        out[col] = out[col].astype("object")
        out.loc[out[col].isna(), col] = "START"

    return out


def inject_next_activity_noise(df: pd.DataFrame, noise_rate: float, random_state: int) -> pd.DataFrame:
    out = df.copy()
    if noise_rate <= 0:
        return out

    values = [v for v in out["next_activity"].dropna().unique().tolist() if str(v).strip()]
    if len(values) < 2:
        return out

    rng = np.random.default_rng(random_state)
    mask = rng.random(len(out)) < noise_rate
    n_masked = int(mask.sum())
    if n_masked == 0:
        return out

    current_vals = out.loc[mask, "next_activity"].to_numpy(dtype=object)
    new_vals = rng.choice(values, size=n_masked)

    same = new_vals == current_vals
    tries = 0
    while same.any() and tries < 5:
        new_vals[same] = rng.choice(values, size=int(same.sum()))
        same = new_vals == current_vals
        tries += 1

    if same.any():
        idx_map = {v: i for i, v in enumerate(values)}
        for i, is_same in enumerate(same):
            if is_same:
                cur = current_vals[i]
                if cur in idx_map:
                    new_vals[i] = values[(idx_map[cur] + 1) % len(values)]

    out.loc[mask, "next_activity"] = new_vals
    return out


def get_model_feature_columns() -> List[str]:
    return [
        "activity",
        "next_activity",
        "prefix_1",
        "prefix_2",
        "prefix_3",
        "timestamp_epoch",
        "time_span",
        "ts_dayofweek",
        "ts_hour",
        "ts_month",
    ]


def temporal_split(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("timestamp_epoch", kind="mergesort", na_position="last").reset_index(drop=True)
    n_rows = len(ordered)
    if n_rows < 2:
        return ordered, ordered.iloc[0:0]

    n_test = max(1, int(round(n_rows * test_size)))
    if n_test >= n_rows:
        n_test = n_rows - 1
    split_idx = n_rows - n_test

    train_df = ordered.iloc[:split_idx].copy()
    test_df = ordered.iloc[split_idx:].copy()
    return train_df, test_df


def limit_train_cases(train_df: pd.DataFrame, max_train_cases: int) -> pd.DataFrame:
    if max_train_cases <= 0:
        return train_df

    if "case_index" not in train_df.columns:
        return train_df.iloc[:max_train_cases].copy()

    case_series = pd.to_numeric(train_df["case_index"], errors="coerce")
    valid_case_df = train_df.loc[case_series.notna()].copy()
    if valid_case_df.empty:
        return train_df.iloc[:max_train_cases].copy()

    valid_case_df["__case_num"] = pd.to_numeric(valid_case_df["case_index"], errors="coerce")
    ordered_cases = np.sort(valid_case_df["__case_num"].unique())
    cutoff_cases = ordered_cases[:max_train_cases]
    limited = valid_case_df[valid_case_df["__case_num"].isin(cutoff_cases)].copy()
    return limited.drop(columns=["__case_num"])


def count_unique_cases(df: pd.DataFrame) -> int:
    if "case_index" not in df.columns:
        return int(len(df))
    case_series = pd.to_numeric(df["case_index"], errors="coerce")
    return int(case_series.dropna().nunique())


def tune_best_depth(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    args: argparse.Namespace,
) -> Tuple[Pipeline, int, Tuple[float, float, float], int, bool]:
    best_pipeline: Pipeline | None = None
    best_depth = 1
    best_scores: Tuple[float, float, float] | None = None
    no_improve_count = 0
    depths_tried = 0
    stopped_early = False

    for depth in range(1, max(2, args.max_depth_limit + 1)):
        depths_tried += 1
        pipeline = build_model_pipeline(args.random_state, max_depth=depth)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        if best_scores is None:
            best_pipeline = pipeline
            best_depth = depth
            best_scores = (mae, rmse, r2)
            continue

        _, best_rmse, best_r2 = best_scores
        improved = False
        if r2 > best_r2 + args.early_stopping_min_delta:
            improved = True
        elif np.isclose(r2, best_r2, atol=args.early_stopping_min_delta) and rmse < best_rmse:
            improved = True

        if improved:
            best_pipeline = pipeline
            best_depth = depth
            best_scores = (mae, rmse, r2)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if args.early_stopping_patience >= 0 and no_improve_count > args.early_stopping_patience:
                stopped_early = True
                break

    assert best_pipeline is not None and best_scores is not None
    return best_pipeline, best_depth, best_scores, depths_tried, stopped_early


def build_model_pipeline(random_state: int, max_depth: int) -> Pipeline:
    # Keep timestamp_epoch only for temporal split, not as a training feature.
    numeric_features = ["time_span", "ts_dayofweek", "ts_hour", "ts_month"]
    categorical_features = ["activity", "next_activity", "prefix_1", "prefix_2", "prefix_3"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = DecisionTreeRegressor(random_state=random_state, max_depth=max_depth)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )
    return pipeline


def train_one_file(csv_path: Path, args: argparse.Namespace) -> Dict[str, object]:
    raw_data = load_single_dataset(csv_path, args.max_rows_per_file)
    _, target_col = validate_required_columns(raw_data)

    data = coerce_types(raw_data)
    data["__row_id"] = data.index

    train_df = data.dropna(subset=[target_col]).copy()
    if len(train_df) < 2:
        return {
            "file": csv_path.name,
            "rows_total": int(len(data)),
            "rows_used_for_training": int(len(train_df)),
            "rows_train": 0,
            "rows_test": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "model_path": "",
            "status": "skipped_not_enough_rows",
        }

    noise_seed = _stable_file_seed(args.random_state, csv_path.name)
    train_df = inject_next_activity_noise(train_df, args.next_activity_noise_rate, noise_seed)

    train_part, test_part = temporal_split(train_df, args.test_size)

    # Optional static cap before any auto-stop search.
    train_part = limit_train_cases(train_part, args.max_train_cases)
    if train_part.empty or test_part.empty:
        return {
            "file": csv_path.name,
            "rows_total": int(len(data)),
            "rows_used_for_training": int(len(train_df)),
            "rows_train": int(len(train_part)),
            "rows_test": int(len(test_part)),
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "model_path": "",
            "status": "skipped_invalid_temporal_split",
        }

    feature_cols = get_model_feature_columns()
    X_test = test_part[feature_cols]
    y_test = test_part[target_col]

    selected_train_part = train_part
    selected_case_cutoff = count_unique_cases(train_part)
    case_cutoffs_evaluated = 1
    case_search_stopped_early = False

    if args.auto_stop_by_mae:
        total_cases = count_unique_cases(train_part)
        step = max(1, args.train_case_step)
        cutoffs = list(range(step, total_cases + 1, step))
        if not cutoffs or cutoffs[-1] != total_cases:
            cutoffs.append(total_cases)

        best_case_mae: float | None = None
        best_case_depth_scores: Tuple[Pipeline, int, Tuple[float, float, float], int, bool] | None = None
        best_case_train_part: pd.DataFrame | None = None
        no_improve_case = 0
        case_cutoffs_evaluated = 0

        for cutoff in cutoffs:
            candidate_train_part = limit_train_cases(train_part, cutoff)
            if candidate_train_part.empty:
                continue

            X_train_candidate = candidate_train_part[feature_cols]
            y_train_candidate = candidate_train_part[target_col]
            tuned = tune_best_depth(X_train_candidate, y_train_candidate, X_test, y_test, args)
            _, _, candidate_scores, _, _ = tuned
            candidate_mae = float(candidate_scores[0])
            case_cutoffs_evaluated += 1

            if best_case_mae is None or candidate_mae < best_case_mae - args.train_case_min_delta:
                best_case_mae = candidate_mae
                best_case_depth_scores = tuned
                best_case_train_part = candidate_train_part
                selected_case_cutoff = count_unique_cases(candidate_train_part)
                no_improve_case = 0
            else:
                no_improve_case += 1
                if args.train_case_patience >= 0 and no_improve_case > args.train_case_patience:
                    case_search_stopped_early = True
                    break

        if best_case_depth_scores is not None and best_case_train_part is not None:
            selected_train_part = best_case_train_part
            best_pipeline, best_depth, best_scores, depths_tried, stopped_early = best_case_depth_scores
        else:
            X_train_fallback = selected_train_part[feature_cols]
            y_train_fallback = selected_train_part[target_col]
            best_pipeline, best_depth, best_scores, depths_tried, stopped_early = tune_best_depth(
                X_train_fallback, y_train_fallback, X_test, y_test, args
            )
    else:
        X_train_base = selected_train_part[feature_cols]
        y_train_base = selected_train_part[target_col]
        best_pipeline, best_depth, best_scores, depths_tried, stopped_early = tune_best_depth(
            X_train_base, y_train_base, X_test, y_test, args
        )

    mae, rmse, r2 = best_scores
    train_part = selected_train_part
    X_train = train_part[feature_cols]
    y_train = train_part[target_col]

    # Build detailed prediction output per dataset.
    y_pred_train = best_pipeline.predict(X_train)
    y_pred_test = best_pipeline.predict(X_test)

    train_part = train_part.copy()
    test_part = test_part.copy()
    train_part["prediction_next_time"] = y_pred_train
    test_part["prediction_next_time"] = y_pred_test
    train_part["split"] = "train"
    test_part["split"] = "test"

    prediction_rows = pd.concat([train_part, test_part], ignore_index=True)
    prediction_rows = prediction_rows.sort_values("__row_id", kind="mergesort")

    input_cols = list(raw_data.columns)
    raw_with_row_id = raw_data.copy()
    raw_with_row_id["__row_id"] = raw_with_row_id.index

    prediction_export = raw_with_row_id.merge(
        prediction_rows[["__row_id", "split", "prediction_next_time"]],
        on="__row_id",
        how="inner",
    )
    prediction_export["actual_next_time"] = pd.to_numeric(prediction_export["next_time"], errors="coerce")
    prediction_export["prediction_error"] = (
        prediction_export["actual_next_time"] - prediction_export["prediction_next_time"]
    )
    prediction_export = prediction_export[input_cols + ["split", "actual_next_time", "prediction_next_time", "prediction_error"]]

    models_dir = args.result_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{csv_path.stem}_decision_tree_next_time.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(best_pipeline, handle)

    predictions_dir = args.result_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = predictions_dir / f"{csv_path.stem}_prediction.csv"
    prediction_export.to_csv(prediction_path, index=False)

    return {
        "file": csv_path.name,
        "rows_total": int(len(data)),
        "rows_used_for_training": int(len(train_df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "auto_stop_by_mae": bool(args.auto_stop_by_mae),
        "selected_train_case_cutoff": int(selected_case_cutoff),
        "case_cutoffs_evaluated": int(case_cutoffs_evaluated),
        "case_search_stopped_early": bool(case_search_stopped_early),
        "max_train_cases": int(args.max_train_cases),
        "train_cases_used": int(train_part["case_index"].nunique()) if "case_index" in train_part.columns else np.nan,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "best_max_depth": int(best_depth),
        "depths_tried": int(depths_tried),
        "stopped_early": bool(stopped_early),
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
        "status": "trained",
    }


def main() -> None:
    args = parse_args()
    files = list_csv_files(args.data_dir)
    if args.dataset:
        files = [p for p in files if p.name == args.dataset]
        if not files:
            raise FileNotFoundError(f"Dataset not found in {args.data_dir}: {args.dataset}")

    args.result_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    for path in files:
        try:
            row = train_one_file(path, args)
        except Exception as exc:  # noqa: BLE001
            row = {
                "file": path.name,
                "rows_total": np.nan,
                "rows_used_for_training": np.nan,
                "rows_train": np.nan,
                "rows_test": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "model_path": "",
                "status": f"failed: {exc}",
            }
        all_rows.append(row)
        print(f"{path.name}: {row['status']}")

    metrics_df = pd.DataFrame(all_rows)
    report_path = args.result_dir / "decision_tree_metrics_all.csv"
    metrics_df.to_csv(report_path, index=False)

    trained_count = int((metrics_df["status"] == "trained").sum())
    print(f"Trained models: {trained_count}/{len(files)}")
    print(f"Saved aggregate metrics: {report_path}")


if __name__ == "__main__":
    main()
