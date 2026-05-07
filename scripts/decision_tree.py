#!/usr/bin/env python3
"""Train Decision Tree regressors to predict next_time from event context.

Reads CSV files from data_csv/, trains one model per CSV file, and uses the
requested feature set:
- activity
- timestamp
- prefix_sequence
- time_span
- next_activity (also accepts the misspelled next_actvity)
- predicted_time_regime_q1_pct
- predicted_time_regime_q2_pct
- predicted_time_regime_q3_pct

Target label:
- next_time

Input selection:
    Use --data-dir for the split CSV folder. Optionally use --dataset to train
    only one dataset stem.

Example:
    python scripts/decision_tree.py --data-dir data_csv --result-dir results/decision_tree

Usage:
    --data-dir PATH           Folder containing input CSV files
    --result-dir PATH         Folder for models and reports
    --dataset NAME            Optional single CSV file to train
    --preceding-k N           Number of preceding activities to encode
    --max-depth-limit N       Upper bound for max_depth tuning
    --use-next-activity       Include next_activity as model input
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
DEFAULT_PRECEDING_K = 3
SECONDS_PER_DAY = 86400.0
SOFT_REGIME_COLUMNS = [
    "predicted_time_regime_q1_pct",
    "predicted_time_regime_q2_pct",
    "predicted_time_regime_q3_pct",
]


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
        "--preceding-k",
        type=int,
        default=DEFAULT_PRECEDING_K,
        help="Number of preceding activities to encode as prefix features (default: 3)",
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


def strip_known_suffixes(stem: str) -> str:
    for suffix in ("_train_with_time_regime", "_test_with_time_regime", "_train", "_test"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def build_dataset_pairs(data_dir: Path, dataset: str) -> List[Tuple[str, Path, Path]]:
    files = list_csv_files(data_dir)
    by_name = {p.name: p for p in files}
    pairs: List[Tuple[str, Path, Path]] = []

    if dataset:
        stem = strip_known_suffixes(Path(dataset).stem)
        train_name = f"{stem}_train_with_time_regime.csv"
        if train_name not in by_name:
            train_name = f"{stem}_train.csv"
        test_name = f"{stem}_test_with_time_regime.csv"
        if test_name not in by_name:
            test_name = f"{stem}_test.csv"
        if train_name not in by_name:
            raise FileNotFoundError(f"Train split not found in {data_dir}: {train_name}")
        if test_name not in by_name:
            raise FileNotFoundError(f"Test split not found in {data_dir}: {test_name}")
        return [(stem, by_name[train_name], by_name[test_name])]

    for path in files:
        if not path.stem.endswith("_train"):
            continue
        stem = strip_known_suffixes(path.stem)
        soft_train_name = f"{stem}_train_with_time_regime.csv"
        if soft_train_name in by_name:
            path = by_name[soft_train_name]
        test_name = f"{stem}_test_with_time_regime.csv"
        test_path = by_name.get(test_name)
        if test_path is None:
            test_path = by_name.get(f"{stem}_test.csv")
        if test_path is None:
            continue
        pairs.append((stem, path, test_path))

    if not pairs:
        raise FileNotFoundError(
            f"No train/test pairs found in {data_dir}. Expected <dataset>_train.csv and <dataset>_test_with_time_regime.csv"
        )

    return sorted(pairs, key=lambda item: item[0])


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


def has_soft_regime_columns(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in SOFT_REGIME_COLUMNS)


def split_prefix_sequence(value: object, k: int) -> List[str]:
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


def coerce_types(df: pd.DataFrame, preceding_k: int) -> pd.DataFrame:
    out = df.copy()

    ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out["timestamp_epoch"] = ts.astype("int64") / 1e9
    out.loc[ts.isna(), "timestamp_epoch"] = np.nan
    out["ts_dayofweek"] = ts.dt.dayofweek
    out["ts_hour"] = ts.dt.hour
    out["ts_month"] = ts.dt.month

    out["time_span"] = pd.to_numeric(out["time_span"], errors="coerce")
    out["next_time"] = pd.to_numeric(out["next_time"], errors="coerce")

    for col in SOFT_REGIME_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    prefix_values = out["prefix_sequence"].apply(lambda value: split_prefix_sequence(value, k=preceding_k))
    for i in range(preceding_k):
        out[f"prefix_{i + 1}"] = prefix_values.str[i]

    numeric_cols = ["timestamp_epoch", "time_span", "ts_dayofweek", "ts_hour", "ts_month"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Use plain object dtype + np.nan so sklearn imputers handle missing values reliably.
    for col in ["activity", "prefix_sequence", "next_activity", "time_regime"]:
        out[col] = out[col].astype("object")
        out.loc[out[col].isna(), col] = np.nan

    for i in range(preceding_k):
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


def get_model_feature_columns(preceding_k: int, use_soft_regime: bool) -> List[str]:
    cols = ["activity"]
    if use_soft_regime:
        cols.extend(SOFT_REGIME_COLUMNS)
    else:
        cols.append("time_regime")
    cols.extend([f"prefix_{i + 1}" for i in range(preceding_k)])
    cols.extend(["timestamp_epoch", "time_span", "ts_dayofweek", "ts_hour", "ts_month"])
    return cols


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
    use_soft_regime: bool,
) -> Tuple[Pipeline, int, Tuple[float, float, float], int, bool]:
    best_pipeline: Pipeline | None = None
    best_depth = 1
    best_scores: Tuple[float, float, float] | None = None
    no_improve_count = 0
    depths_tried = 0
    stopped_early = False

    for depth in range(1, max(2, args.max_depth_limit + 1)):
        depths_tried += 1
        pipeline = build_model_pipeline(
            random_state=args.random_state,
            max_depth=depth,
            preceding_k=args.preceding_k,
            use_soft_regime=use_soft_regime,
        )
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


def build_model_pipeline(
    random_state: int,
    max_depth: int,
    preceding_k: int,
    use_soft_regime: bool,
) -> Pipeline:
    # Keep timestamp_epoch only for temporal split, not as a training feature.
    numeric_features = ["time_span", "ts_dayofweek", "ts_hour", "ts_month"]
    if use_soft_regime:
        numeric_features = [*numeric_features, *SOFT_REGIME_COLUMNS]
        categorical_features = ["activity"]
    else:
        categorical_features = ["activity", "time_regime"]
    categorical_features.extend([f"prefix_{i + 1}" for i in range(preceding_k)])

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


def train_one_pair(train_csv_path: Path, test_csv_path: Path, dataset_stem: str, args: argparse.Namespace) -> Dict[str, object]:
    raw_train_data = load_single_dataset(train_csv_path, args.max_rows_per_file)
    raw_test_data = load_single_dataset(test_csv_path, args.max_rows_per_file)

    use_soft_regime = has_soft_regime_columns(raw_train_data) and has_soft_regime_columns(raw_test_data)
    regime_feature_mode = "soft_probabilities" if use_soft_regime else "hard_label"
    regime_feature_columns = ",".join(SOFT_REGIME_COLUMNS if use_soft_regime else ["time_regime"])

    # For pipeline usage, prefer inferred/predicted regime in test split when available.
    if "predicted_time_regime" in raw_test_data.columns:
        raw_test_data = raw_test_data.copy()
        raw_test_data["time_regime"] = raw_test_data["predicted_time_regime"]

    if not use_soft_regime:
        missing_regime = [col for col in ["time_regime"] if col not in raw_train_data.columns or col not in raw_test_data.columns]
        if missing_regime:
            raise ValueError(f"Missing required time-regime columns for legacy fallback: {missing_regime}")

    _, target_col = validate_required_columns(raw_train_data)
    validate_required_columns(raw_test_data)

    if args.preceding_k <= 0:
        raise ValueError("--preceding-k must be a positive integer")

    train_data = coerce_types(raw_train_data, preceding_k=args.preceding_k)
    test_data = coerce_types(raw_test_data, preceding_k=args.preceding_k)
    train_data["__row_id"] = train_data.index
    test_data["__row_id"] = test_data.index

    train_df = train_data.dropna(subset=[target_col]).copy()
    test_df = test_data.dropna(subset=[target_col]).copy()
    if len(train_df) < 2 or len(test_df) < 1:
        return {
            "file": f"{dataset_stem}.csv",
            "train_file": train_csv_path.name,
            "test_file": test_csv_path.name,
            "rows_total": int(len(train_data) + len(test_data)),
            "rows_used_for_training": int(len(train_df) + len(test_df)),
            "rows_train": 0,
            "rows_test": 0,
            "mae": np.nan,
            "mae_days": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "model_path": "",
            "status": "skipped_not_enough_rows",
        }
    train_part = limit_train_cases(train_df, args.max_train_cases)
    test_part = test_df
    if train_part.empty or test_part.empty:
        return {
            "file": f"{dataset_stem}.csv",
            "train_file": train_csv_path.name,
            "test_file": test_csv_path.name,
            "rows_total": int(len(train_data) + len(test_data)),
            "rows_used_for_training": int(len(train_df) + len(test_df)),
            "rows_train": int(len(train_part)),
            "rows_test": int(len(test_part)),
            "mae": np.nan,
            "mae_days": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "model_path": "",
            "status": "skipped_invalid_split_files",
        }

    feature_cols = get_model_feature_columns(
        preceding_k=args.preceding_k,
        use_soft_regime=use_soft_regime,
    )
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
            tuned = tune_best_depth(
                X_train_candidate,
                y_train_candidate,
                X_test,
                y_test,
                args,
                use_soft_regime=use_soft_regime,
            )
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
                X_train_fallback,
                y_train_fallback,
                X_test,
                y_test,
                args,
                use_soft_regime=use_soft_regime,
            )
    else:
        X_train_base = selected_train_part[feature_cols]
        y_train_base = selected_train_part[target_col]
        best_pipeline, best_depth, best_scores, depths_tried, stopped_early = tune_best_depth(
            X_train_base,
            y_train_base,
            X_test,
            y_test,
            args,
            use_soft_regime=use_soft_regime,
        )

    mae, rmse, r2 = best_scores
    mae_days = float(mae) / SECONDS_PER_DAY
    train_part = selected_train_part
    X_train = train_part[feature_cols]

    # Build detailed prediction output per dataset.
    y_pred_train = best_pipeline.predict(X_train)
    y_pred_test = best_pipeline.predict(X_test)

    train_part = train_part.copy()
    test_part = test_part.copy()
    train_part["prediction_next_time"] = y_pred_train
    test_part["prediction_next_time"] = y_pred_test
    train_part["split"] = "train"
    test_part["split"] = "test"
    train_part["regime_feature_mode"] = regime_feature_mode
    test_part["regime_feature_mode"] = regime_feature_mode
    train_part["regime_feature_columns"] = regime_feature_columns
    test_part["regime_feature_columns"] = regime_feature_columns

    prediction_rows = pd.concat([train_part, test_part], ignore_index=True)
    prediction_rows = prediction_rows.sort_values("__row_id", kind="mergesort")

    train_input_cols = list(raw_train_data.columns)
    raw_train_with_row_id = raw_train_data.copy()
    raw_train_with_row_id["__row_id"] = raw_train_with_row_id.index
    train_export = raw_train_with_row_id.merge(
        train_part[["__row_id", "split", "prediction_next_time", "regime_feature_mode", "regime_feature_columns"]],
        on="__row_id",
        how="inner",
    )
    train_export["actual_next_time"] = pd.to_numeric(train_export["next_time"], errors="coerce")
    train_export["prediction_error"] = train_export["actual_next_time"] - train_export["prediction_next_time"]
    train_export = train_export[
        train_input_cols
        + [
            "split",
            "regime_feature_mode",
            "regime_feature_columns",
            "actual_next_time",
            "prediction_next_time",
            "prediction_error",
        ]
    ]

    test_input_cols = list(raw_test_data.columns)
    raw_test_with_row_id = raw_test_data.copy()
    raw_test_with_row_id["__row_id"] = raw_test_with_row_id.index
    test_export = raw_test_with_row_id.merge(
        test_part[["__row_id", "split", "prediction_next_time", "regime_feature_mode", "regime_feature_columns"]],
        on="__row_id",
        how="inner",
    )
    test_export["actual_next_time"] = pd.to_numeric(test_export["next_time"], errors="coerce")
    test_export["prediction_error"] = test_export["actual_next_time"] - test_export["prediction_next_time"]
    test_export = test_export[
        test_input_cols
        + [
            "split",
            "regime_feature_mode",
            "regime_feature_columns",
            "actual_next_time",
            "prediction_next_time",
            "prediction_error",
        ]
    ]

    prediction_export = pd.concat([train_export, test_export], ignore_index=True)

    models_dir = args.result_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{dataset_stem}_decision_tree_next_time.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(best_pipeline, handle)

    predictions_dir = args.result_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = predictions_dir / f"{dataset_stem}_prediction.csv"
    prediction_export.to_csv(prediction_path, index=False)

    return {
        "file": f"{dataset_stem}.csv",
        "train_file": train_csv_path.name,
        "test_file": test_csv_path.name,
        "rows_total": int(len(train_data) + len(test_data)),
        "rows_used_for_training": int(len(train_df) + len(test_df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "auto_stop_by_mae": bool(args.auto_stop_by_mae),
        "selected_train_case_cutoff": int(selected_case_cutoff),
        "case_cutoffs_evaluated": int(case_cutoffs_evaluated),
        "case_search_stopped_early": bool(case_search_stopped_early),
        "max_train_cases": int(args.max_train_cases),
        "preceding_k": int(args.preceding_k),
        "regime_feature_mode": regime_feature_mode,
        "regime_feature_columns": regime_feature_columns,
        "train_cases_used": int(train_part["case_index"].nunique()) if "case_index" in train_part.columns else np.nan,
        "mae": float(mae),
        "mae_days": float(mae_days),
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
    pairs = build_dataset_pairs(args.data_dir, args.dataset)

    args.result_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    for dataset_stem, train_path, test_path in pairs:
        try:
            row = train_one_pair(train_path, test_path, dataset_stem, args)
        except Exception as exc:  # noqa: BLE001
            row = {
                "file": f"{dataset_stem}.csv",
                "train_file": train_path.name,
                "test_file": test_path.name,
                "rows_total": np.nan,
                "rows_used_for_training": np.nan,
                "rows_train": np.nan,
                "rows_test": np.nan,
                "mae": np.nan,
                "mae_days": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "model_path": "",
                "status": f"failed: {exc}",
            }
        all_rows.append(row)
        print(f"{dataset_stem}: {row['status']}")

    new_metrics_df = pd.DataFrame(all_rows)
    report_path = args.result_dir / "decision_tree_metrics_all.csv"
    if report_path.exists():
        existing_metrics_df = pd.read_csv(report_path)
        metrics_df = pd.concat([existing_metrics_df, new_metrics_df], ignore_index=True, sort=False)
    else:
        metrics_df = new_metrics_df

    metrics_df.to_csv(report_path, index=False)

    trained_count = int((new_metrics_df["status"] == "trained").sum())
    print(f"Trained models this run: {trained_count}/{len(pairs)}")
    print(f"Saved aggregate metrics: {report_path}")


if __name__ == "__main__":
    main()
