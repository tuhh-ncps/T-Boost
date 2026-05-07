#!/usr/bin/env python3
"""Train Decision Tree classifiers to predict time_regime from event context.

Reads CSV files from data_csv/, trains one model per CSV file, and uses the
same feature family as decision_tree.py, except time_regime is used as the
label (target), not as an input feature.

Requested feature set:
- transition
- activity
- prefix_k
- time_span
- long_term_ratio
- prefix_dt_last
- prefix_dt_mean
- prefix_dt_std
- delta_t_1, delta_t_2, delta_t_3
- delta_trend
- relative_speed
- prefix_length
- log_prefix_dt_last
- log_time_span
- transition_time_mean
- transition_time_std

Target label:
- time_regime

Input selection:
    Use --data-dir for the split CSV folder. Optionally use --dataset to train
    only one dataset stem.

Example:
    python scripts/decision_tree_time_regime.py --data-dir data_csv --result-dir results/decision_tree_time_regime

Usage:
    --data-dir PATH          Folder containing input CSV files
    --result-dir PATH        Folder for models and reports
    --dataset NAME           Optional single CSV file to train
    --preceding-k N          Number of preceding activities to encode
    --max-depth-limit N      Upper bound for max_depth tuning
    --class-weight MODE      none or balanced (default: balanced)
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from lightgbm import LGBMClassifier, early_stopping


DEFAULT_DATA_DIR = Path("data_csv")
DEFAULT_RESULT_DIR = Path("results/decision_tree_time_regime")
DEFAULT_PRECEDING_K = 3

VALID_REGIMES = {"q1", "q2", "q3", "q4"}
TIME_REGIME_TO_INT = {"q1": 0, "q2": 1, "q3": 2}
INT_TO_TIME_REGIME = {v: k for k, v in TIME_REGIME_TO_INT.items()}
TIME_REGIME_CLASSES = ["q1", "q2", "q3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one LightGBM classifier per CSV file to predict time_regime."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder containing input CSV files")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Folder to store all models and reports (default: results/decision_tree_time_regime)",
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
        "--max-depth-limit",
        type=int,
        default=10,
        help="Upper bound for max_depth tuning (default: 10)",
    )
    parser.add_argument(
        "--preceding-k",
        type=int,
        default=DEFAULT_PRECEDING_K,
        help="Number of preceding activities to encode as prefix features (default: 6)",
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
        help="Minimum accuracy improvement required to reset patience (default: 1e-4)",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=["accuracy", "f1_macro"],
        default="accuracy",
        help="Metric used to choose best depth on validation split (default: accuracy)",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        choices=["none", "balanced"],
        default="balanced",
        help="Class weighting strategy (default: balanced)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of boosting iterations (default: 300)",
    )
    parser.add_argument("--learning-rate", type=float, default=0.005, help="LightGBM learning rate (default: 0.005)")
    parser.add_argument("--num-leaves", type=int, default=31, help="LightGBM number of leaves (default: 31)")
    parser.add_argument("--subsample", type=float, default=0.9, help="Row subsampling ratio (default: 0.9)")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column subsampling ratio (default: 0.8)")
    parser.add_argument(
        "--min-child-samples",
        type=int,
        default=5,
        help="Minimum samples per leaf child for LightGBM (default: 5)",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples required at a leaf node for depth tuning fallback (default: 1)",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="",
        help="Optional explicit train CSV filename or path (overrides automatic discovery)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="",
        help="Optional explicit test CSV filename or path (overrides automatic discovery)",
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
    for suffix in ("_train_with_time_regime", "_test_with_time_regime", "_train_with_gmm_regime", "_test_with_gmm_regime", "_train", "_test"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def build_dataset_pairs(data_dir: Path, dataset: str, train_file_arg: str, test_file_arg: str) -> List[Tuple[str, Path, Path]]:
    """Return list of (stem, train_path, test_path) pairs.

    If both --train-file and --test-file are provided, use them directly (paths may be absolute
    or relative to `data_dir`). Otherwise fall back to discovering pairs by looking for files
    named `<stem>_train.csv` and `<stem>_test.csv` in `data_dir`.
    """
    files = list_csv_files(data_dir)
    by_name = {p.name: p for p in files}
    pairs: List[Tuple[str, Path, Path]] = []

    # If explicit train/test files provided, use them directly
    if train_file_arg and test_file_arg:
        train_path = Path(train_file_arg)
        test_path = Path(test_file_arg)
        if not train_path.is_absolute():
            train_path = data_dir / train_path.name
        if not test_path.is_absolute():
            test_path = data_dir / test_path.name

        if not train_path.exists():
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")

        # derive a stem for naming outputs: remove common trailing tokens if present
        stem = strip_known_suffixes(train_path.stem)
        return [(stem, train_path, test_path)]

    # Dataset-based discovery: look for exact `<stem>_train.csv` and `<stem>_test.csv` pairs
    if dataset:
        stem = Path(dataset).stem
        train_name = f"{stem}_train.csv"
        test_name = f"{stem}_test.csv"
        if train_name not in by_name:
            raise FileNotFoundError(f"Train split not found in {data_dir}: {train_name}")
        if test_name not in by_name:
            raise FileNotFoundError(f"Test split not found in {data_dir}: {test_name}")
        return [(stem, by_name[train_name], by_name[test_name])]

    # Scan directory for conventional pairs
    for path in files:
        if not path.stem.endswith("_train"):
            continue
        stem = path.stem[: -len("_train")]
        train_path = path
        test_name = f"{stem}_test.csv"
        test_path = by_name.get(test_name)
        if test_path is None:
            continue
        pairs.append((stem, train_path, test_path))

    if not pairs:
        raise FileNotFoundError(
            f"No train/test split pairs found in {data_dir}. Provide --train-file and --test-file, or name files as <stem>_train.csv and <stem>_test.csv"
        )

    return sorted(pairs, key=lambda item: item[0])


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df


def load_single_dataset(path: Path, max_rows_per_file: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)
    if max_rows_per_file > 0:
        df = df.head(max_rows_per_file)
    return df


def validate_required_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    feature_cols = ["activity", "prefix_sequence", "time_span", "prefix_delta_t_sequence"]
    target_col = "time_regime"

    missing = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return feature_cols, target_col


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


def split_float_sequence(value: object) -> List[float]:
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    values: List[float] = []
    for part in re.split(r"\s*(?:,|;|\|)\s*", text):
        token = part.strip()
        if not token:
            continue
        try:
            values.append(max(0.0, float(token)))
        except ValueError:
            values.append(0.0)
    return values


def compute_log_regime_bins(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & (values > 0.0)

    if valid.sum() == 0:
        return np.array([], dtype=float)

    log_values = np.log(values[valid])
    # y = qcut(log(next_time)) with duplicate-edge protection.
    _, raw_bins = pd.qcut(log_values, q=4, retbins=True, duplicates="drop")
    edges = np.unique(np.concatenate(([-np.inf], raw_bins[1:-1], [np.inf])))
    return edges


def compensate_zero_time_values(series: pd.Series, replacement_seconds: float = 1.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = numeric.copy()
    out.loc[out == 0.0] = float(replacement_seconds)
    return out


def assign_time_regime_from_edges(series: pd.Series, edges: np.ndarray) -> pd.Series:
    labels = [f"q{i + 1}" for i in range(len(edges) - 1)]
    out = pd.Series(index=series.index, dtype="object")

    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna() & (numeric > 0.0)
    if valid.any() and len(labels) > 0:
        bins = pd.cut(np.log(numeric[valid]), bins=edges, labels=labels, include_lowest=True)
        out.loc[valid] = bins.astype(str)

    out = out.astype(str).str.strip().str.lower()
    out.loc[~out.isin(VALID_REGIMES)] = np.nan
    return out


def map_time_regime_to_int(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(TIME_REGIME_TO_INT)


def map_int_to_time_regime(values: np.ndarray) -> np.ndarray:
    return np.array([INT_TO_TIME_REGIME.get(int(v), "q1") for v in values], dtype=object)


def map_time_regime_probabilities(model: LGBMClassifier, x_data: pd.DataFrame) -> pd.DataFrame:
    probabilities = np.asarray(model.predict_proba(x_data), dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("predict_proba must return a 2D probability matrix")

    class_labels = np.asarray(getattr(model, "classes_", []), dtype=int)
    class_to_index = {int(label): idx for idx, label in enumerate(class_labels)}
    full_probabilities = np.zeros((len(x_data), len(TIME_REGIME_CLASSES)), dtype=float)

    for regime_name, regime_int in TIME_REGIME_TO_INT.items():
        source_index = class_to_index.get(int(regime_int))
        if source_index is not None:
            full_probabilities[:, int(regime_int)] = probabilities[:, source_index] * 100.0

    return pd.DataFrame(
        full_probabilities,
        index=x_data.index,
        columns=[f"predicted_time_regime_{regime}_pct" for regime in TIME_REGIME_CLASSES],
    )


def gaussian_intersection(
    w1: float,
    mu1: float,
    var1: float,
    w2: float,
    mu2: float,
    var2: float,
) -> float:
    var1 = max(float(var1), 1e-9)
    var2 = max(float(var2), 1e-9)
    w1 = max(float(w1), 1e-12)
    w2 = max(float(w2), 1e-12)

    a = (1.0 / (2.0 * var2)) - (1.0 / (2.0 * var1))
    b = (mu1 / var1) - (mu2 / var2)
    c = (mu2**2 / (2.0 * var2)) - (mu1**2 / (2.0 * var1)) + np.log((w2 * np.sqrt(var1)) / (w1 * np.sqrt(var2)))

    # Near-equal variances produce a linear equation.
    if np.isclose(a, 0.0, atol=1e-12):
        if np.isclose(b, 0.0, atol=1e-12):
            return 0.5 * (mu1 + mu2)
        return float(-c / b)

    roots = np.roots([a, b, c])
    real_roots = roots[np.isreal(roots)].real
    if real_roots.size == 0:
        return 0.5 * (mu1 + mu2)

    low, high = (mu1, mu2) if mu1 <= mu2 else (mu2, mu1)
    between = real_roots[(real_roots >= low) & (real_roots <= high)]
    if between.size > 0:
        return float(between[np.argmin(np.abs(between - 0.5 * (mu1 + mu2)))])

    return float(real_roots[np.argmin(np.abs(real_roots - 0.5 * (mu1 + mu2)))])


def build_sample_weights(y: pd.Series, class_weight: str) -> np.ndarray | None:
    if class_weight != "balanced":
        return None
    weights = compute_sample_weight("balanced", y)
    return np.asarray(weights, dtype=float)


def compute_dt1_bin_edges(series: pd.Series, q: int = 10) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna()
    if not valid.any():
        return np.array([-np.inf, np.inf], dtype=float)

    _, raw_bins = pd.qcut(values[valid], q=q, retbins=True, duplicates="drop")
    edges = np.unique(np.concatenate(([-np.inf], raw_bins[1:-1], [np.inf]))).astype(float)
    if len(edges) < 2:
        return np.array([-np.inf, np.inf], dtype=float)
    return edges


def add_transition_dt1_feature(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    q: int = 10,
) -> tuple[pd.DataFrame, np.ndarray]:
    train_log_dt1 = pd.to_numeric(train_df["log_delta_t_1"], errors="coerce")
    edges = compute_dt1_bin_edges(train_log_dt1, q=q)

    out = target_df.copy()
    target_log_dt1 = pd.to_numeric(out["log_delta_t_1"], errors="coerce")
    out["dt1_bin"] = pd.cut(target_log_dt1, bins=edges, include_lowest=True)
    out["transition_dt1"] = out["transition"].astype(str) + "_" + out["dt1_bin"].astype(str)
    out["dt1_bin"] = out["dt1_bin"].astype("object")
    out["transition_dt1"] = out["transition_dt1"].astype("object")
    out.loc[out["dt1_bin"].isna(), "dt1_bin"] = np.nan
    out.loc[out["transition_dt1"].isna(), "transition_dt1"] = np.nan
    return out, edges


def add_transition_long_term_ratio_feature(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    q: int = 10,
) -> tuple[pd.DataFrame, np.ndarray]:
    train_ltr = pd.to_numeric(train_df["long_term_ratio"], errors="coerce")
    edges = compute_dt1_bin_edges(train_ltr, q=q)

    out = target_df.copy()
    target_ltr = pd.to_numeric(out["long_term_ratio"], errors="coerce")
    out["long_term_ratio_bin"] = pd.cut(target_ltr, bins=edges, include_lowest=True)
    out["transition_long_term_ratio"] = out["transition"].astype(str) + "_" + out["long_term_ratio_bin"].astype(str)
    out["long_term_ratio_bin"] = out["long_term_ratio_bin"].astype("object")
    out["transition_long_term_ratio"] = out["transition_long_term_ratio"].astype("object")
    out.loc[out["long_term_ratio_bin"].isna(), "long_term_ratio_bin"] = np.nan
    out.loc[out["transition_long_term_ratio"].isna(), "transition_long_term_ratio"] = np.nan
    return out, edges


def add_transition_stats_from_train(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    time_col: str,
) -> pd.DataFrame:
    transition_target = pd.to_numeric(train_df[time_col], errors="coerce")
    transition_stats = (
        pd.DataFrame({"transition": train_df["transition"], "next_time_target": transition_target})
        .dropna(subset=["transition", "next_time_target"])
        .groupby("transition", dropna=False)["next_time_target"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "transition_time_mean", "std": "transition_time_std"})
    )

    global_mean = float(transition_target.dropna().mean()) if transition_target.notna().any() else 0.0
    global_std = float(transition_target.dropna().std()) if transition_target.notna().any() else 0.0

    out = target_df.join(transition_stats, on="transition")
    out["transition_time_mean"] = pd.to_numeric(out["transition_time_mean"], errors="coerce").fillna(global_mean)
    out["transition_time_std"] = pd.to_numeric(out["transition_time_std"], errors="coerce").fillna(global_std)
    return out


def coerce_types(df: pd.DataFrame, preceding_k: int) -> pd.DataFrame:
    out = df.copy()

    prefix_values = out["prefix_sequence"].apply(lambda value: split_prefix_sequence(value, k=preceding_k))
    for i in range(preceding_k):
        out[f"prefix_{i + 1}"] = prefix_values.str[i]

    prefix_last = prefix_values.map(lambda values: values[-1] if values else "START")
    out["transition"] = prefix_last.astype(str) + "→" + out["activity"].fillna("START").astype(str)

    delta_values = out["prefix_delta_t_sequence"].apply(split_float_sequence)
    out["prefix_dt_last"] = delta_values.map(lambda arr: float(arr[-1]) if arr else 0.0)
    out["prefix_dt_mean"] = delta_values.map(lambda arr: float(np.mean(arr)) if arr else 0.0)
    out["prefix_dt_std"] = delta_values.map(lambda arr: float(np.std(arr)) if arr else 0.0)

    # Keep explicit recent deltas instead of summing; delta_t_1 is the most recent.
    out["delta_t_1"] = delta_values.map(lambda arr: float(arr[-1]) if len(arr) >= 1 else 0.0)
    out["delta_t_2"] = delta_values.map(lambda arr: float(arr[-2]) if len(arr) >= 2 else 0.0)
    out["delta_t_3"] = delta_values.map(lambda arr: float(arr[-3]) if len(arr) >= 3 else 0.0)
    out["delta_trend"] = out["delta_t_1"] - out["delta_t_2"]
    out["trend_2"] = out["delta_t_2"] - out["delta_t_3"]
    out["trend_acceleration"] = out["delta_t_1"] - (2.0 * out["delta_t_2"]) + out["delta_t_3"]
    out["relative_speed"] = out["delta_t_1"] / (out["prefix_dt_mean"] + 1e-6)
    out["growth_ratio"] = out["delta_t_1"] / (out["delta_t_3"] + 1e-6)
    out["recent_std"] = np.sqrt(
        (
            (out["delta_t_1"] - (out["delta_t_1"] + out["delta_t_2"] + out["delta_t_3"]) / 3.0) ** 2
            + (out["delta_t_2"] - (out["delta_t_1"] + out["delta_t_2"] + out["delta_t_3"]) / 3.0) ** 2
            + (out["delta_t_3"] - (out["delta_t_1"] + out["delta_t_2"] + out["delta_t_3"]) / 3.0) ** 2
        )
        / 3.0
    )
    out["log_delta_t_1"] = np.log1p(pd.to_numeric(out["delta_t_1"], errors="coerce").clip(lower=0))

    out["prefix_length"] = prefix_values.map(lambda arr: int(sum(1 for token in arr if token != "START")))
    out["time_span"] = pd.to_numeric(out["time_span"], errors="coerce")
    prefix_length_num = pd.to_numeric(out["prefix_length"], errors="coerce")
    out["long_term_ratio"] = np.where(
        prefix_length_num > 0,
        out["time_span"] / prefix_length_num,
        1.0,
    )
    out["position_ratio"] = prefix_length_num / (prefix_length_num + 5.0)

    out["log_prefix_dt_last"] = np.log1p(pd.to_numeric(out["prefix_dt_last"], errors="coerce").clip(lower=0))
    out["log_time_span"] = np.log1p(pd.to_numeric(out["time_span"], errors="coerce").clip(lower=0))

    numeric_cols = [
        "time_span",
        "prefix_dt_last",
        "prefix_dt_mean",
        "prefix_dt_std",
        "delta_t_1",
        "delta_t_2",
        "delta_t_3",
        "delta_trend",
        "trend_2",
        "trend_acceleration",
        "relative_speed",
        "growth_ratio",
        "position_ratio",
        "recent_std",
        "log_delta_t_1",
        "prefix_length",
        "long_term_ratio",
        "log_prefix_dt_last",
        "log_time_span",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["activity", "prefix_sequence", "prefix_delta_t_sequence", "transition", "time_regime"]:
        out[col] = out[col].astype("object")
        out.loc[out[col].isna(), col] = np.nan

    # Keep only known regimes for stable classification targets.
    out["time_regime"] = out["time_regime"].astype(str).str.strip().str.lower()
    out.loc[~out["time_regime"].isin(VALID_REGIMES), "time_regime"] = np.nan

    for i in range(preceding_k):
        col = f"prefix_{i + 1}"
        out[col] = out[col].astype("object")
        out.loc[out[col].isna(), col] = "START"

    return out


def get_model_feature_columns(preceding_k: int) -> List[str]:
    cols = ["transition", "transition_dt1", "transition_long_term_ratio", "activity", f"prefix_{preceding_k}"]
    cols.extend(
        [
            "time_span",
            "prefix_dt_last",
            "prefix_dt_mean",
            "prefix_dt_std",
            "delta_t_1",
            "delta_t_2",
            "delta_t_3",
            "delta_trend",
            "trend_2",
            "trend_acceleration",
            "relative_speed",
            "growth_ratio",
            "position_ratio",
            "recent_std",
            "log_delta_t_1",
            "prefix_length",
            "long_term_ratio",
            "log_prefix_dt_last",
            "log_time_span",
            "transition_time_mean",
            "transition_time_std",
        ]
    )
    return cols


def get_categorical_feature_columns(preceding_k: int) -> List[str]:
    return ["transition", "transition_dt1", "transition_long_term_ratio", "activity", f"prefix_{preceding_k}"]


def get_numeric_feature_columns(preceding_k: int) -> List[str]:
    feature_cols = get_model_feature_columns(preceding_k)
    categorical_cols = set(get_categorical_feature_columns(preceding_k))
    return [col for col in feature_cols if col not in categorical_cols]


def prepare_lgbm_features(
    df: pd.DataFrame,
    preceding_k: int,
    numeric_medians: Dict[str, float] | None = None,
    categorical_levels: Dict[str, List[str]] | None = None,
) -> tuple[pd.DataFrame, Dict[str, float], Dict[str, List[str]]]:
    out = df.copy()
    numeric_cols = get_numeric_feature_columns(preceding_k)
    categorical_cols = get_categorical_feature_columns(preceding_k)

    if numeric_medians is None:
        numeric_medians = {}
        for col in numeric_cols:
            series = pd.to_numeric(out[col], errors="coerce")
            med = float(series.median()) if series.notna().any() else 0.0
            if not np.isfinite(med):
                med = 0.0
            numeric_medians[col] = med

    for col in numeric_cols:
        series = pd.to_numeric(out[col], errors="coerce")
        out[col] = series.fillna(float(numeric_medians.get(col, 0.0))).astype(float)

    if categorical_levels is None:
        categorical_levels = {}
        for col in categorical_cols:
            filled = out[col].astype("string").fillna("__MISSING__")
            levels = pd.Index(filled.unique().tolist())
            if "__MISSING__" not in levels:
                levels = levels.append(pd.Index(["__MISSING__"]))
            if "__UNK__" not in levels:
                levels = levels.append(pd.Index(["__UNK__"]))
            categorical_levels[col] = levels.astype(str).tolist()

    for col in categorical_cols:
        levels = list(categorical_levels.get(col, ["__MISSING__", "__UNK__"]))
        filled = out[col].astype("string").fillna("__MISSING__")
        known = filled.isin(levels)
        filled = filled.where(known, "__UNK__")
        out[col] = pd.Categorical(filled.astype(str), categories=levels)

    return out, numeric_medians, categorical_levels


def build_model_pipeline(
    random_state: int,
    max_depth: int,
    preceding_k: int,
    class_weight: str,
    min_samples_leaf: int,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    subsample: float,
    colsample_bytree: float,
    min_child_samples: int,
) -> LGBMClassifier:
    _ = (preceding_k, class_weight, min_samples_leaf)
    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    return model


def generate_oof_probs(
    model_builder,
    X: pd.DataFrame,
    y: pd.Series,
    preceding_k: int,
    class_weight: str,
    n_splits: int = 3,
) -> pd.DataFrame:
    oof_probs = pd.DataFrame(
        0.0,
        index=X.index,
        columns=[f"predicted_time_regime_{r}_pct" for r in TIME_REGIME_CLASSES],
    )

    if len(X) <= n_splits:
        raise ValueError("Not enough samples for TimeSeriesSplit OOF prediction.")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    categorical_features = get_categorical_feature_columns(preceding_k)

    for train_idx, val_idx in tscv.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]

        X_train_prepared, numeric_medians_fold, categorical_levels_fold = prepare_lgbm_features(
            X_train_fold,
            preceding_k=preceding_k,
        )
        X_val_prepared, _, _ = prepare_lgbm_features(
            X_val_fold,
            preceding_k=preceding_k,
            numeric_medians=numeric_medians_fold,
            categorical_levels=categorical_levels_fold,
        )

        model = model_builder()
        sample_weight_fold = build_sample_weights(y_train_fold, class_weight)
        fit_kwargs = {"categorical_feature": categorical_features}
        if sample_weight_fold is not None:
            fit_kwargs["sample_weight"] = sample_weight_fold

        model.fit(X_train_prepared, y_train_fold, **fit_kwargs)

        val_index = X.index[val_idx]
        oof_probs.loc[val_index, :] = map_time_regime_probabilities(model, X_val_prepared).to_numpy()

    return oof_probs


def build_alpha_grid(alpha_min: float, alpha_max: float, alpha_step: float) -> List[float]:
    if alpha_step <= 0:
        raise ValueError("--quantile-alpha-step must be > 0")

    lo = min(alpha_min, alpha_max)
    hi = max(alpha_min, alpha_max)
    values = np.arange(lo, hi + alpha_step * 0.5, alpha_step, dtype=float)
    values = np.clip(values, 1e-6, 1.0 - 1e-6)
    return sorted({round(float(v), 6) for v in values})


def tune_best_depth(
    X_train_fit: pd.DataFrame,
    y_train_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    args: argparse.Namespace,
) -> Tuple[
    LGBMClassifier,
    int,
    float,
    Tuple[float, float],
    int,
    bool,
    Dict[str, List[str]],
]:
    best_pipeline: LGBMClassifier | None = None
    best_categorical_levels: Dict[str, List[str]] | None = None
    best_depth = 1
    best_scores: Tuple[float, float] | None = None
    depths_tried = 0
    stopped_early = False
    categorical_features = get_categorical_feature_columns(args.preceding_k)
    sample_weight_fit = build_sample_weights(y_train_fit, args.class_weight)
    X_fit_prepared, numeric_medians, categorical_levels = prepare_lgbm_features(
        X_train_fit,
        preceding_k=args.preceding_k,
    )
    X_val_prepared, _, _ = prepare_lgbm_features(
        X_val,
        preceding_k=args.preceding_k,
        numeric_medians=numeric_medians,
        categorical_levels=categorical_levels,
    )

    no_improve_count = 0
    for depth in range(1, max(2, args.max_depth_limit + 1)):
        depths_tried += 1
        pipeline = build_model_pipeline(
            random_state=args.random_state,
            max_depth=depth,
            preceding_k=args.preceding_k,
            class_weight=args.class_weight,
            min_samples_leaf=args.min_samples_leaf,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            min_child_samples=args.min_child_samples,
        )
        fit_kwargs = {
            "categorical_feature": categorical_features,
            "eval_set": [(X_val_prepared, y_val)],
            "eval_metric": "multi_logloss",
            "callbacks": [early_stopping(stopping_rounds=50, verbose=False)],
        }
        if sample_weight_fit is not None:
            fit_kwargs["sample_weight"] = sample_weight_fit
        pipeline.fit(X_fit_prepared, y_train_fit, **fit_kwargs)
        y_pred = pipeline.predict(X_val_prepared)

        accuracy = float(accuracy_score(y_val.to_numpy(), y_pred))
        f1_macro = float(f1_score(y_val.to_numpy(), y_pred, average="macro"))

        if best_scores is None:
            best_pipeline = pipeline
            best_depth = depth
            best_scores = (accuracy, f1_macro)
            best_categorical_levels = categorical_levels
            continue

        best_accuracy, best_f1_macro = best_scores
        improved = False
        if args.selection_metric == "accuracy":
            if accuracy > best_accuracy + args.early_stopping_min_delta:
                improved = True
            elif np.isclose(accuracy, best_accuracy, atol=args.early_stopping_min_delta) and f1_macro > best_f1_macro:
                improved = True
        else:
            if f1_macro > best_f1_macro + args.early_stopping_min_delta:
                improved = True
            elif np.isclose(f1_macro, best_f1_macro, atol=args.early_stopping_min_delta) and accuracy > best_accuracy:
                improved = True

        if improved:
            best_pipeline = pipeline
            best_depth = depth
            best_scores = (accuracy, f1_macro)
            best_categorical_levels = categorical_levels
            no_improve_count = 0
        else:
            no_improve_count += 1
            if args.early_stopping_patience >= 0 and no_improve_count > args.early_stopping_patience:
                stopped_early = True
                break

    assert best_pipeline is not None and best_scores is not None
    assert best_categorical_levels is not None
    return (
        best_pipeline,
        best_depth,
        best_scores,
        depths_tried,
        stopped_early,
        best_categorical_levels,
    )


def train_one_pair(train_csv_path: Path, test_csv_path: Path, dataset_stem: str, args: argparse.Namespace) -> Dict[str, object]:
    raw_train_data = load_single_dataset(train_csv_path, args.max_rows_per_file)
    raw_test_data = load_single_dataset(test_csv_path, args.max_rows_per_file)
    _, target_col = validate_required_columns(raw_train_data)
    validate_required_columns(raw_test_data)

    if args.preceding_k <= 0:
        raise ValueError("--preceding-k must be a positive integer")

    train_data = coerce_types(raw_train_data, preceding_k=args.preceding_k)
    test_data = coerce_types(raw_test_data, preceding_k=args.preceding_k)

    # Build train-derived log-scale target bins, then apply same edges to test.
    time_col = "next_time" if "next_time" in train_data.columns else "next_time_seconds"
    train_data[time_col] = compensate_zero_time_values(train_data[time_col], replacement_seconds=1.0)
    test_data[time_col] = compensate_zero_time_values(test_data[time_col], replacement_seconds=1.0)
    log_bin_edges = compute_log_regime_bins(train_data[time_col])
    if len(log_bin_edges) >= 2:
        train_data["time_regime"] = assign_time_regime_from_edges(train_data[time_col], log_bin_edges)
        test_data["time_regime"] = assign_time_regime_from_edges(test_data[time_col], log_bin_edges)

    # Train-only transition statistics, merged into both train and test features.
    train_data = add_transition_stats_from_train(train_data, train_data, time_col=time_col)
    test_data = add_transition_stats_from_train(train_data, test_data, time_col=time_col)

    # Train-derived qcut bins over log_delta_t_1, then transition + bin combination.
    train_data, dt1_edges = add_transition_dt1_feature(train_data, train_data, q=10)
    test_data, _ = add_transition_dt1_feature(train_data, test_data, q=10)
    train_data, long_term_ratio_edges = add_transition_long_term_ratio_feature(train_data, train_data, q=10)
    test_data, _ = add_transition_long_term_ratio_feature(train_data, test_data, q=10)

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
            "accuracy": np.nan,
            "f1_macro": np.nan,
            "model_path": "",
            "status": "skipped_not_enough_rows",
        }

    feature_cols = get_model_feature_columns(preceding_k=args.preceding_k)

    X_train = train_df[feature_cols]
    y_train = map_time_regime_to_int(train_df[target_col])
    X_test = test_df[feature_cols]
    y_test = map_time_regime_to_int(test_df[target_col])

    train_valid_mask = y_train.notna()
    test_valid_mask = y_test.notna()
    X_train = X_train.loc[train_valid_mask].copy()
    y_train = y_train.loc[train_valid_mask].astype(int)
    X_test = X_test.loc[test_valid_mask].copy()
    y_test = y_test.loc[test_valid_mask].astype(int)
    train_df = train_df.loc[train_valid_mask].copy()
    test_df = test_df.loc[test_valid_mask].copy()

    if len(X_train) < 2 or len(X_test) < 1:
        return {
            "file": f"{dataset_stem}.csv",
            "train_file": train_csv_path.name,
            "test_file": test_csv_path.name,
            "rows_total": int(len(train_data) + len(test_data)),
            "rows_used_for_training": int(len(train_df) + len(test_df)),
            "rows_train": 0,
            "rows_test": 0,
            "accuracy": np.nan,
            "f1_macro": np.nan,
            "model_path": "",
            "status": "skipped_not_enough_mapped_rows",
        }

    # Tune hyperparameters on train-only data to avoid test-set leakage.
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=max(0.1, min(0.3, float(args.test_size))),
        random_state=args.random_state,
        stratify=y_train,
    )

    if len(y_train_fit) < 2 or len(y_val) < 2:
        X_train_fit = X_train
        y_train_fit = y_train
        X_val = X_train
        y_val = y_train

    _, best_depth, _, depths_tried, stopped_early, _ = tune_best_depth(
        X_train_fit, y_train_fit, X_val, y_val, args
    )

    final_pipeline = build_model_pipeline(
        random_state=args.random_state,
        max_depth=best_depth,
        preceding_k=args.preceding_k,
        class_weight=args.class_weight,
        min_samples_leaf=args.min_samples_leaf,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_samples=args.min_child_samples,
    )
    categorical_features = get_categorical_feature_columns(args.preceding_k)
    X_train_prepared, numeric_medians, categorical_levels = prepare_lgbm_features(
        X_train,
        preceding_k=args.preceding_k,
    )
    X_test_prepared, _, _ = prepare_lgbm_features(
        X_test,
        preceding_k=args.preceding_k,
        numeric_medians=numeric_medians,
        categorical_levels=categorical_levels,
    )

    def build_model() -> LGBMClassifier:
        return build_model_pipeline(
            random_state=args.random_state,
            max_depth=best_depth,
            preceding_k=args.preceding_k,
            class_weight=args.class_weight,
            min_samples_leaf=args.min_samples_leaf,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            min_child_samples=args.min_child_samples,
        )

    y_proba_train_pct = generate_oof_probs(
        build_model,
        X_train,
        y_train,
        preceding_k=args.preceding_k,
        class_weight=args.class_weight,
        n_splits=3,
    )

    sample_weight_full = build_sample_weights(y_train, args.class_weight)
    final_fit_kwargs = {"categorical_feature": categorical_features}
    if sample_weight_full is not None:
        final_fit_kwargs["sample_weight"] = sample_weight_full
    final_pipeline.fit(X_train_prepared, y_train, **final_fit_kwargs)
    y_pred_train_raw = final_pipeline.predict(X_train_prepared)
    y_pred_test_raw = final_pipeline.predict(X_test_prepared)
    y_proba_test_pct = map_time_regime_probabilities(final_pipeline, X_test_prepared)
    y_test_values = y_test.to_numpy(dtype=int)
    accuracy = float(accuracy_score(y_test_values, y_pred_test_raw))
    f1_macro = float(f1_score(y_test_values, y_pred_test_raw, average="macro"))

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["prediction_time_regime"] = map_int_to_time_regime(np.asarray(y_pred_train_raw, dtype=int))
    test_df["prediction_time_regime"] = map_int_to_time_regime(np.asarray(y_pred_test_raw, dtype=int))
    train_df = pd.concat([train_df, y_proba_train_pct], axis=1)
    test_df = pd.concat([test_df, y_proba_test_pct], axis=1)

    train_input_cols = list(raw_train_data.columns)
    test_input_cols = list(raw_test_data.columns)
    raw_train_with_row_id = raw_train_data.copy()
    raw_train_with_row_id["__row_id"] = raw_train_with_row_id.index
    raw_test_with_row_id = raw_test_data.copy()
    raw_test_with_row_id["__row_id"] = raw_test_with_row_id.index

    train_export = raw_train_with_row_id.merge(
        train_df[["__row_id", "prediction_time_regime", *y_proba_train_pct.columns]],
        on="__row_id",
        how="inner",
    )
    train_export["actual_time_regime"] = map_int_to_time_regime(y_train.to_numpy(dtype=int))
    train_export["prediction_time_regime"] = train_export["prediction_time_regime"].astype(str)
    train_export = train_export[
        train_input_cols
        + [
            "actual_time_regime",
            "prediction_time_regime",
            *y_proba_train_pct.columns,
        ]
    ]

    prediction_export = raw_test_with_row_id.merge(
        test_df[["__row_id", "prediction_time_regime", *y_proba_test_pct.columns]],
        on="__row_id",
        how="inner",
    )
    prediction_export["actual_time_regime"] = map_int_to_time_regime(y_test_values)
    prediction_export["prediction_time_regime"] = prediction_export["prediction_time_regime"].astype(str)
    prediction_export = prediction_export[
        test_input_cols
        + [
            "actual_time_regime",
            "prediction_time_regime",
            *y_proba_test_pct.columns,
        ]
    ]

    train_with_regime_export = raw_train_with_row_id.merge(
        train_df[["__row_id", "prediction_time_regime", *y_proba_train_pct.columns]],
        on="__row_id",
        how="left",
    )
    # Replace the original raw time_regime with the reassigned values used by this run.
    train_with_regime_export["time_regime"] = map_int_to_time_regime(y_train.to_numpy(dtype=int))
    train_with_regime_export = train_with_regime_export.drop(columns=["__row_id"])
    train_with_regime_export = train_with_regime_export.rename(columns={"prediction_time_regime": "predicted_time_regime"})

    test_with_regime_export = raw_test_with_row_id.merge(
        test_df[["__row_id", "prediction_time_regime", *y_proba_test_pct.columns]],
        on="__row_id",
        how="left",
    )
    # Replace the original raw time_regime with the reassigned values used by this run.
    test_with_regime_export["time_regime"] = map_int_to_time_regime(y_test_values)
    test_with_regime_export = test_with_regime_export.drop(columns=["__row_id"])
    test_with_regime_export = test_with_regime_export.rename(columns={"prediction_time_regime": "predicted_time_regime"})

    train_with_regime_path = args.data_dir / f"{dataset_stem}_train_with_time_regime.csv"
    train_with_regime_export.to_csv(train_with_regime_path, index=False)

    test_with_regime_path = args.data_dir / f"{dataset_stem}_test_with_time_regime.csv"
    test_with_regime_export.to_csv(test_with_regime_path, index=False)

    models_dir = args.result_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{dataset_stem}_decision_tree_time_regime.pkl"
    model_artifact = {
        "model": final_pipeline,
        "preceding_k": int(args.preceding_k),
        "numeric_medians": numeric_medians,
        "categorical_levels": categorical_levels,
        "categorical_features": get_categorical_feature_columns(args.preceding_k),
        "time_regime_to_int": TIME_REGIME_TO_INT,
        "int_to_time_regime": INT_TO_TIME_REGIME,
        "dt1_bin_edges": dt1_edges.tolist(),
        "long_term_ratio_bin_edges": long_term_ratio_edges.tolist(),
        "class_weight": str(args.class_weight),
    }
    with model_path.open("wb") as handle:
        pickle.dump(model_artifact, handle)

    predictions_dir = args.result_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = predictions_dir / f"{dataset_stem}_prediction.csv"
    prediction_export.to_csv(prediction_path, index=False)

    return {
        "file": f"{dataset_stem}.csv",
        "model_type": "lightgbm_classifier",
        "train_file": train_csv_path.name,
        "test_file": test_csv_path.name,
        "test_with_time_regime_file": test_with_regime_path.name,
        "rows_total": int(len(train_data) + len(test_data)),
        "rows_used_for_training": int(len(train_df) + len(test_df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "preceding_k": int(args.preceding_k),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "best_max_depth": int(best_depth),
        "depths_tried": int(depths_tried),
        "stopped_early": bool(stopped_early),
        "threshold_t1": np.nan,
        "threshold_t2": np.nan,
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
        "status": "trained",
    }


def main() -> None:
    args = parse_args()
    pairs = build_dataset_pairs(args.data_dir, args.dataset, args.train_file, args.test_file)

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
                "accuracy": np.nan,
                "f1_macro": np.nan,
                "model_path": "",
                "status": f"failed: {exc}",
            }
        all_rows.append(row)
        print(f"{dataset_stem}: {row['status']}")

    new_metrics_df = pd.DataFrame(all_rows)
    report_path = args.result_dir / "decision_tree_time_regime_metrics_all.csv"
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
