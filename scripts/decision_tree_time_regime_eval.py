#!/usr/bin/env python3
"""Evaluate a trained time-regime classifier and save metrics/plots.

Input selection:
    Uses --data-dir and --result-dir folders. Optionally use --dataset to
    evaluate one dataset stem.

Example:
    python scripts/decision_tree_time_regime_eval.py --data-dir data_csv --result-dir results/decision_tree_time_regime

Usage:
    --data-dir PATH   Folder containing split CSV files
    --result-dir PATH Result directory containing trained models and metrics
    --dataset NAME    Optional dataset filename or stem
    --top-k N         Top-k cutoff for top-k accuracy (1 to 3)
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

_time_regime_module = importlib.import_module("scripts.decision_tree_time_regime")
add_transition_dt1_feature = _time_regime_module.add_transition_dt1_feature
add_transition_long_term_ratio_feature = _time_regime_module.add_transition_long_term_ratio_feature
compensate_zero_time_values = _time_regime_module.compensate_zero_time_values
coerce_types = _time_regime_module.coerce_types
get_model_feature_columns = _time_regime_module.get_model_feature_columns
INT_TO_TIME_REGIME = _time_regime_module.INT_TO_TIME_REGIME
TIME_REGIME_CLASSES = _time_regime_module.TIME_REGIME_CLASSES
map_time_regime_to_int = _time_regime_module.map_time_regime_to_int
prepare_lgbm_features = _time_regime_module.prepare_lgbm_features


DEFAULT_DATA_DIR = Path("data_csv")
DEFAULT_RESULT_DIR = Path("results/decision_tree_time_regime")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate time-regime classification model with accuracy/F1 and classification plots."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing split CSV files")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR, help="Result directory")
    parser.add_argument("--dataset", type=str, default="", help="Dataset filename or stem (e.g., helpdesk.csv or helpdesk)")
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k cutoff for top-k accuracy (must be between 1 and 3 for the current regime labels)",
    )
    return parser.parse_args()


def _normalize_dataset_stem(dataset: str) -> str:
    if not dataset:
        return ""
    return Path(dataset).stem


def load_latest_trained_row(metrics_path: Path, dataset_stem: str) -> pd.Series:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["status"] == "trained"].copy()
    if dataset_stem:
        metrics = metrics[metrics["file"].astype(str).map(lambda s: Path(s).stem == dataset_stem)]

    if metrics.empty:
        scope = f" for dataset '{dataset_stem}'" if dataset_stem else ""
        raise ValueError(f"No trained rows found in metrics{scope}.")

    return metrics.iloc[-1]


def _pick_split_files(row: pd.Series) -> tuple[str, str]:
    dataset_stem = Path(str(row["file"])).stem

    train_file = row.get("train_file", "")
    test_file = row.get("test_file", "")

    if pd.isna(train_file) or not str(train_file).strip():
        train_file = f"{dataset_stem}_train.csv"
    if pd.isna(test_file) or not str(test_file).strip():
        test_file = f"{dataset_stem}_test.csv"

    return str(train_file), str(test_file)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    labels: list[int],
    label_names: list[str],
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names)
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted regime")
    ax.set_ylabel("Actual regime")
    ax.set_title("Time Regime Classification: Confusion Matrix")

    threshold = float(cm.max()) / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=10,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    num_classes = len(TIME_REGIME_CLASSES)
    if args.top_k < 1 or args.top_k > num_classes:
        raise ValueError(f"--top-k must be between 1 and {num_classes} for the current time-regime target.")

    metrics_path = args.result_dir / "decision_tree_time_regime_metrics_all.csv"
    dataset_stem = _normalize_dataset_stem(args.dataset)
    row = load_latest_trained_row(metrics_path, dataset_stem)

    dataset_file = str(row["file"])
    dataset_stem = Path(dataset_file).stem
    preceding_k = int(row.get("preceding_k", 3))
    model_path = Path(str(row["model_path"]))

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    train_file, test_file = _pick_split_files(row)
    test_path = args.data_dir / test_file
    train_path = args.data_dir / train_file
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    train_df_raw = pd.read_csv(train_path, low_memory=False)
    test_df_raw = pd.read_csv(test_path, low_memory=False)
    train_df = coerce_types(train_df_raw, preceding_k=preceding_k)
    test_df = coerce_types(test_df_raw, preceding_k=preceding_k)

    time_col = "next_time" if "next_time" in train_df.columns else "next_time_seconds"
    train_df[time_col] = compensate_zero_time_values(train_df[time_col], replacement_seconds=1.0)
    test_df[time_col] = compensate_zero_time_values(test_df[time_col], replacement_seconds=1.0)

    train_df, _ = add_transition_dt1_feature(train_df, train_df, q=10)
    test_df, _ = add_transition_dt1_feature(train_df, test_df, q=10)
    train_df, _ = add_transition_long_term_ratio_feature(train_df, train_df, q=10)
    test_df, _ = add_transition_long_term_ratio_feature(train_df, test_df, q=10)

    test_df = test_df.dropna(subset=["time_regime"]).copy()

    feature_cols = get_model_feature_columns(preceding_k=preceding_k)
    x_test = test_df[feature_cols]
    y_test_int = map_time_regime_to_int(test_df["time_regime"]).astype(float)
    valid_mask = y_test_int.notna()
    x_test = x_test.loc[valid_mask].copy()
    y_test_int = y_test_int.loc[valid_mask].astype(float)

    with model_path.open("rb") as handle:
        model_obj = pickle.load(handle)

    model = model_obj
    numeric_medians = None
    categorical_levels = None
    if isinstance(model_obj, dict) and "model" in model_obj:
        model = model_obj["model"]
        numeric_medians = model_obj.get("numeric_medians")
        categorical_levels = model_obj.get("categorical_levels")

    x_test_prepared, _, _ = prepare_lgbm_features(
        x_test,
        preceding_k=preceding_k,
        numeric_medians=numeric_medians,
        categorical_levels=categorical_levels,
    )

    y_proba = model.predict_proba(x_test_prepared)
    y_pred = np.asarray(model.predict(x_test_prepared), dtype=int)
    y_true = y_test_int.to_numpy(dtype=int)
    class_labels = np.asarray(getattr(model, "classes_", np.arange(num_classes)), dtype=int).tolist()
    class_names = [INT_TO_TIME_REGIME.get(int(lbl), f"q{int(lbl) + 1}") for lbl in class_labels]

    accuracy = float(accuracy_score(y_true, y_pred))
    top_k_accuracy = float(top_k_accuracy_score(y_true, y_proba, k=args.top_k, labels=class_labels))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    plots_dir = args.result_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = plots_dir / f"{dataset_stem}_classification_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, confusion_path, labels=class_labels, label_names=class_names)

    tables_dir = args.result_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_table_path = tables_dir / f"{dataset_stem}_classification_metrics.csv"
    pd.DataFrame(
        [
            {
                "accuracy": accuracy,
                "top_k_accuracy": top_k_accuracy,
                "top_k": int(args.top_k),
                "f1_macro": f1_macro,
                "samples": int(len(y_true)),
            }
        ]
    ).to_csv(metrics_table_path, index=False)

    print(f"dataset: {dataset_file}")
    print(f"train_file: {train_file}")
    print(f"test_file: {test_file}")
    print(f"preceding_k: {preceding_k}")
    print(f"accuracy: {accuracy:.6f}")
    print(f"top_{args.top_k}_accuracy: {top_k_accuracy:.6f}")
    print(f"f1_macro: {f1_macro:.6f}")
    print(f"confusion_matrix_path: {confusion_path}")
    print(f"metrics_table_path: {metrics_table_path}")


if __name__ == "__main__":
    main()
