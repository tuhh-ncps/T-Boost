#!/usr/bin/env python3
"""Plot Normalized MAE lines for train vs validation splits.

Default input:
- results/decision_tree/predictions/BPI12_prediction.csv

Expected columns in input CSV:
- split (train/test)
- case_index
- actual_next_time
- prediction_next_time

Input selection:
    File-based via --input. Provide one prediction CSV path directly.

Example:
    python scripts/decision_tree_plot.py --input results/decision_tree/predictions/BPI12_prediction.csv

Usage:
    --input PATH          Prediction CSV to plot
    --output PATH         Output PNG file path
    --metrics-output PATH Output CSV with MAE summary metrics
    --rolling-window N    Rolling mean window used for smoothing
    --per-resource        Also generate one plot per resource value
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("results/decision_tree/predictions/BPI12_prediction.csv")
DEFAULT_OUTPUT = Path("results/decision_tree/plots/BPI12_prediction_mae_line.png")
DEFAULT_METRICS_OUTPUT = Path("results/decision_tree/plots/BPI12_prediction_mae_summary.csv")
SECONDS_PER_DAY = 86400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/test MAE line plot from prediction CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to prediction CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output PNG plot")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_OUTPUT,
        help="Path to output MAE summary CSV",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=20000,
        help="Rolling window for MAE smoothing at case level (default: 200)",
    )
    parser.add_argument(
        "--per-resource",
        action="store_true",
        help="Also create one MAE plot per resource value",
    )
    parser.add_argument(
        "--resource-col",
        type=str,
        default="org:resource",
        help="Resource column name used for per-resource plots (default: org:resource)",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    required = {"split", "actual_next_time", "prediction_next_time"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_split_mae(df: pd.DataFrame, split_name: str, rolling_window: int) -> pd.DataFrame:
    split_df = df[df["split"].astype(str).str.lower() == split_name].copy()
    if split_df.empty:
        return split_df

    split_df["actual_next_time"] = pd.to_numeric(split_df["actual_next_time"], errors="coerce")
    split_df["prediction_next_time"] = pd.to_numeric(split_df["prediction_next_time"], errors="coerce")
    split_df = split_df.dropna(subset=["actual_next_time", "prediction_next_time"]).reset_index(drop=True)
    if split_df.empty:
        return split_df

    split_df["actual_days"] = split_df["actual_next_time"] / SECONDS_PER_DAY
    split_df["abs_error_days"] = (split_df["actual_next_time"] - split_df["prediction_next_time"]).abs() / SECONDS_PER_DAY

    mean_next_time_days = float(split_df["actual_days"].mean())
    if np.isclose(mean_next_time_days, 0.0):
        split_df["nmae"] = np.nan
    else:
        split_df["nmae"] = split_df["abs_error_days"] / mean_next_time_days

    split_df["step"] = np.arange(1, len(split_df) + 1)
    window = max(1, min(int(rolling_window), len(split_df)))
    split_df["nmae_line"] = split_df["nmae"].rolling(window=window, min_periods=1).mean()
    return split_df


def _safe_name(value: object) -> str:
    name = str(value).strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("_")
    return name or "unknown"


def plot_mae_lines(train_df: pd.DataFrame, test_df: pd.DataFrame, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    if not train_df.empty:
        plt.plot(train_df["step"], train_df["nmae_line"], label="Train NMAE", linewidth=2.0)
    if not test_df.empty:
        plt.plot(test_df["step"], test_df["nmae_line"], label="Validation NMAE", linewidth=2.0)

    plt.title(title)
    plt.xlabel("Event Index in Split")
    plt.ylabel("Normalized MAE")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def create_per_resource_plots(df: pd.DataFrame, args: argparse.Namespace) -> Path | None:
    if args.resource_col not in df.columns:
        print(f"Skip per-resource plots: column not found: {args.resource_col}")
        return None

    per_resource_dir = args.output.parent / "per_resource"
    per_resource_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    stem = args.output.stem
    for resource in sorted(df[args.resource_col].dropna().unique().tolist(), key=lambda x: str(x)):
        resource_df = df[df[args.resource_col] == resource].copy()
        train_df = prepare_split_mae(resource_df, "train", args.rolling_window)
        test_df = prepare_split_mae(resource_df, "test", args.rolling_window)
        if train_df.empty and test_df.empty:
            continue

        safe_resource = _safe_name(resource)
        output_path = per_resource_dir / f"{stem}_{safe_resource}.png"
        plot_mae_lines(train_df, test_df, f"Normalized MAE Plot: {resource} (Train vs Validation)", output_path)

        if not train_df.empty:
            rows.append(
                {
                    "resource": resource,
                    "split": "train",
                    "nmae": float(train_df["nmae"].mean()),
                    "mean_next_time_days": float(train_df["actual_days"].mean()),
                    "rows": int(len(train_df)),
                    "plot_path": str(output_path),
                }
            )
        if not test_df.empty:
            rows.append(
                {
                    "resource": resource,
                    "split": "validation",
                    "nmae": float(test_df["nmae"].mean()),
                    "mean_next_time_days": float(test_df["actual_days"].mean()),
                    "rows": int(len(test_df)),
                    "plot_path": str(output_path),
                }
            )

    summary_path = per_resource_dir / "nmae_summary_per_resource.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, low_memory=False)
    validate_columns(df)

    train_df = prepare_split_mae(df, "train", args.rolling_window)
    test_df = prepare_split_mae(df, "test", args.rolling_window)

    if train_df.empty and test_df.empty:
        raise ValueError("No valid rows found for either train or test split.")

    plot_mae_lines(train_df, test_df, "Normalized MAE Plot: Train vs Validation", args.output)

    summary_rows = []
    if not train_df.empty:
        summary_rows.append(
            {
                "split": "train",
                "nmae": float(train_df["nmae"].mean()),
                "mean_next_time_days": float(train_df["actual_days"].mean()),
                "rows": int(len(train_df)),
            }
        )
    if not test_df.empty:
        summary_rows.append(
            {
                "split": "validation",
                "nmae": float(test_df["nmae"].mean()),
                "mean_next_time_days": float(test_df["actual_days"].mean()),
                "rows": int(len(test_df)),
            }
        )

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.metrics_output, index=False)

    per_resource_summary_path = None
    if args.per_resource:
        per_resource_summary_path = create_per_resource_plots(df, args)

    print(f"Saved plot: {args.output}")
    print(f"Saved MAE summary: {args.metrics_output}")
    if per_resource_summary_path is not None:
        print(f"Saved per-resource summary: {per_resource_summary_path}")


if __name__ == "__main__":
    main()
