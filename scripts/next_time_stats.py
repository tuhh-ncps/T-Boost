#!/usr/bin/env python3
"""Compute mean and median of next_time for one or all CSV datasets.

Examples:
  python scripts/next_time_stats.py
  python scripts/next_time_stats.py --dataset BPI17.csv
  python scripts/next_time_stats.py --output-csv results/next_time_stats.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("data_csv")
DEFAULT_OUTPUT_CSV = Path("results/next_time_stats.csv")
DEFAULT_PLOTS_DIR = Path("results/next_time_plots")
SECONDS_PER_DAY = 86400.0
TIME_UNIT_TO_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate average and median of next_time per CSV dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder containing input CSV files (default: data_csv)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional single CSV filename (e.g., BPI17.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path for stats table (default: results/next_time_stats.csv)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=["none", "hist", "kde", "both"],
        default="none",
        help="Optional log1p(next_time) plot type per dataset (default: none)",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Folder for generated plots (default: results/next_time_plots)",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=50,
        help="Number of bins for histogram plot (default: 50)",
    )
    return parser.parse_args()


def detect_time_unit_seconds(df: pd.DataFrame) -> float:
    if "time_unit" not in df.columns:
        return 1.0

    series = df["time_unit"].dropna().astype(str).str.strip().str.lower()
    if series.empty:
        return 1.0

    unit = series.iloc[0]
    return float(TIME_UNIT_TO_SECONDS.get(unit, 1.0))


def list_csv_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {data_dir}")

    files = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return files


def select_files(data_dir: Path, dataset: str) -> List[Path]:
    files = list_csv_files(data_dir)
    by_name = {p.name: p for p in files}

    if dataset:
        if dataset not in by_name:
            raise FileNotFoundError(f"Dataset not found in {data_dir}: {dataset}")
        return [by_name[dataset]]

    return files


def compute_next_time_stats(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    unit_seconds = detect_time_unit_seconds(df)

    if "next_time" not in df.columns:
        return {
            "dataset": path.name,
            "rows": int(len(df)),
            "valid_next_time_rows": 0,
            "next_time_mean": np.nan,
            "next_time_median": np.nan,
            "next_time_mean_days": np.nan,
            "next_time_median_days": np.nan,
            "mae_days_mean_baseline": np.nan,
            "status": "missing_next_time_column",
        }

    next_time = pd.to_numeric(df["next_time"], errors="coerce")
    valid = next_time.dropna()

    if valid.empty:
        return {
            "dataset": path.name,
            "rows": int(len(df)),
            "valid_next_time_rows": 0,
            "next_time_mean": np.nan,
            "next_time_median": np.nan,
            "next_time_mean_days": np.nan,
            "next_time_median_days": np.nan,
            "mae_days_mean_baseline": np.nan,
            "status": "no_valid_next_time_rows",
        }

    next_time_mean = float(valid.mean())
    next_time_median = float(valid.median())
    next_time_mean_days = (next_time_mean * unit_seconds) / SECONDS_PER_DAY
    next_time_median_days = (next_time_median * unit_seconds) / SECONDS_PER_DAY
    mae_days_mean_baseline = (float((valid - next_time_mean).abs().mean()) * unit_seconds) / SECONDS_PER_DAY

    return {
        "dataset": path.name,
        "rows": int(len(df)),
        "valid_next_time_rows": int(len(valid)),
        "next_time_mean": next_time_mean,
        "next_time_median": next_time_median,
        "next_time_mean_days": float(next_time_mean_days),
        "next_time_median_days": float(next_time_median_days),
        "mae_days_mean_baseline": float(mae_days_mean_baseline),
        "status": "ok",
    }


def save_log1p_next_time_plot(path: Path, mode: str, plots_dir: Path, bins: int) -> tuple[Path | None, str]:
    df = pd.read_csv(path, low_memory=False)
    if "next_time" not in df.columns:
        return None, "missing_next_time_column"

    next_time = pd.to_numeric(df["next_time"], errors="coerce")
    valid = next_time.dropna().clip(lower=0.0)
    if valid.empty:
        return None, "no_valid_next_time_rows"

    log_values = np.log1p(valid.to_numpy(dtype=float))
    if log_values.size == 0:
        return None, "no_valid_next_time_rows"

    # Local import keeps plotting dependencies optional unless plotting is requested.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))

    if mode in {"hist", "both"}:
        ax.hist(log_values, bins=max(5, int(bins)), density=True, alpha=0.7, color="#4e79a7", label="Histogram")

    if mode in {"kde", "both"}:
        if log_values.size > 1 and float(np.std(log_values)) > 0.0:
            xs = np.linspace(float(log_values.min()), float(log_values.max()), 300)
            std = float(np.std(log_values, ddof=1)) if log_values.size > 1 else 0.0
            bw = max(1e-3, 1.06 * std * (log_values.size ** (-1.0 / 5.0)))
            z = (xs[:, None] - log_values[None, :]) / bw
            density = np.exp(-0.5 * z * z).mean(axis=1) / (bw * np.sqrt(2.0 * np.pi))
            ax.plot(xs, density, color="#e15759", linewidth=2.0, label="KDE")
        else:
            ax.axvline(float(log_values.mean()), color="#e15759", linewidth=2.0, label="KDE fallback")

    ax.set_title(f"log1p(next_time) - {path.name}")
    ax.set_xlabel("log1p(next_time)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"{path.stem}_log1p_next_time_{mode}.pdf"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, "ok"


def main() -> None:
    args = parse_args()
    files = select_files(args.data_dir, args.dataset)

    rows = [compute_next_time_stats(path) for path in files]
    stats_df = pd.DataFrame(rows)

    print(stats_df.to_string(index=False))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved stats table: {args.output_csv}")

    if args.plot != "none":
        print("\nPlot outputs:")
        for path in files:
            plot_path, status = save_log1p_next_time_plot(
                path=path,
                mode=args.plot,
                plots_dir=args.plots_dir,
                bins=args.hist_bins,
            )
            if plot_path is not None:
                print(f"{path.name}: {plot_path}")
            else:
                print(f"{path.name}: skipped ({status})")


if __name__ == "__main__":
    main()
