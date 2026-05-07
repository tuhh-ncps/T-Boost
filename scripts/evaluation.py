#!/usr/bin/env python3
"""Evaluate time-regime distributions and characteristics.

Generates three analytical plots:
1. Bin statistics (median, variance, IQR for each regime)
2. Boxplot: regime vs next_time (log-scale)
3. Entropy plot: entropy of regime probs vs next_time (optional, requires probability columns)

Input:
    CSV file with 'time_regime' (q1, q2, q3, q4 labels) and 'next_time' columns
    Optionally includes 'prob_regime_*' or 'predicted_time_regime_q*_pct' columns for entropy calculation

Example:
    python scripts/evaluation.py --csv data_csv/BPI12.csv
    python scripts/evaluation.py --csv results/decision_tree_time_regime/output/BPI12_train_with_time_regime.csv

Usage:
    --csv PATH          Input CSV with time_regime and next_time columns (required)
    --output-dir PATH   Output directory for plots (default: results/evaluation)
    --log-scale         Use log-scale for next_time (default: True)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate time-regime distributions")
    parser.add_argument("--csv", required=True, type=Path, help="Input CSV with time_regime and next_time columns")
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"), help="Output directory for plots")
    parser.add_argument("--log-scale", action="store_true", default=True, help="Use log-scale for next_time")
    return parser.parse_args()


def plot_bin_statistics(df: pd.DataFrame, out_dir: Path, dataset_name: str) -> None:
    """Plot median, variance, and IQR for each regime."""
    regimes = sorted(df["time_regime"].unique())
    
    stats = []
    for regime in regimes:
        regime_data = df[df["time_regime"] == regime]["next_time"]
        if len(regime_data) == 0:
            continue
        
        median = regime_data.median()
        variance = regime_data.var()
        q1 = regime_data.quantile(0.25)
        q3 = regime_data.quantile(0.75)
        iqr = q3 - q1
        
        stats.append({
            "regime": regime,
            "median": median,
            "variance": variance,
            "iqr": iqr,
            "q1": q1,
            "q3": q3,
            "count": len(regime_data),
        })
    
    stats_df = pd.DataFrame(stats)
    print("\nBin Statistics:")
    print(stats_df.to_string(index=False))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Median (log1p(next_time))
    axes[0].bar(stats_df["regime"], stats_df["median"], color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Regime")
    axes[0].set_ylabel("Median log1p(next_time)")
    axes[0].set_title("Median by Regime (log1p)")
    axes[0].grid(True, alpha=0.3)
    
    # Variance (log1p(next_time))
    axes[1].bar(stats_df["regime"], stats_df["variance"], color="coral", alpha=0.7)
    axes[1].set_xlabel("Regime")
    axes[1].set_ylabel("Variance (log1p)")
    axes[1].set_title("Variance by Regime (log1p)")
    axes[1].grid(True, alpha=0.3)
    
    # IQR (log1p(next_time))
    axes[2].bar(stats_df["regime"], stats_df["iqr"], color="mediumseagreen", alpha=0.7)
    axes[2].set_xlabel("Regime")
    axes[2].set_ylabel("IQR (log1p)")
    axes[2].set_title("IQR by Regime (log1p)")
    axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    out_path = out_dir / f"{dataset_name}_01_bin_statistics.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_regime_vs_next_time(df: pd.DataFrame, out_dir: Path, dataset_name: str, log_scale: bool = True) -> None:
    """Plot boxplot of next_time for each regime (log-scale)."""
    regimes = sorted(df["time_regime"].unique())
    data_by_regime = [df[df["time_regime"] == regime]["next_time"].values for regime in regimes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data_by_regime, tick_labels=regimes, patch_artist=True)
    
    # Color boxes
    colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFD700"]
    for patch, color in zip(bp["boxes"], colors[: len(regimes)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Data are already in log1p(next_time); show that in the label.
    if log_scale:
        # user requested axis log-scaling on top of log1p values (rare); keep support
        ax.set_yscale("log")
        ylabel = "log1p(next_time) (axis log-scaled)"
    else:
        ylabel = "log1p(next_time)"
    
    ax.set_xlabel("Time Regime")
    ax.set_ylabel(ylabel)
    ax.set_title("Next-Time Distribution by Regime (Boxplot)")
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.tight_layout()
    out_path = out_dir / f"{dataset_name}_02_regime_boxplot.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_entropy_vs_next_time(df: pd.DataFrame, out_dir: Path, dataset_name: str) -> None:
    """Plot entropy of regime probabilities vs next_time.
    
    Requires 'entropy' column or 'prob_regime_*' or 'predicted_time_regime_*_pct' columns.
    """
    # Check if entropy column exists
    if "entropy" in df.columns:
        entropy_col = "entropy"
    else:
        # Try to find probability columns (two possible formats)
        prob_cols = [col for col in df.columns if col.startswith("prob_regime_")]
        
        if not prob_cols:
            # Try the trainer output format: predicted_time_regime_q*_pct
            prob_cols = sorted([col for col in df.columns if col.startswith("predicted_time_regime_q") and col.endswith("_pct")])
        
        if not prob_cols:
            print("Warning: No entropy or probability columns found. Skipping entropy plot.")
            return
        
        # Calculate entropy: -sum(p * log(p))
        probs = df[prob_cols].values
        eps = 1e-12
        entropy_col_vals = -np.sum(probs * np.log(probs + eps), axis=1)
        df["entropy"] = entropy_col_vals
        entropy_col = "entropy"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    regimes = sorted(df["time_regime"].unique())
    colors_map = {"q1": "#FF9999", "q2": "#66B2FF", "q3": "#99FF99", "q4": "#FFD700"}
    
    for regime in regimes:
        regime_data = df[df["time_regime"] == regime]
        color = colors_map.get(regime, "gray")
        ax.scatter(
            regime_data["next_time"],
            regime_data[entropy_col],
            label=regime,
            alpha=0.5,
            s=20,
            color=color,
        )
    
    ax.set_xlabel("log1p(next_time)")
    ax.set_ylabel("Entropy of regime probabilities")
    ax.set_title("Entropy vs Next-Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    out_path = out_dir / f"{dataset_name}_03_entropy_scatter.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    
    # Validate input file
    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(args.csv)
    
    # Validate required columns
    if "time_regime" not in df.columns:
        raise ValueError(f"Column 'time_regime' not found in {args.csv}")
    if "next_time" not in df.columns:
        raise ValueError(f"Column 'next_time' not found in {args.csv}")
    
    # Clean data: remove rows with NaN in time_regime or next_time
    df = df.dropna(subset=["time_regime", "next_time"]).copy()
    df["next_time"] = pd.to_numeric(df["next_time"], errors="coerce")
    df = df[df["next_time"] > 0].copy()  # Keep only positive next_time

    # Convert next_time to log-scale (log1p) for all analysis and plots
    df["next_time"] = np.log1p(df["next_time"])  # now contains log1p(next_time)
    
    dataset_name = args.csv.stem
    print(f"Dataset: {dataset_name}")
    print(f"Total rows: {len(df)}")
    print(f"Regimes: {sorted(df['time_regime'].unique())}")
    
    # Generate plots (all use log1p(next_time) now)
    plot_bin_statistics(df, args.output_dir, dataset_name)
    # boxplot will show log1p(next_time) values; do not re-apply log scale
    plot_regime_vs_next_time(df, args.output_dir, dataset_name, log_scale=False)
    # entropy x-axis expects log1p(next_time) values (no axis log-scaling)
    plot_entropy_vs_next_time(df, args.output_dir, dataset_name)
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
