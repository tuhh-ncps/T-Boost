#!/usr/bin/env python3
"""
Plot learning curves from all trained MLP datasets on a single plot.

Scans the results/mlp/plots/ directory for all *_learning_curve.csv files
and plots them overlay with distinct colors for each dataset.

Usage:
  python scripts/mlp_plot_learning_curves.py --input-dir results/mlp/plots \\
            --output results/mlp/plots/all_learning_curves.png --metrics train_loss val_loss

Options:
  --input-dir DIR       Directory containing learning_curve.csv files (default: results/mlp/plots)
  --output PATH         Output PNG file path (default: results/mlp/plots/all_learning_curves.png)
    --metrics METRIC ...  Metrics to plot: train_loss, val_loss, train_mae_days, val_mae_days (default: train_loss val_loss)
  --figsize WxH         Figure size as WxH (default: 14x7)
  --title TITLE         Plot title (default: Learning Curves - All Datasets)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("results/mlp/plots")
DEFAULT_OUTPUT = Path("results/mlp/plots/all_learning_curves.png")
VALID_METRICS = ["train_loss", "val_loss", "train_mae_days", "val_mae_days"]
DEFAULT_METRICS = ["train_loss", "val_loss"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all learning curves from MLP training on a single plot."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing learning_curve.csv files (default: results/mlp/plots)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG file path (default: results/mlp/plots/all_learning_curves.png)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=str,
        default=DEFAULT_METRICS,
        choices=VALID_METRICS,
        help="Metrics to plot (default: train_loss val_loss)",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="14x7",
        help="Figure size as WxH (default: 14x7)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Learning Curves - All Datasets",
        help="Plot title (default: Learning Curves - All Datasets)",
    )
    return parser.parse_args()


def parse_figsize(figsize_str: str) -> tuple[float, float]:
    try:
        w, h = figsize_str.split("x")
        return (float(w), float(h))
    except (ValueError, AttributeError):
        return (14, 7)


def find_learning_curve_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.glob("*_learning_curve.csv"))
    if not files:
        raise FileNotFoundError(f"No learning_curve.csv files found in {input_dir}")
    return files


def extract_dataset_name(csv_path: Path) -> str:
    stem = csv_path.stem
    if stem.endswith("_learning_curve"):
        return stem[: -len("_learning_curve")]
    return stem
# Optional mapping from extracted dataset name -> display title used in legends
# Edit this to customize displayed dataset names (e.g. map 'bpi_12_w' -> 'BPI12 (W)')
FILE_TITLE_MAP: dict[str, str] = {
    "BPI12": "BPI12",
    "BPI13": "BPI13",
    "bpi_12_w": "BPI12 (W)",
    "helpdesk": "Helpdesk16",
    "helpdesk17": "Helpdesk17",
}


def _format_metric_label(metric: str) -> str:
    if metric == "train_loss":
        return "train loss"
    if metric == "val_loss":
        return "val. loss"
    if metric == "train_mae_days":
        return "train MAE (days)"
    if metric == "val_mae_days":
        return "val. MAE (days)"
    return metric.replace("_", " ")


def plot_learning_curves(
    csv_files: List[Path],
    metrics: List[str],
    output_path: Path,
    figsize: tuple[float, float],
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    # Use a colormap with enough distinct colors
    cmap = plt.get_cmap("tab20")
    num_files = len(csv_files)
    colors = [cmap(i % 20) for i in range(num_files)]
    line_styles = {
        "train_loss": "-",
        "val_loss": "--",
        "train_mae_days": ":",
        "val_mae_days": "-.",
    }

    for csv_file, color in zip(csv_files, colors):
        dataset_name = extract_dataset_name(csv_file)
        display_name = FILE_TITLE_MAP.get(dataset_name, FILE_TITLE_MAP.get(dataset_name.lower(), dataset_name))
        try:
            df = pd.read_csv(csv_file, low_memory=False)

            epochs = pd.to_numeric(df["epoch"], errors="coerce").to_numpy()
            plotted_any_metric = False

            # Determine best epoch for this dataset: prefer `val_loss` if present,
            # otherwise fall back to `train_loss`. We'll truncate plotted series
            # at the best epoch so curves stop where training selected the best model.
            best_epoch = None
            for cand in ("val_loss", "train_loss"):
                if cand in df.columns:
                    cand_series = pd.to_numeric(df[cand], errors="coerce").to_numpy()
                    if np.any(~np.isnan(cand_series)):
                        # numpy.nanargmin ignores NaNs
                        try:
                            idx = int(np.nanargmin(cand_series))
                            best_epoch = int(idx + 1)
                        except ValueError:
                            best_epoch = None
                        break

            for idx, metric in enumerate(metrics):
                if metric not in df.columns:
                    print(f"Skip {dataset_name}: column '{metric}' not found")
                    continue

                values = pd.to_numeric(df[metric], errors="coerce").to_numpy()
                valid_mask = ~np.isnan(epochs) & ~np.isnan(values)
                if not np.any(valid_mask):
                    print(f"Skip {dataset_name}: no valid {metric} values")
                    continue

                epochs_valid = epochs[valid_mask]
                values_valid = values[valid_mask]
                # If a best epoch was found, trim plotted points at that epoch.
                if best_epoch is not None:
                    trim_mask = epochs_valid <= best_epoch
                    if not np.any(trim_mask):
                        print(f"Skip {dataset_name}: no {metric} values before best_epoch={best_epoch}")
                        continue
                    epochs_valid = epochs_valid[trim_mask]
                    values_valid = values_valid[trim_mask]
                label = f"{display_name} { _format_metric_label(metric) }"
                ax.plot(
                    epochs_valid,
                    values_valid,
                    label=label,
                    linewidth=2.0,
                    color=color,
                    linestyle=line_styles.get(metric, "-"),
                    alpha=0.95 if idx == 0 else 0.8,
                )
                plotted_any_metric = True

            if not plotted_any_metric:
                print(f"Skip {dataset_name}: no requested metrics found")
            
        except Exception as exc:  # noqa: BLE001
            print(f"Error loading {csv_file}: {exc}")
            continue

    # Title removed per user request.
    ax.set_xlabel("Epoch", fontsize=12)
    if len(metrics) == 1:
        ax.set_ylabel(metrics[0].replace("_", " ").title(), fontsize=12)
    else:
        ax.set_ylabel("Metric Value", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved combined learning curve plot: {output_path}")


def main() -> None:
    args = parse_args()
    figsize = parse_figsize(args.figsize)

    csv_files = find_learning_curve_files(args.input_dir)
    print(f"Found {len(csv_files)} learning curve files")

    plot_learning_curves(csv_files, args.metrics, args.output, figsize, args.title)


if __name__ == "__main__":
    main()
