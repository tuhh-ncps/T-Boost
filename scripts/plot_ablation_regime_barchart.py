#!/usr/bin/env python3
"""Plot the MAE ablation table as a grouped bar chart.

The chart visualizes the three MAE columns from the table:
- Mean Baseline
- Without TS
- With TS

It also annotates the gain (%) above the With TS bar for each dataset.

Example:
  python scripts/plot_ablation_regime_barchart.py --output results/ablation_regime_barchart.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    "BPI12 (W)",
    "Helpdesk16",  # matches the table text
    "BPI12",
    "Helpdesk17",
    "BPI13",
    "BPI17",
    "BPI20 DD",
    "BPI20 ID",
    "BPI20 PD",
    "BPI20 RP",
    "BPI20 TC",
]

MEAN_BASELINE = [1.94, 3.51, 0.73, 10.76, 2.25, 1.00, 2.48, 10.21, 9.31, 2.57, 5.48]
WITHOUT_TS = [1.26, 2.24, 0.39, 6.71, 0.85, 0.43, 1.30, 2.69, 4.90, 1.46, 1.85]
WITH_TS = [0.63, 1.53, 0.28, 5.86, 0.54, 0.40, 0.89, 2.09, 3.23, 1.05, 1.26]
GAIN = [50.0, 31.7, 28.2, 12.7, 36.5, 7.0, 31.5, 22.3, 34.1, 28.1, 31.9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the ablation MAE table as a grouped bar chart.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ablation_regime_barchart.pdf"),
        help="Output figure path (default: results/ablation_regime_barchart.pdf)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x = np.arange(len(DATASETS))
    width = 0.26

    # Use a clean, publication-style palette.
    colors = {
        "Mean Baseline": "#4c78a8",
        "Without TS": "#f58518",
        "With TS": "#54a24b",
        "Gain": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    # bars1 = ax.bar(x - width, MEAN_BASELINE, width, label="Mean Baseline", color=colors["Mean Baseline"])
    bars2 = ax.bar(x, WITHOUT_TS, width, label="Without TS", color=colors["Without TS"])
    bars3 = ax.bar(x + width, WITH_TS, width, label="With TS", color=colors["With TS"])

    for idx, (bar, gain) in enumerate(zip(bars3, GAIN)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(WITH_TS) * 0.03,
            f"{gain:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
            color=colors["Gain"],
        )

    ax.set_ylabel("MAE (days)")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=2, frameon=False, loc="upper right")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart to: {args.output}")


if __name__ == "__main__":
    main()