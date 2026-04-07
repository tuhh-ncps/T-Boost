#!/usr/bin/env python3
"""Plot federated learning metrics for the MLP experiment.

This script reads the CSV metrics produced by the federated training run and
creates server plots plus per-activity client plots:
- server_metrics.pdf: server metrics by round, with MAE/RMSE shown in days
- client_metrics_by_activity/: one 2x2 metrics plot per activity
- client_epoch_loss_by_activity/: one epoch-loss plot per activity
- mae_per_activity.csv and plots under client_metrics_by_activity/plots/: MAE per activity from predictions
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

SECONDS_PER_DAY = 86400.0

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

logger = logging.getLogger(__name__)


def load_metrics(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the server, client, and epoch CSV files."""
    server_file = result_dir / "server_metrics.csv"
    client_file = result_dir / "client_metrics.csv"
    epoch_file = result_dir / "epoch_metrics.csv"

    for path in (server_file, client_file, epoch_file):
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")

    server_df = pd.read_csv(server_file)
    client_df = pd.read_csv(client_file)
    epoch_df = pd.read_csv(epoch_file)

    logger.info("Loaded %s server rows, %s client rows, %s epoch rows", len(server_df), len(client_df), len(epoch_df))
    return server_df, client_df, epoch_df


def _sorted_round_labels(values: Iterable) -> list[int]:
    labels = []
    for value in values:
        if pd.isna(value):
            continue
        labels.append(int(value))
    return sorted(dict.fromkeys(labels))


def _infer_round_labels(df: pd.DataFrame, server_rounds: list[int]) -> pd.Series:
    """Infer round labels when the CSV only contains a flat round column.

    The current metrics logs were written in contiguous blocks per round, so we
    split the rows evenly across the known server rounds and map each block back
    to the actual server round numbers.
    """
    if df.empty:
        return pd.Series(dtype=int)

    if "round" in df.columns:
        rounds = _sorted_round_labels(df["round"].unique())
        if len(rounds) > 1:
            return df["round"].astype(int)

    if not server_rounds:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    block_indices = np.array_split(np.arange(len(df)), len(server_rounds))
    inferred = np.empty(len(df), dtype=int)
    for round_label, block in zip(server_rounds, block_indices):
        inferred[block] = int(round_label)
    return pd.Series(inferred, index=df.index)


def _convert_to_days(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") / SECONDS_PER_DAY


def _style_round_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return safe.strip("_") or "activity"


def _load_prediction_metrics(prediction_file: Path) -> pd.DataFrame:
    """Load prediction rows and compute absolute error per activity."""
    if not prediction_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")

    df = pd.read_csv(
        prediction_file,
        usecols=["activity", "actual_next_time", "prediction_next_time"],
        low_memory=False,
    )
    df["activity"] = df["activity"].astype(str)
    df["actual_next_time"] = pd.to_numeric(df["actual_next_time"], errors="coerce")
    df["prediction_next_time"] = pd.to_numeric(df["prediction_next_time"], errors="coerce")
    df = df.dropna(subset=["activity", "actual_next_time", "prediction_next_time"]).copy()
    df["abs_error"] = (df["actual_next_time"] - df["prediction_next_time"]).abs()

    mae_df = (
        df.groupby("activity", as_index=False)
        .agg(n_samples=("abs_error", "size"), mae=("abs_error", "mean"))
        .sort_values("mae", ascending=False)
        .reset_index(drop=True)
    )
    return mae_df


def plot_mae_per_activity(mae_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot MAE per activity as a line chart with one point per activity."""
    if mae_df.empty:
        logger.warning("MAE per activity DataFrame is empty")
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "mae_per_activity.csv"
    mae_df.to_csv(csv_path, index=False)

    plot_df = mae_df.copy()
    plot_df["activity_label"] = plot_df["activity"].astype(str)
    fig_height = max(6, 0.35 * len(plot_df) + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.plot(plot_df["activity_label"], plot_df["mae"], marker="o", linewidth=2.0, color="#1f77b4")
    ax.set_title("MAE per Activity")
    ax.set_xlabel("Activity")
    ax.set_ylabel("MAE (seconds)")
    ax.tick_params(axis="x", labelrotation=35)
    _style_round_axis(ax)
    plt.tight_layout()

    png_path = plot_dir / "mae_per_activity.png"
    pdf_path = plot_dir / "mae_per_activity.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved %s", csv_path)
    logger.info("Saved %s", png_path)
    logger.info("Saved %s", pdf_path)


def _select_weakest_clients(
    client_df: pd.DataFrame,
    metric_col: str,
    n_weakest: int,
) -> list[str]:
    """Return the client IDs with the worst average value for the given metric."""
    if n_weakest <= 0 or client_df.empty or metric_col not in client_df.columns:
        return []

    summary = (
        client_df.assign(_metric=pd.to_numeric(client_df[metric_col], errors="coerce"))
        .groupby("client_id", as_index=True)["_metric"]
        .mean(numeric_only=True)
        .dropna()
        .sort_values(ascending=False)
    )
    return summary.head(n_weakest).index.tolist()


def _filter_client_frames(
    client_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
    weakest_metric: str,
    exclude_weakest_activities: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    weakest_clients = _select_weakest_clients(client_df, weakest_metric, exclude_weakest_activities)
    if not weakest_clients:
        return client_df, epoch_df, []

    filtered_client_df = (
        client_df[~client_df["client_id"].isin(weakest_clients)].copy()
        if "client_id" in client_df.columns
        else client_df.copy()
    )
    filtered_epoch_df = (
        epoch_df[~epoch_df["client_id"].isin(weakest_clients)].copy()
        if "client_id" in epoch_df.columns
        else epoch_df.copy()
    )
    return filtered_client_df, filtered_epoch_df, weakest_clients


def plot_server_metrics(server_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot global server metrics and convert MAE/RMSE to days."""
    if server_df.empty:
        logger.warning("Server metrics DataFrame is empty")
        return

    plot_df = server_df.copy()
    plot_df["round"] = pd.to_numeric(plot_df["round"], errors="coerce").astype(int)
    plot_df = plot_df.sort_values("round")
    plot_df["mae_days"] = _convert_to_days(plot_df["mae"])
    plot_df["rmse_days"] = _convert_to_days(plot_df["rmse"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Global Model Metrics by Round", fontsize=15, fontweight="bold")

    axes[0, 0].plot(plot_df["round"], plot_df["mae_days"], marker="o", linewidth=2.2, color="#1f77b4")
    axes[0, 0].set_title("Mean Absolute Error")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("MAE (days)")
    _style_round_axis(axes[0, 0])

    axes[0, 1].plot(plot_df["round"], plot_df["rmse_days"], marker="o", linewidth=2.2, color="#ff7f0e")
    axes[0, 1].set_title("Root Mean Squared Error")
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("RMSE (days)")
    _style_round_axis(axes[0, 1])

    axes[1, 0].plot(plot_df["round"], plot_df["r2"], marker="o", linewidth=2.2, color="#2ca02c")
    axes[1, 0].set_title("Coefficient of Determination")
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("R2")
    axes[1, 0].set_ylim(-1.2, 1.2)
    _style_round_axis(axes[1, 0])

    axes[1, 1].plot(plot_df["round"], plot_df["normalized_mae"], marker="o", linewidth=2.2, color="#d62728")
    axes[1, 1].set_title("Normalized Mean Absolute Error")
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("Normalized MAE")
    axes[1, 1].set_ylim(-1.2, 1.2)
    _style_round_axis(axes[1, 1])

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = output_dir / "server_metrics.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_client_metrics(
    client_df: pd.DataFrame,
    server_rounds: list[int],
    output_dir: Path,
    exclude_weakest_activities: int = 0,
    weakest_activity_metric: str = "rmse",
    mae_by_activity: dict[str, float] | None = None,
) -> None:
    """Plot client metrics as combined and per-activity views across rounds."""
    if client_df.empty:
        logger.warning("Client metrics DataFrame is empty")
        return

    filtered_client_df, _, weakest_clients = _filter_client_frames(
        client_df=client_df,
        epoch_df=pd.DataFrame(),
        weakest_metric=weakest_activity_metric,
        exclude_weakest_activities=exclude_weakest_activities,
    )
    if weakest_clients:
        logger.info(
            "Excluding %s weakest activities by %s: %s",
            len(weakest_clients),
            weakest_activity_metric,
            ", ".join(weakest_clients),
        )

    plot_df = filtered_client_df.copy()
    plot_df["plot_round"] = _infer_round_labels(plot_df, server_rounds)
    plot_df["plot_round"] = plot_df["plot_round"].astype(int)
    plot_df = plot_df.sort_values(["client_id", "plot_round"])

    clients = list(dict.fromkeys(plot_df["client_id"].tolist()))

    metrics = [
        ("mae", "Mean Absolute Error", "MAE"),
        ("rmse", "Root Mean Squared Error", "RMSE"),
        ("r2", "Coefficient of Determination", "R2"),
        ("normalized_mae", "Normalized Mean Absolute Error", "Normalized MAE"),
    ]

    # Combined plot (all activities on one figure)
    palette = sns.color_palette("husl", n_colors=max(1, len(clients)))
    color_map = {client_id: palette[idx] for idx, client_id in enumerate(clients)}

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle("Client Metrics by Round", fontsize=13, fontweight="bold", y=0.985)

    legend_handles = []
    legend_labels = []
    for ax, (metric_col, title, y_label) in zip(axes.flat, metrics):
        y_label = f"{y_label} (days)" if metric_col in {"mae", "rmse"} else y_label
        for client_id in clients:
            client_rows = plot_df[plot_df["client_id"] == client_id].sort_values("plot_round")
            if client_rows.empty or metric_col not in client_rows.columns:
                continue
            y_values = pd.to_numeric(client_rows[metric_col], errors="coerce")
            if metric_col in {"mae", "rmse"}:
                y_values = y_values / SECONDS_PER_DAY
            line = ax.plot(
                client_rows["plot_round"],
                y_values,
                marker="o",
                linewidth=1.4,
                markersize=3,
                alpha=0.85,
                color=color_map[client_id],
                label=client_id,
            )
            if not legend_handles:
                legend_handles.extend(line)
                legend_labels.append(client_id)
            elif client_id not in legend_labels:
                legend_handles.extend(line)
                legend_labels.append(client_id)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Round")
        ax.set_ylabel(y_label)
        if metric_col in {"r2", "normalized_mae"}:
            ax.set_ylim(-1.2, 1.2)
        _style_round_axis(ax)
        ax.tick_params(axis="both", labelsize=8)
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=6,
        frameon=True,
        ncol=4,
        title="Client",
        title_fontsize=7,
        columnspacing=1.0,
        handlelength=1.6,
        handletextpad=0.4,
    )
    plt.tight_layout(rect=(0, 0.13, 1, 0.95), w_pad=1.0, h_pad=1.2)

    output_path = output_dir / "client_metrics.pdf"
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("Saved %s", output_path)

    by_activity_dir = output_dir / "client_metrics_by_activity"
    by_activity_dir.mkdir(parents=True, exist_ok=True)

    for client_id in clients:
        client_rows = plot_df[plot_df["client_id"] == client_id].sort_values("plot_round")
        if client_rows.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Client Metrics by Round: {client_id}", fontsize=13, fontweight="bold", y=0.98)

        for ax, (metric_col, title, y_label) in zip(axes.flat, metrics):
            if metric_col not in client_rows.columns:
                continue
            y_values = pd.to_numeric(client_rows[metric_col], errors="coerce")
            if metric_col in {"mae", "rmse"}:
                y_values = y_values / SECONDS_PER_DAY
                y_label = f"{y_label} (days)"

            ax.plot(
                client_rows["plot_round"],
                y_values,
                marker="o",
                linewidth=1.8,
                markersize=3.8,
                alpha=0.9,
                color="#1f77b4",
            )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Round")
            ax.set_ylabel(y_label)
            if metric_col == "mae" and mae_by_activity is not None and client_id in mae_by_activity:
                ref_value = mae_by_activity[client_id] / SECONDS_PER_DAY
                ax.axhline(ref_value, color="#d62728", linestyle="--", linewidth=1.4, alpha=0.85)
                ax.text(
                    0.99,
                    0.95,
                    f"CSV MAE = {ref_value:.4f} days",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="#d62728",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
            if metric_col in {"r2", "normalized_mae"}:
                ax.set_ylim(-1.2, 1.2)
            _style_round_axis(ax)

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        output_path = by_activity_dir / f"client_metrics_{_safe_name(client_id)}.pdf"
        plt.savefig(output_path, bbox_inches="tight", format="pdf")
        plt.close(fig)

    logger.info("Saved per-activity client metric plots to %s", by_activity_dir)


def plot_client_epoch_loss(
    epoch_df: pd.DataFrame,
    client_df: pd.DataFrame,
    server_rounds: list[int],
    output_dir: Path,
    exclude_weakest_activities: int = 0,
    weakest_activity_metric: str = "rmse",
) -> None:
    """Plot combined and per-activity epoch losses on one timeline."""
    if epoch_df.empty:
        logger.warning("Epoch metrics DataFrame is empty")
        return

    _, filtered_epoch_df, weakest_clients = _filter_client_frames(
        client_df=client_df,
        epoch_df=epoch_df,
        weakest_metric=weakest_activity_metric,
        exclude_weakest_activities=exclude_weakest_activities,
    )
    if weakest_clients:
        logger.info(
            "Excluding %s weakest activities by %s from epoch plot: %s",
            len(weakest_clients),
            weakest_activity_metric,
            ", ".join(weakest_clients),
        )

    plot_df = filtered_epoch_df.copy()
    plot_df["plot_round"] = _infer_round_labels(plot_df, server_rounds)
    plot_df["plot_round"] = plot_df["plot_round"].astype(int)

    round_order = _sorted_round_labels(plot_df["plot_round"].unique())
    round_index = {round_label: idx for idx, round_label in enumerate(round_order)}
    plot_df["round_index"] = plot_df["plot_round"].map(round_index).astype(int)
    plot_df["epoch"] = pd.to_numeric(plot_df["epoch"], errors="coerce").astype(int)
    plot_df["loss"] = pd.to_numeric(plot_df["loss"], errors="coerce")

    local_epochs = int(plot_df["epoch"].max()) if not plot_df.empty else 1
    local_epochs = max(local_epochs, 1)
    plot_df["global_epoch"] = plot_df["round_index"] * local_epochs + plot_df["epoch"]

    clients = list(dict.fromkeys(plot_df["client_id"].tolist()))

    # Combined plot (all activities on one figure)
    palette = sns.color_palette("tab20", n_colors=max(1, len(clients)))
    color_map = {client_id: palette[idx % len(palette)] for idx, client_id in enumerate(clients)}

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.suptitle("Client Loss Across Concatenated Epochs", fontsize=15, fontweight="bold")

    for client_id in clients:
        client_rows = plot_df[plot_df["client_id"] == client_id].sort_values("global_epoch")
        if client_rows.empty:
            continue
        ax.plot(
            client_rows["global_epoch"],
            client_rows["loss"],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            alpha=0.9,
            color=color_map[client_id],
            label=client_id,
        )

    for round_idx in range(1, len(round_order)):
        ax.axvline(round_idx * local_epochs + 0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Concatenated Epoch")
    ax.set_ylabel("Loss")
    _style_round_axis(ax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=True)

    plt.tight_layout(rect=(0, 0, 0.84, 0.95))
    output_path = output_dir / "client_epoch_loss.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("Saved %s", output_path)

    by_activity_dir = output_dir / "client_epoch_loss_by_activity"
    by_activity_dir.mkdir(parents=True, exist_ok=True)

    for client_id in clients:
        client_rows = plot_df[plot_df["client_id"] == client_id].sort_values("global_epoch")
        if client_rows.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 5))
        fig.suptitle(f"Client Loss Across Concatenated Epochs: {client_id}", fontsize=13, fontweight="bold")
        ax.plot(
            client_rows["global_epoch"],
            client_rows["loss"],
            marker="o",
            linewidth=1.8,
            markersize=3.2,
            alpha=0.9,
            color="#1f77b4",
        )

        for round_idx in range(1, len(round_order)):
            ax.axvline(round_idx * local_epochs + 0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.45)

        ax.set_xlabel("Concatenated Epoch")
        ax.set_ylabel("Loss")
        _style_round_axis(ax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        output_path = by_activity_dir / f"client_epoch_loss_{_safe_name(client_id)}.pdf"
        plt.savefig(output_path, dpi=220, bbox_inches="tight", format="pdf")
        plt.close(fig)

    logger.info("Saved per-activity epoch-loss plots to %s", by_activity_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot federated MLP metrics from CSV files")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/flwr_mlp"),
        help="Directory containing server_metrics.csv, client_metrics.csv, and epoch_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots. Defaults to the result directory.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--exclude-weakest-activities",
        type=int,
        default=0,
        help="Exclude the N weakest activities from the client plots (0 = keep all)",
    )
    parser.add_argument(
        "--weakest-activity-metric",
        choices=["rmse", "mae", "normalized_mae"],
        default="rmse",
        help="Metric used to rank weakest activities before filtering",
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=Path("results/mlp/predictions/BPI12_prediction.csv"),
        help="Prediction CSV used to compute MAE per activity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result_dir = args.result_dir
    output_dir = args.output_dir or result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    server_df, client_df, epoch_df = load_metrics(result_dir)
    server_rounds = _sorted_round_labels(server_df["round"].unique())
    mae_df = None
    try:
        mae_df = _load_prediction_metrics(args.prediction_file)
    except FileNotFoundError as exc:
        logger.warning("Skipping MAE per activity plot: %s", exc)

    plot_server_metrics(server_df, output_dir)
    plot_client_metrics(
        client_df,
        server_rounds,
        output_dir,
        exclude_weakest_activities=args.exclude_weakest_activities,
        weakest_activity_metric=args.weakest_activity_metric,
        mae_by_activity=None if mae_df is None else dict(zip(mae_df["activity"], mae_df["mae"])),
    )
    plot_client_epoch_loss(
        epoch_df,
        client_df,
        server_rounds,
        output_dir,
        exclude_weakest_activities=args.exclude_weakest_activities,
        weakest_activity_metric=args.weakest_activity_metric,
    )

    if mae_df is not None:
        plot_mae_per_activity(mae_df, output_dir / "client_metrics_by_activity")

    logger.info("All plots saved to %s", output_dir)


if __name__ == "__main__":
    main()
