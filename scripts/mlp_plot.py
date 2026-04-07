#!/usr/bin/env python3
"""Plot MLP metrics.

Supports two modes:
- epoch-loss: train/validation loss per epoch from model checkpoint
- event-nmae: normalized MAE by event from prediction CSV (train vs validation)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DEFAULT_RESULT_DIR = Path("results/mlp")
DEFAULT_DATASET = "BPI12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MLP epoch loss or event-level normalized MAE")
    parser.add_argument(
        "--mode",
        type=str,
        default="epoch-loss",
        choices=["epoch-loss", "event-nmae", "case-nmae"],
        help="Plot mode: epoch-loss or event-nmae (case-nmae is kept as alias)",
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset stem, e.g. BPI12 or BPI17")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR, help="Base result directory")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to saved MLP model")
    parser.add_argument("--prediction-path", type=Path, default=None, help="Path to prediction CSV")
    parser.add_argument("--output", type=Path, default=None, help="Path to output plot PNG")
    parser.add_argument("--csv-output", type=Path, default=None, help="Path to output epoch CSV")
    parser.add_argument("--title", type=str, default=None, help="Optional custom plot title")
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=20000,
        help="Rolling window for event-level NMAE smoothing (default: 20000)",
    )
    return parser.parse_args()


def resolve_epoch_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    model_path = args.model_path or (args.result_dir / "models" / f"{args.dataset}_mlp_next_time.pt")
    output_path = args.output or (args.result_dir / "plots" / f"{args.dataset}_mlp_epoch_error.png")
    csv_path = args.csv_output or (args.result_dir / "plots" / f"{args.dataset}_mlp_epoch_error.csv")
    return model_path, output_path, csv_path


def resolve_event_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    prediction_path = args.prediction_path or (args.result_dir / "predictions" / f"{args.dataset}_prediction.csv")
    output_path = args.output or (args.result_dir / "plots" / f"{args.dataset}_prediction_nmae_event_line.png")
    csv_path = args.csv_output or (args.result_dir / "plots" / f"{args.dataset}_prediction_nmae_event_summary.csv")
    return prediction_path, output_path, csv_path


def plot_epoch_loss(args: argparse.Namespace) -> None:
    model_path, output_path, csv_output_path = resolve_epoch_paths(args)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "history" not in checkpoint:
        raise ValueError("Checkpoint does not contain training history")

    history = checkpoint["history"]
    train_error = np.array(history.get("train_loss", []), dtype=float)
    validation_error = np.array(history.get("val_loss", []), dtype=float)

    if len(train_error) == 0:
        raise ValueError("No train_loss found in checkpoint history")

    if len(validation_error) == 0:
        validation_error = np.full_like(train_error, np.nan)

    min_len = min(len(train_error), len(validation_error))
    if min_len > 0:
        train_error = train_error[:min_len]
        validation_error = validation_error[:min_len]

    epochs = np.arange(1, len(train_error) + 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_error, label="Training Loss", linewidth=2)
    if not np.isnan(validation_error).all():
        plt.plot(epochs, validation_error, label="Validation Loss", linewidth=2)

    plt.title(args.title or "MLP Error per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    df = pd.DataFrame(
        {
            "epoch": epochs,
            "training_loss": train_error,
            "validation_loss": validation_error,
        }
    )
    df.to_csv(csv_output_path, index=False)

    print(f"Saved plot: {output_path}")
    print(f"Saved CSV: {csv_output_path}")


def _prepare_event_nmae(df: pd.DataFrame, split_name: str, rolling_window: int) -> pd.DataFrame:
    required = {"split", "actual_next_time", "prediction_next_time"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for event-nmae mode: {missing}")

    split_df = df[df["split"].astype(str).str.lower() == split_name].copy()
    if split_df.empty:
        return split_df

    split_df["actual_next_time"] = pd.to_numeric(split_df["actual_next_time"], errors="coerce")
    split_df["prediction_next_time"] = pd.to_numeric(split_df["prediction_next_time"], errors="coerce")
    split_df = split_df.dropna(subset=["actual_next_time", "prediction_next_time"])
    if split_df.empty:
        return split_df

    split_df["abs_error"] = (split_df["actual_next_time"] - split_df["prediction_next_time"]).abs()

    mean_next_time = float(split_df["actual_next_time"].mean())
    if np.isclose(mean_next_time, 0.0):
        split_df["nmae"] = np.nan
    else:
        split_df["nmae"] = split_df["abs_error"] / mean_next_time

    split_df = split_df.reset_index(drop=True)
    split_df["step"] = np.arange(1, len(split_df) + 1)
    window = max(1, min(int(rolling_window), len(split_df)))
    split_df["nmae_line"] = split_df["nmae"].rolling(window=window, min_periods=1).mean()
    return split_df


def plot_event_nmae(args: argparse.Namespace) -> None:
    prediction_path, output_path, csv_output_path = resolve_event_paths(args)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    df = pd.read_csv(prediction_path, low_memory=False)
    train_events = _prepare_event_nmae(df, "train", args.rolling_window)
    test_events = _prepare_event_nmae(df, "test", args.rolling_window)
    if train_events.empty and test_events.empty:
        raise ValueError("No valid train/test rows found for event-nmae mode")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    if not train_events.empty:
        plt.plot(train_events["step"], train_events["nmae_line"], label="Train NMAE", linewidth=2.0)
    if not test_events.empty:
        plt.plot(test_events["step"], test_events["nmae_line"], label="Validation NMAE", linewidth=2.0)

    plt.title(args.title or "MLP Normalized MAE by Event: Train vs Validation")
    plt.xlabel("Event Index in Split")
    plt.ylabel("Normalized MAE")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    rows = []
    if not train_events.empty:
        rows.append(
            {
                "split": "train",
                "nmae": float(train_events["nmae"].mean()),
                "mean_next_time": float(train_events["actual_next_time"].mean()),
                "rows": int(len(train_events)),
            }
        )
    if not test_events.empty:
        rows.append(
            {
                "split": "validation",
                "nmae": float(test_events["nmae"].mean()),
                "mean_next_time": float(test_events["actual_next_time"].mean()),
                "rows": int(len(test_events)),
            }
        )
    pd.DataFrame(rows).to_csv(csv_output_path, index=False)

    print(f"Saved plot: {output_path}")
    print(f"Saved CSV: {csv_output_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "epoch-loss":
        plot_epoch_loss(args)
    else:
        plot_event_nmae(args)


if __name__ == "__main__":
    main()
