#!/usr/bin/env python3
"""
Evaluate a decision_tree_time_regime model by plotting predictive density
curves over the true log1p(next_time) histogram.

Histogram: true log1p(next_time) distribution.
For each regime: curve = mean predicted soft probability for samples in each
log-space bin (i.e., average soft probabilities mapped onto the time axis).

Usage:
  python scripts/regime_evaluation.py --model results/decision_tree_time_regime/models/BPI12_decision_tree_time_regime.pkl \
        --train-file data_csv/BPI12_train.csv --test-file data_csv/BPI12_test.csv --output results/decision_tree_time_regime/regime_eval_BPI12.pdf
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import importlib.util
from typing import List

import matplotlib.pyplot as plt
 
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot regime predictive curves over true log1p(next_time)")
    parser.add_argument("--model", type=Path, required=True, help="Path to model artifact pickle")
    parser.add_argument("--train-file", type=Path, required=True, help="Train CSV file (needed to compute feature bins)")
    parser.add_argument("--test-file", type=Path, required=True, help="Test CSV file used for evaluation")
    parser.add_argument("--output", type=Path, default=Path("results/decision_tree_time_regime/regime_eval.pdf"), help="Output figure path (.png/.pdf)")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins along log-time axis (default: 100)")
    parser.add_argument("--figsize", type=str, default="10x6", help="Figure size WxH (default: 10x6)")
    return parser.parse_args()


def parse_figsize(s: str) -> tuple[float, float]:
    try:
        w, h = s.split("x")
        return float(w), float(h)
    except Exception:
        return 10.0, 6.0


def load_module_from_scripts(name: str, fname: Path):
    spec = importlib.util.spec_from_file_location(name, str(fname))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    args = parse_args()
    figsize = parse_figsize(args.figsize)

    # Load model artifact
    with args.model.open("rb") as fh:
        artifact = pickle.load(fh)

    model = artifact.get("model")
    preceding_k = int(artifact.get("preceding_k", 3))
    numeric_medians = artifact.get("numeric_medians", None)
    categorical_levels = artifact.get("categorical_levels", None)

    # Import helper functions from decision_tree_time_regime script so we
    # reuse the same feature preparation code. Try archive version first.
    script_path_archive = Path(__file__).parent / "decision_tree_time_regime_archive.py"
    script_path = Path(__file__).parent / "decision_tree_time_regime.py"
    
    try:
        dt = load_module_from_scripts("decision_tree_time_regime", script_path_archive)
        print("Using archive script for feature preparation")
    except Exception:
        dt = load_module_from_scripts("decision_tree_time_regime", script_path)
        print("Using standard script for feature preparation")

    # Load train and test data, coerce types
    train_df_raw = pd.read_csv(args.train_file, low_memory=False)
    test_df_raw = pd.read_csv(args.test_file, low_memory=False)
    train_df = dt.coerce_types(train_df_raw, preceding_k=preceding_k)
    test_df = dt.coerce_types(test_df_raw, preceding_k=preceding_k)

    # Identify time column
    time_col = "next_time" if "next_time" in train_df.columns else "next_time_seconds"

    # Match training feature engineering order.
    if hasattr(dt, "add_transition_stats_from_train"):
        train_df = dt.add_transition_stats_from_train(train_df, train_df, time_col=time_col)
        test_df = dt.add_transition_stats_from_train(train_df, test_df, time_col=time_col)
    else:
        train_targets = pd.to_numeric(train_df[time_col], errors="coerce")
        train_transitions = pd.DataFrame({"transition": train_df["transition"], "next_time_target": train_targets})
        stats = (
            train_transitions.dropna(subset=["transition", "next_time_target"])
            .groupby("transition", dropna=False)["next_time_target"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "transition_time_mean", "std": "transition_time_std"})
        )
        global_mean = float(train_targets.dropna().mean()) if train_targets.notna().any() else 0.0
        global_std = float(train_targets.dropna().std()) if train_targets.notna().any() else 0.0
        train_df = train_df.join(stats, on="transition")
        test_df = test_df.join(stats, on="transition")
        for frame in (train_df, test_df):
            frame["transition_time_mean"] = pd.to_numeric(frame["transition_time_mean"], errors="coerce").fillna(global_mean)
            frame["transition_time_std"] = pd.to_numeric(frame["transition_time_std"], errors="coerce").fillna(global_std)


    # Apply feature engineering: train derives edges, test applies them
    # 1. Add dt1 feature (train derives edges, test applies them)
    train_df, dt1_edges = dt.add_transition_dt1_feature(train_df, train_df, q=10)
    test_df, _ = dt.add_transition_dt1_feature(train_df, test_df, q=10)

    # 2. Add long-term ratio feature
    train_df, ltr_edges = dt.add_transition_long_term_ratio_feature(train_df, train_df, q=10)
    test_df, _ = dt.add_transition_long_term_ratio_feature(train_df, test_df, q=10)

    # Get log_time for histogram
    numeric_time = pd.to_numeric(test_df[time_col], errors="coerce").clip(lower=0.0)
    log_time = np.log1p(numeric_time)

    # Select feature columns (same as training does before prepare_lgbm_features)
    feature_cols = dt.get_model_feature_columns(preceding_k=preceding_k)
    test_df_features = test_df[feature_cols].copy()

    # Prepare model features for test data
    X_prepared, _, _ = dt.prepare_lgbm_features(test_df_features, preceding_k=preceding_k, numeric_medians=numeric_medians, categorical_levels=categorical_levels)

    # Map predicted probabilities to a full TIME_REGIME_CLASSES-aware frame
    probs_df = dt.map_time_regime_probabilities(model, X_prepared)
    # map_time_regime_probabilities returns percentages (0-100); convert to 0-1
    probs = probs_df.to_numpy(dtype=float) / 100.0
    confidence = probs.max(axis=1)
    regime_names = [c.replace("predicted_time_regime_", "").replace("_pct", "") for c in probs_df.columns]

    # Set up bins over observed log_time
    valid_mask = np.isfinite(log_time)
    if not np.any(valid_mask):
        raise ValueError("No valid next_time values in test file")
    log_vals = log_time[valid_mask].to_numpy(dtype=float)
    pidx = np.flatnonzero(valid_mask)

    quantiles = np.linspace(0.0, 1.0, args.bins + 1)
    edges = np.unique(np.quantile(log_vals, quantiles))
    if edges.size < 2:
        raise ValueError("Not enough unique quantile edges for the requested number of bins")
    hist_vals, edges = np.histogram(log_vals, bins=edges, density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = np.diff(edges)

    # For each regime, compute mean predicted probability for samples in each bin
    regime_curves = np.zeros((len(regime_names), len(bin_centers)), dtype=float)
    for i in range(len(bin_centers)):
        left = edges[i]
        right = edges[i + 1]
        # include right edge on last bin
        if i == len(bin_centers) - 1:
            sel = (log_vals >= left) & (log_vals <= right)
        else:
            sel = (log_vals >= left) & (log_vals < right)
        if not np.any(sel):
            regime_curves[:, i] = np.nan
            continue
        indices = pidx[sel]
        regime_curves[:, i] = np.nanmean(probs[indices, :], axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()

    # Probability curves on the left axis
    cmap = plt.cm.get_cmap("tab10")
    for idx, regime in enumerate(regime_names):
        curve = regime_curves[idx]
        ax.plot(bin_centers, curve, label=f"{regime} probability", color=cmap(idx % 10), linewidth=2.0)

    # Density histogram on the right axis
    ax2.bar(
        bin_centers,
        hist_vals,
        width=bin_widths,
        alpha=0.8,
        color="gray",
        edgecolor="white",
        linewidth=0.4,
        align="center",
        label="True log1p(next_time) density",
    )

    ax.set_xlabel("log1p(next_time)")
    ax.set_ylabel("Probability")
    ax2.set_ylabel("Density")
    ax.grid(alpha=0.3)

    # no entropy colorbar (entropy removed)

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax2.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", framealpha=0.9)
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 1.0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save using extension from output path; matplotlib infers format from suffix.
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved regime evaluation plot: {args.output}")


if __name__ == "__main__":
    main()
