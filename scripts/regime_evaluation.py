#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot temporal-scale predictive curves over true log1p(next_time) density"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/decision_tree_time_regime/regime_eval.pdf"),
    )
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--figsize", type=str, default="7x5")
    return parser.parse_args()


def parse_figsize(s: str) -> tuple[float, float]:
    try:
        w, h = s.split("x")
        return float(w), float(h)
    except Exception:
        return 7.0, 5.0


def load_module_from_scripts(name: str, fname: Path):
    spec = importlib.util.spec_from_file_location(name, str(fname))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    args = parse_args()
    figsize = parse_figsize(args.figsize)

    with args.model.open("rb") as fh:
        artifact = pickle.load(fh)

    model = artifact.get("model")
    preceding_k = int(artifact.get("preceding_k", 3))
    numeric_medians = artifact.get("numeric_medians", None)
    categorical_levels = artifact.get("categorical_levels", None)

    script_path_archive = Path(__file__).parent / "decision_tree_time_regime_archive.py"
    script_path = Path(__file__).parent / "decision_tree_time_regime.py"

    try:
        dt = load_module_from_scripts("decision_tree_time_regime", script_path_archive)
        print("Using archive script for feature preparation")
    except Exception:
        dt = load_module_from_scripts("decision_tree_time_regime", script_path)
        print("Using standard script for feature preparation")

    train_df_raw = pd.read_csv(args.train_file, low_memory=False)
    test_df_raw = pd.read_csv(args.test_file, low_memory=False)

    train_df = dt.coerce_types(train_df_raw, preceding_k=preceding_k)
    test_df = dt.coerce_types(test_df_raw, preceding_k=preceding_k)

    time_col = "next_time" if "next_time" in train_df.columns else "next_time_seconds"

    if hasattr(dt, "add_transition_stats_from_train"):
        train_df = dt.add_transition_stats_from_train(train_df, train_df, time_col=time_col)
        test_df = dt.add_transition_stats_from_train(train_df, test_df, time_col=time_col)
    else:
        train_targets = pd.to_numeric(train_df[time_col], errors="coerce")
        train_transitions = pd.DataFrame(
            {"transition": train_df["transition"], "next_time_target": train_targets}
        )

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
            frame["transition_time_mean"] = (
                pd.to_numeric(frame["transition_time_mean"], errors="coerce")
                .fillna(global_mean)
            )
            frame["transition_time_std"] = (
                pd.to_numeric(frame["transition_time_std"], errors="coerce")
                .fillna(global_std)
            )

    train_df, _ = dt.add_transition_dt1_feature(train_df, train_df, q=10)
    test_df, _ = dt.add_transition_dt1_feature(train_df, test_df, q=10)

    train_df, _ = dt.add_transition_long_term_ratio_feature(train_df, train_df, q=10)
    test_df, _ = dt.add_transition_long_term_ratio_feature(train_df, test_df, q=10)

    numeric_time = pd.to_numeric(test_df[time_col], errors="coerce").clip(lower=0.0)
    log_time = np.log1p(numeric_time)

    feature_cols = dt.get_model_feature_columns(preceding_k=preceding_k)
    test_df_features = test_df[feature_cols].copy()

    X_prepared, _, _ = dt.prepare_lgbm_features(
        test_df_features,
        preceding_k=preceding_k,
        numeric_medians=numeric_medians,
        categorical_levels=categorical_levels,
    )

    probs_df = dt.map_time_regime_probabilities(model, X_prepared)
    probs = probs_df.to_numpy(dtype=float) / 100.0

    regime_names = [
        c.replace("predicted_time_regime_", "").replace("_pct", "")
        for c in probs_df.columns
    ]

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

    regime_curves = np.zeros((len(regime_names), len(bin_centers)), dtype=float)

    for i in range(len(bin_centers)):
        left = edges[i]
        right = edges[i + 1]

        if i == len(bin_centers) - 1:
            sel = (log_vals >= left) & (log_vals <= right)
        else:
            sel = (log_vals >= left) & (log_vals < right)

        if not np.any(sel):
            regime_curves[:, i] = np.nan
            continue

        indices = pidx[sel]
        regime_curves[:, i] = np.nanmean(probs[indices, :], axis=0)

    k_scales = len(regime_names)

    if k_scales > 1:
        major_q = np.linspace(0.0, 1.0, k_scales + 1)
        major_edges = np.unique(np.quantile(log_vals, major_q))
        major_internal = major_edges[1:-1]
    else:
        major_internal = np.array([])

    if len(major_internal) > 0:
        area_boundaries = [float(log_vals.min())] + list(major_internal) + [float(log_vals.max())]
    else:
        area_boundaries = [float(log_vals.min()), float(log_vals.max())]

    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    cmap = plt.get_cmap("tab10")

    for area_idx in range(len(area_boundaries) - 1):
        area_left = area_boundaries[area_idx]
        area_right = area_boundaries[area_idx + 1]
        area_center_x = 0.5 * (area_left + area_right)

        ax.axvspan(area_left, area_right, alpha=0.06, zorder=0)

        ax.text(
            area_center_x,
            1.02,
            f"Q{area_idx + 1}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            clip_on=False,
        )

    legend_added: set[str] = set()

    for idx, regime in enumerate(regime_names):
        curve = regime_curves[idx]
        color = cmap(idx % 10)

        for area_idx in range(len(area_boundaries) - 1):
            area_left = area_boundaries[area_idx]
            area_right = area_boundaries[area_idx + 1]

            in_area = (bin_centers >= area_left) & (bin_centers <= area_right)

            if not np.any(in_area):
                continue

            is_correct_scale = idx == area_idx

            linewidth = 2.0 if is_correct_scale else 1.3
            alpha = 1.0 if is_correct_scale else 0.3

            label = None
            if regime not in legend_added:
                label = f"{regime} probability"
                legend_added.add(regime)

            ax.plot(
                bin_centers[in_area],
                curve[in_area],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
                zorder=4 if is_correct_scale else 2,
            )

    ax2.bar(
        bin_centers,
        hist_vals,
        width=bin_widths,
        alpha=0.75,
        color="gray",
        edgecolor="white",
        linewidth=0.4,
        align="center",
        label="True log1p(next_time) density",
        zorder=1,
    )

    for edge in major_internal:
        ax.axvline(
            edge,
            color="#555555",
            linestyle=":",
            linewidth=1.0,
            alpha=0.9,
            zorder=3,
        )

    ax.set_xlabel("log1p(next_time)")
    ax.set_ylabel("Probability")
    ax2.set_ylabel("Density")

    ax.grid(alpha=0.25)

    custom_handles = []

    for idx, regime in enumerate(regime_names):
        color = cmap(idx % 10)

        custom_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2.0,
                alpha=1.0,
                label=f"{regime} probability",
            )
        )

    density_handle = Line2D(
        [0],
        [0],
        color="gray",
        lw=4,
        alpha=0.75,
        label="True log1p(next_time) density",
    )

    custom_handles.append(density_handle)

    ax.legend(
        handles=custom_handles,
        loc="upper left",
        framealpha=0.9,
        fontsize=9,
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved temporal-scale evaluation plot: {args.output}")


if __name__ == "__main__":
    main()