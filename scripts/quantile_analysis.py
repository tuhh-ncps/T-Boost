#!/usr/bin/env python3
"""Plot a log1p(next_time) histogram using optimal bin selection via L2-distance.

Implements the histogram binning method from:
  Heinrich, L., et al. (2020) "Optimal Histogram Binning"
  arXiv:2005.09018

The script uses Monte Carlo simulation to select the number of bins that
minimizes L2 distance from a flat (uniform) histogram, indicating optimal
data-driven bin count selection.

Examples:
  python scripts/quantile_analysis.py --dataset data_csv/bpi_12_w.csv
  python scripts/quantile_analysis.py --data-dir data_csv --dataset BPI12.csv --output results/bpi12_optimal_hist.pdf
  python scripts/quantile_analysis.py --data-dir data_csv --output results/optimal_histograms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("data_csv")
DEFAULT_OUTPUT = Path("results/quantile_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize log1p(next_time) with optimal histogram bin selection."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing CSV files (default: data_csv)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional dataset filename inside --data-dir (e.g. BPI12.csv)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated list of dataset filenames inside --data-dir (e.g. BPI12.csv,bpi_12_w.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file path for one dataset, or a directory when processing multiple datasets",
    )
    parser.add_argument(
        "--candidate-bins",
        type=str,
        default="2-20",
        help="Candidate bin range as 'min-max' (default: 2-20)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for critical distance (default: 0.05)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Elbow threshold: select first K where marginal improvement < epsilon (default: 0.05)",
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="8x4.8",
        help="Figure size in WxH format (default: 8x4.8)",
    )
    return parser.parse_args()


def parse_figsize(spec: str) -> tuple[float, float]:
    try:
        width, height = spec.lower().split("x", maxsplit=1)
        return float(width), float(height)
    except Exception:
        return 8.0, 4.8


def parse_candidate_bins(spec: str) -> list[int]:
    try:
        parts = spec.split("-")
        min_k = int(parts[0].strip())
        max_k = int(parts[1].strip())
        return list(range(min_k, max_k + 1))
    except Exception:
        return list(range(2, 21))


def list_csv_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {data_dir}")

    files = sorted(path for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csv")
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return files


def select_files(data_dir: Path, dataset: str) -> list[Path]:
    files = list_csv_files(data_dir)
    by_name = {path.name: path for path in files}

    if dataset:
        if dataset in by_name:
            return [by_name[dataset]]

        candidate = Path(dataset)
        if candidate.exists():
            return [candidate]

        raise FileNotFoundError(f"Dataset not found: {dataset}")

    return files


def load_log1p_next_time(path: Path) -> np.ndarray:
    df = pd.read_csv(path, low_memory=False)
    if "next_time" not in df.columns:
        raise ValueError(f"Missing next_time column in {path.name}")

    next_time = pd.to_numeric(df["next_time"], errors="coerce").dropna().clip(lower=0.0)
    if next_time.empty:
        raise ValueError(f"No valid next_time values in {path.name}")

    return np.log1p(next_time.to_numpy(dtype=float))


def histogram_l2_distance(x: np.ndarray, k: int) -> float:
    """
    Compute L2 distance of histogram from flat (uniform) histogram.
    
    Args:
        x: normalized values in [0, 1]
        k: number of bins
    
    Returns:
        L2 distance (lower = flatter = better fit)
    """
    counts, _ = np.histogram(x, bins=k, range=(0.0, 1.0))
    n = len(x)
    
    # normalized histogram height
    # flat histogram has height 1 in every bin
    h = counts / (n / k)
    
    return float(np.mean((h - 1.0) ** 2))


def monte_carlo_critical_distance(
    n: int,
    k: int,
    alpha: float = 0.05,
    n_sim: int = 10000,
    random_state: int = 42,
) -> float:
    """
    Compute critical distance threshold via Monte Carlo simulation.
    
    Args:
        n: sample size
        k: number of bins
        alpha: significance level
        n_sim: number of simulations
        random_state: random seed
    
    Returns:
        Critical distance c(alpha, k, n)
    """
    rng = np.random.default_rng(random_state)
    distances = []

    for _ in range(n_sim):
        x = rng.uniform(0.0, 1.0, size=n)
        d = histogram_l2_distance(x, k)
        distances.append(d)

    distances = np.array(distances)
    threshold = float(np.quantile(distances, 1.0 - alpha))
    return threshold


def select_k_elbow(
    values: np.ndarray,
    candidate_ks: list[int],
    epsilon: float = 0.05,
) -> tuple[int, pd.DataFrame]:
    """
    Select optimal bin count via elbow method on reconstruction loss.
    
    Computes reconstruction loss for each k and selects the bin count at the elbow,
    defined as the first K where marginal improvement G(K) < epsilon.
    Returns K-1 (the previous bin count) as the elbow point.
    
    Args:
        values: array of data values
        candidate_ks: list of candidate bin counts
        epsilon: elbow threshold for marginal improvement (default: 0.05)
    
    Returns:
        (selected_k, summary_dataframe)
    """
    rows = []
    losses = []

    for k in candidate_ks:
        loss = compute_reconstruction_loss(values, k)
        losses.append(loss)

    losses = np.array(losses)
    improvements = compute_marginal_improvement(losses)

    for k, loss, gain in zip(candidate_ks, losses, improvements):
        rows.append({
            "K": k,
            "reconstruction_loss": loss,
            "marginal_improvement": gain,
        })

    df = pd.DataFrame(rows)

    # Elbow detection: find first K (with valid loss) where G(K) < epsilon
    # Fallback: use last k where loss is valid
    selected_k = None
    
    # First pass: find k where G(K) < epsilon and loss is valid
    for i, row in df.iterrows():
        loss = row["reconstruction_loss"]
        gain = row["marginal_improvement"]
        
        if pd.isna(loss):
            continue
            
        if pd.notna(gain) and gain < epsilon:
            selected_k = int(candidate_ks[i - 1]) if i > 0 else int(candidate_ks[0])
            break
    
    # Fallback: if no k satisfies condition, use last k where loss is valid
    if selected_k is None:
        for i in range(len(df) - 1, -1, -1):
            if pd.notna(df.iloc[i]["reconstruction_loss"]):
                selected_k = int(df.iloc[i]["K"])
                break
    
    # Final fallback: use first candidate (shouldn't happen)
    if selected_k is None:
        selected_k = int(candidate_ks[0])

    return selected_k, df


def normalize_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """Normalize values to [0, 1] range."""
    val_min = float(np.min(values))
    val_max = float(np.max(values))
    if np.isclose(val_max - val_min, 0.0):
        return np.zeros_like(values)
    return (values - val_min) / (val_max - val_min)


def compute_reconstruction_loss(values: np.ndarray, k: int) -> float:
    """
    Compute lossy reconstruction error after quantization into k bins.

    Each bin reconstructs values using the bin median.
    """

    bins = pd.qcut(values, q=k, duplicates="drop")

    actual_k = len(bins.categories)
    if actual_k < k:
        return np.nan

    codes = bins.codes

    reconstructed = np.zeros_like(values)

    for bucket in range(len(bins.categories)):
        idx = codes == bucket

        # representative value
        med = np.median(values[idx])

        reconstructed[idx] = med

    loss = np.mean((values - reconstructed) ** 2)

    return float(loss)


def compute_marginal_improvement(losses: np.ndarray) -> np.ndarray:
    """
    Compute marginal improvement G(K) = (L(K-1) - L(K)) / L(K-1).
    
    Represents percentage improvement when adding another bin.
    
    Args:
        losses: array of L2 losses for K = 2, 3, 4, ...
    
    Returns:
        Array of marginal improvements (same length as losses, first is NaN)
    """
    improvements = np.full_like(losses, np.nan, dtype=float)
    
    for i in range(1, len(losses)):
        l_prev = losses[i - 1]
        l_curr = losses[i]
        
        if l_prev > 0 and not np.isclose(l_prev - l_curr, 0.0):
            improvements[i] = (l_prev - l_curr) / l_prev
        
    return improvements


def plot_histogram(
    values: np.ndarray,
    dataset_name: str,
    output_path: Path,
    figsize: tuple[float, float],
    selected_k: int,
    selection_df: pd.DataFrame,
) -> None:
    """Plot histogram with optimal bin count."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(values, bins=selected_k, density=True, color="#4e79a7", alpha=0.8, edgecolor="white", linewidth=0.6)

    title = f"log1p(next_time) histogram - {dataset_name}"
    ax.set_title(title)
    ax.set_xlabel("log1p(next_time)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)

    # Summary box
    summary_lines = [
        f"Optimal bins (elbow method): {selected_k}",
        f"Sample size: {len(values)}",
        f"Range: {float(np.max(values) - np.min(values)):.6g}",
        f"Min: {float(np.min(values)):.6g}",
        f"Max: {float(np.max(values)):.6g}",
    ]
    ax.text(
        0.98,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#444444", alpha=0.92),
    )

    # Bin selection details box
    best_row = selection_df.loc[selection_df["K"] == selected_k].iloc[0]
    details_lines = [
        f"reconstruction_loss: {best_row['reconstruction_loss']:.6g}",
        f"marginal_improvement: {best_row['marginal_improvement']:.6g}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(details_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#666666", alpha=0.85),
    )

    ax.axvline(float(np.median(values)), color="#e15759", linestyle="--", linewidth=1.5, alpha=0.85, label="Median")
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_elbow_analysis(
    candidate_ks: list[int],
    losses: np.ndarray,
    marginal_improvements: np.ndarray,
    selected_k: int,
    output_path: Path,
    figsize: tuple[float, float],
    epsilon: float = 0.05,
) -> None:
    """Plot reconstruction loss curves for one or many datasets.

    This function draws a single axes showing L(K) for each dataset provided.
    """

    # When a single dataset is passed, `losses` will be a 1D array.
    # When used for multi-dataset plotting, `losses` may be a 2D array (ndarray of shape [n_datasets, n_k]).
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize losses input into a mapping of label -> array
    if isinstance(losses, dict):
        all_losses = losses
    elif isinstance(losses, np.ndarray):
        if losses.ndim == 1:
            all_losses = {"dataset": losses}
        elif losses.ndim == 2:
            all_losses = {f"ds_{i}": losses[i] for i in range(losses.shape[0])}
        else:
            # unexpected shape
            all_losses = {"dataset": losses.ravel()}
    elif isinstance(losses, list):
        # list of arrays: label them by index
        all_losses = {f"ds_{i}": np.asarray(row) for i, row in enumerate(losses)}
    else:
        # fallback: attempt to iterate
        try:
            all_losses = {f"ds_{i}": np.asarray(row) for i, row in enumerate(losses)}
        except Exception:
            all_losses = {"dataset": np.asarray(losses)}

    # Plot each dataset curve
    for label, loss_arr in all_losses.items():
        if loss_arr is None:
            continue
        ax.plot(candidate_ks, loss_arr, marker="o", linewidth=2.0, markersize=4, label=label)

    ax.set_xlabel("Number of bins (K)")
    ax.set_ylabel("Reconstruction Loss L(K)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=12)

    output_file = output_path.parent / f"{output_path.stem}_reconstruction_curves.pdf"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resolve_output_path(output: Path, dataset: Path, multi_file: bool) -> Path:
    if not multi_file:
        if output.suffix:
            return output
        return output / f"{dataset.stem}_optimal_hist.pdf"

    return output / f"{dataset.stem}_optimal_hist.pdf"


def main() -> None:
    args = parse_args()
    figsize = parse_figsize(args.figsize)
    candidate_ks = parse_candidate_bins(args.candidate_bins)
    # Support selecting multiple datasets via --datasets (comma-separated)
    if args.datasets:
        requested = [s.strip() for s in args.datasets.split(",") if s.strip()]
        all_files = list_csv_files(args.data_dir)
        by_name = {p.name: p for p in all_files}
        files = []
        for name in requested:
            if name in by_name:
                files.append(by_name[name])
            else:
                candidate = Path(name)
                if candidate.exists():
                    files.append(candidate)
                else:
                    raise FileNotFoundError(f"Dataset not found: {name}")
    else:
        files = select_files(args.data_dir, args.dataset)
    multi_file = len(files) > 1

    if multi_file and args.output.suffix:
        raise ValueError("When processing multiple datasets, --output should be a directory, not a file path")

    for path in files:
        values = load_log1p_next_time(path)
        
        # Select optimal k using elbow method on reconstruction loss
        selected_k, selection_df = select_k_elbow(
            values=values,
            candidate_ks=candidate_ks,
            epsilon=args.epsilon,
        )

        # store per-dataset results for multi-curve plotting
        dataset_name = path.stem
        if "all_losses" not in locals():
            all_losses = {}
            all_selection_dfs = {}

        all_losses[dataset_name] = selection_df["reconstruction_loss"].values
        all_selection_dfs[dataset_name] = selection_df

        output_path = resolve_output_path(args.output, path, multi_file=multi_file)
        plot_histogram(values, path.name, output_path, figsize, selected_k, selection_df)

        print(f"\nDataset: {path.name}")
        print(f"Saved optimal histogram: {output_path}")
        print(f"Selected bins (first K where G(K) < ε={args.epsilon}): {selected_k}")
        print("\nReconstruction loss and marginal improvement analysis:")
        print(selection_df.to_string(index=False))


    # After processing all datasets, if multiple datasets were provided, create the combined reconstruction-loss plot
    if multi_file:
        # Use the first file's output_path as base for naming
        # place combined reconstruction plot in results/ with a clear name
        combined_output = Path("results") / "quantile_analysis_reconstruction_multi.pdf"

        # candidate_ks is same for all datasets
        plot_elbow_analysis(
            candidate_ks,
            all_losses,
            None,
            selected_k=0,
            output_path=Path(combined_output),
            figsize=figsize,
        )

    if __name__ == "__main__":
        main()