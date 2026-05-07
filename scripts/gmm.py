import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def gaussian_pdf(x, mean, var):
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)


def analyze_single_dataset(csv_path, args, out_dir):
    """Analyze single dataset: BIC/AIC plot + histogram plot"""
    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    if args.target not in df.columns:
        raise ValueError(f"Column '{args.target}' not found. Available columns: {df.columns.tolist()}")

    y_raw = df[args.target].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    y_raw = y_raw[y_raw >= 0]

    zero_mask = y_raw == 0
    y_zero = y_raw[zero_mask]
    y_pos = y_raw[~zero_mask]

    print(f"Dataset: {csv_path.name}")
    print(f"Total valid rows: {len(y_raw)}")
    print(f"Zero rows: {len(y_zero)} ({len(y_zero) / len(y_raw):.2%})")
    print(f"Positive rows: {len(y_pos)}")

    if args.include_zero:
        y_model_raw = y_raw
    else:
        y_model_raw = y_pos

    y_log = np.log1p(y_model_raw).values.reshape(-1, 1)

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_log)

    results = []

    for k in range(args.k_min, args.k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42,
            n_init=20,
            reg_covar=1e-6,
        )
        gmm.fit(y_scaled)

        bic = gmm.bic(y_scaled)
        aic = gmm.aic(y_scaled)
        log_likelihood = gmm.score(y_scaled)

        results.append({
            "k": k,
            "bic": bic,
            "aic": aic,
            "avg_log_likelihood": log_likelihood,
            "min_cluster_weight": gmm.weights_.min(),
            "max_cluster_weight": gmm.weights_.max(),
        })

    results_df = pd.DataFrame(results)

    best_k = int(results_df.loc[results_df["bic"].idxmin(), "k"])
    print(f"\nBest K by BIC: {best_k}")
    print("\nModel selection results:")
    print(results_df)

    # Fit final model
    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        random_state=42,
        n_init=50,
        reg_covar=1e-6,
    )
    gmm.fit(y_scaled)

    probs = gmm.predict_proba(y_scaled)
    labels = gmm.predict(y_scaled)

    # Sort labels by mean time, so regime 0 = shortest, regime N = longest
    means_scaled = gmm.means_.reshape(-1, 1)
    means_log = scaler.inverse_transform(means_scaled).reshape(-1)

    sorted_components = np.argsort(means_log)
    remap = {old: new for new, old in enumerate(sorted_components)}
    labels_sorted = np.array([remap[label] for label in labels])

    # Save assignment result
    assigned = pd.DataFrame({
        "next_time": y_model_raw.values,
        "log1p_next_time": y_log.reshape(-1),
        "gmm_label": labels_sorted,
        "max_probability": probs.max(axis=1),
    })

    for old_component in range(best_k):
        new_component = remap[old_component]
        assigned[f"prob_regime_{new_component}"] = probs[:, old_component]

    # Component summary
    summary = []
    for old_component in sorted_components:
        new_component = remap[old_component]

        mean_log = means_log[old_component]
        var_scaled = gmm.covariances_[old_component].reshape(1, 1)
        std_log = np.sqrt(scaler.var_[0] * var_scaled[0, 0])

        cluster_values = assigned[assigned["gmm_label"] == new_component]["next_time"]

        summary.append({
            "regime": new_component,
            "weight": gmm.weights_[old_component],
            "mean_log1p": mean_log,
            "approx_mean_raw": np.expm1(mean_log),
            "std_log1p": std_log,
            "count": len(cluster_values),
            "count_pct": len(cluster_values) / len(assigned),
            "min_raw": cluster_values.min(),
            "median_raw": cluster_values.median(),
            "max_raw": cluster_values.max(),
        })

    summary_df = pd.DataFrame(summary)

    print("\nRegime summary:")
    print(summary_df)

    # Plot 1: BIC / AIC
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["k"], results_df["bic"], marker="o", label="BIC")
    plt.plot(results_df["k"], results_df["aic"], marker="o", label="AIC")
    plt.xlabel("Number of components K")
    plt.ylabel("Score")
    plt.title(f"GMM model selection - {csv_path.stem}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    bic_path = out_dir / f"{csv_path.stem}_01_bic_aic.png"
    plt.savefig(bic_path, dpi=200)
    plt.close()
    print(f"Saved: {bic_path}")

    # Plot 2: histogram + GMM components
    x_log = y_log.reshape(-1)
    x_grid_log = np.linspace(x_log.min(), x_log.max(), 1000).reshape(-1, 1)
    x_grid_scaled = scaler.transform(x_grid_log)

    total_density_scaled = np.exp(gmm.score_samples(x_grid_scaled))

    # Convert scaled density to log-space density
    scale_factor = 1 / scaler.scale_[0]
    total_density_log = total_density_scaled * scale_factor

    plt.figure(figsize=(10, 6))
    plt.hist(x_log, bins=80, density=True, alpha=0.45, label="Histogram")

    for old_component in sorted_components:
        new_component = remap[old_component]
        mean = gmm.means_[old_component, 0]
        var = gmm.covariances_[old_component][0, 0]
        weight = gmm.weights_[old_component]

        component_scaled = weight * gaussian_pdf(x_grid_scaled.reshape(-1), mean, var)
        component_log = component_scaled * scale_factor

        plt.plot(
            x_grid_log.reshape(-1),
            component_log,
            linewidth=2,
            label=f"Regime {new_component}"
        )

    plt.plot(x_grid_log.reshape(-1), total_density_log, linewidth=2.5, linestyle="--", label="GMM total")
    plt.xlabel("log1p(next_time)")
    plt.ylabel("Density")
    plt.title(f"GMM components, K={best_k} - {csv_path.stem}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = out_dir / f"{csv_path.stem}_02_histogram_gmm_components.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved: {hist_path}")

    return results_df


def compare_multiple_datasets(csv_paths, args, out_dir):
    """Compare BIC/AIC curves across multiple datasets in one plot"""
    comparison_data = []
    
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        
        df = pd.read_csv(csv_path)
        if args.target not in df.columns:
            raise ValueError(f"Column '{args.target}' not found in {csv_path}")
        
        y_raw = df[args.target].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        y_raw = y_raw[y_raw >= 0]
        
        if args.include_zero:
            y_model_raw = y_raw
        else:
            y_model_raw = y_raw[y_raw > 0]
        
        y_log = np.log1p(y_model_raw).values.reshape(-1, 1)
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y_log)
        
        results = []
        for k in range(args.k_min, args.k_max + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42,
                n_init=20,
                reg_covar=1e-6,
            )
            gmm.fit(y_scaled)
            bic = gmm.bic(y_scaled)
            aic = gmm.aic(y_scaled)
            results.append({"k": k, "bic": bic, "aic": aic})
        
        results_df = pd.DataFrame(results)
        results_df["dataset"] = csv_path.stem
        comparison_data.append(results_df)
        
        print(f"{csv_path.stem}: Best K (BIC)={int(results_df.loc[results_df['bic'].idxmin(), 'k'])}")
    
    # Plot BIC comparison
    plt.figure(figsize=(12, 6))
    for results_df in comparison_data:
        dataset_name = results_df["dataset"].iloc[0]
        plt.plot(results_df["k"], results_df["bic"], marker="o", label=dataset_name)
    
    plt.xlabel("Number of components K")
    plt.ylabel("BIC Score")
    plt.title("BIC Comparison Across Datasets")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    bic_comparison_path = out_dir / "comparison_bic.png"
    plt.savefig(bic_comparison_path, dpi=200)
    plt.close()
    print(f"Saved: {bic_comparison_path}")
    
    # Plot AIC comparison
    plt.figure(figsize=(12, 6))
    for results_df in comparison_data:
        dataset_name = results_df["dataset"].iloc[0]
        plt.plot(results_df["k"], results_df["aic"], marker="o", label=dataset_name)
    
    plt.xlabel("Number of components K")
    plt.ylabel("AIC Score")
    plt.title("AIC Comparison Across Datasets")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    aic_comparison_path = out_dir / "comparison_aic.png"
    plt.savefig(aic_comparison_path, dpi=200)
    plt.close()
    print(f"Saved: {aic_comparison_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=False, help="Single CSV for analysis")
    parser.add_argument("--compare", nargs="+", required=False, help="Multiple CSVs to compare BIC/AIC")
    parser.add_argument("--target", default="next_time", help="Target column name containing next-time values")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum K for model selection")
    parser.add_argument("--k-max", type=int, default=6, help="Maximum K for model selection")
    parser.add_argument("--include-zero", action="store_true", help="Include zero next_time rows when fitting GMM")
    args = parser.parse_args()

    # Create output directory
    out_dir = Path("results") / "gmm_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        # Comparison mode: plot BIC/AIC for multiple datasets
        print(f"Comparing {len(args.compare)} datasets...")
        compare_multiple_datasets(args.compare, args, out_dir)
    elif args.csv:
        # Single dataset mode
        analyze_single_dataset(args.csv, args, out_dir)
    else:
        raise ValueError("Provide either --csv for single dataset analysis or --compare for multi-dataset comparison")


if __name__ == "__main__":
    main()