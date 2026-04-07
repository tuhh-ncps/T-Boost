#!/usr/bin/env python3
"""Tune MLP model with regularization (dropout + L2 weight decay).

This script runs a matrix of dropout and weight_decay configurations
and reports the results in a summary table.
"""

import subprocess
import time
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import json
import sys

# Tuning configuration
DATASET = "BPI12.csv"
BASE_RESULT_DIR = Path("results/mlp_tune_regularization")
MAX_ITER = 100

# Test configurations
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3]
WEIGHT_DECAY_RATES = [0.0, 1e-5, 1e-4, 5e-4, 1e-3]

def run_configuration(
    dropout: float,
    weight_decay: float,
    config_id: str,
) -> Tuple[dict, bool]:
    """Run a single configuration and return metrics."""
    
    result_dir = BASE_RESULT_DIR / config_id
    result_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "./.venv/bin/python",
        "scripts/mlp.py",
        "--dataset", DATASET,
        "--max-iter", str(MAX_ITER),
        "--dropout", str(dropout),
        "--weight-decay", str(weight_decay),
        "--result-dir", str(result_dir),
    ]
    
    print(f"dropout={dropout:.1f}, weight_decay={weight_decay:.0e}  ", end="", flush=True)
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"FAILED")
            return {
                "dropout": dropout,
                "weight_decay": f"{weight_decay:.0e}",
                "config_id": config_id,
                "status": "FAILED",
                "r2_score": None,
                "mae": None,
                "rmse": None,
                "train_epochs": None,
                "stopped_early": None,
                "elapsed_sec": f"{elapsed:.1f}",
            }, False
        
        # Parse metrics from CSV
        metrics_csv = result_dir / "mlp_metrics_all.csv"
        if not metrics_csv.exists():
            print(f"NO_METRICS")
            return {
                "dropout": dropout,
                "weight_decay": f"{weight_decay:.0e}",
                "config_id": config_id,
                "status": "NO_METRICS",
                "r2_score": None,
                "mae": None,
                "rmse": None,
                "train_epochs": None,
                "stopped_early": None,
                "elapsed_sec": f"{elapsed:.1f}",
            }, False
        
        df_metrics = pd.read_csv(metrics_csv)
        if len(df_metrics) == 0:
            print(f"EMPTY_CSV")
            return {
                "dropout": dropout,
                "weight_decay": f"{weight_decay:.0e}",
                "config_id": config_id,
                "status": "EMPTY_CSV",
                "r2_score": None,
                "mae": None,
                "rmse": None,
                "train_epochs": None,
                "stopped_early": None,
                "elapsed_sec": f"{elapsed:.1f}",
            }, False
        
        row = df_metrics.iloc[0]
        
        # Extract key metrics
        result_metrics = {
            "dropout": dropout,
            "weight_decay": f"{weight_decay:.0e}",
            "config_id": config_id,
            "status": "SUCCESS",
            "r2_score": float(row["r2"]),
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "train_epochs": int(row["train_epochs"]),
            "stopped_early": bool(row["stopped_early"]),
            "elapsed_sec": f"{elapsed:.1f}",
        }
        
        print(f"✓ R2={result_metrics['r2_score']:.4f}, MAE={result_metrics['mae']:.0f}, EP={result_metrics['train_epochs']}, ES={result_metrics['stopped_early']}")
        
        return result_metrics, True
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT")
        return {
            "dropout": dropout,
            "weight_decay": f"{weight_decay:.0e}",
            "config_id": config_id,
            "status": "TIMEOUT",
            "r2_score": None,
            "mae": None,
            "rmse": None,
            "train_epochs": None,
            "stopped_early": None,
            "elapsed_sec": "600+",
        }, False
    except Exception as e:
        print(f"ERROR: {str(e)[:40]}")
        return {
            "dropout": dropout,
            "weight_decay": f"{weight_decay:.0e}",
            "config_id": config_id,
            "status": "EXCEPTION",
            "r2_score": None,
            "mae": None,
            "rmse": None,
            "train_epochs": None,
            "stopped_early": None,
            "elapsed_sec": "N/A",
            "error": str(e)[:100],
        }, False


def main():
    print(f"\nMLP Regularization Tuning Sweep")
    print(f"Dataset: {DATASET}, Max Epochs: {MAX_ITER}")
    print(f"Dropout: {DROPOUT_RATES}, Weight Decay: {WEIGHT_DECAY_RATES}")
    print(f"Total: {len(DROPOUT_RATES) * len(WEIGHT_DECAY_RATES)} configurations\n")
    
    BASE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    successful = 0
    failed = 0
    
    # Run all configurations
    total_configs = len(DROPOUT_RATES) * len(WEIGHT_DECAY_RATES)
    current = 0
    
    print(f"{'Config':^50} | {'Status':^8} | {'R2':>7} | {'MAE':>7} | {'Epochs':>6}")
    print(f"{'-'*50}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")
    
    for dropout in DROPOUT_RATES:
        for weight_decay in WEIGHT_DECAY_RATES:
            current += 1
            config_id = f"d{dropout:.1f}_wd{weight_decay:.0e}".replace("e-0", "e")
            
            metrics, success = run_configuration(dropout, weight_decay, config_id)
            results.append(metrics)
            
            if success:
                successful += 1
            else:
                failed += 1
    
    # Create summary table
    print(f"\n{'='*100}")
    print(f"TUNING SWEEP RESULTS SUMMARY")
    print(f"{'='*100}\n")
    
    df_results = pd.DataFrame(results)
    
    # Sort by R2 (descending) for successful runs
    successful_runs = df_results[df_results["status"] == "SUCCESS"].copy()
    if not successful_runs.empty:
        successful_runs = successful_runs.sort_values("r2_score", ascending=False)
        
        print("Top 10 Configurations (by R2):")
        print(successful_runs[["dropout", "weight_decay", "r2_score", "mae", "rmse", "train_epochs", "stopped_early"]].head(10).to_string(index=False))
        print()
    
    # Save full results
    summary_csv = BASE_RESULT_DIR / "tuning_summary.csv"
    df_results.to_csv(summary_csv, index=False)
    print(f"\nFull results saved to: {summary_csv}")
    
    # Print summary stats
    print(f"\n{'='*100}")
    print(f"SUMMARY: {successful} successful, {failed} failed")
    print(f"{'='*100}\n")
    
    # Find best configuration
    if not successful_runs.empty:
        best_idx = successful_runs["r2_score"].idxmax()
        best_result = successful_runs.loc[best_idx]
        print(f"BEST CONFIGURATION:")
        print(f"  Dropout: {best_result['dropout']:.1f}")
        print(f"  Weight Decay: {best_result['weight_decay']}")
        print(f"  R2 Score: {best_result['r2_score']:.4f}")
        print(f"  MAE: {best_result['mae']:.0f}")
        print(f"  RMSE: {best_result['rmse']:.0f}")
        print(f"  Train Epochs: {best_result['train_epochs']}")
        print(f"  Early Stopped: {best_result['stopped_early']}")
        print(f"\nRecommended command for final training (100 epochs):")
        print(f"  ./.venv/bin/python scripts/mlp.py --dataset {DATASET} --dropout {best_result['dropout']:.1f} --weight-decay {best_result['weight_decay']} --max-iter 100")


if __name__ == "__main__":
    main()
