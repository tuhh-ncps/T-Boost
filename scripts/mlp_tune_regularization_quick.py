#!/usr/bin/env python3
"""Quick regularization tuning sweep (reduced scope for demonstration).

Tests key combinations of dropout and weight_decay on BPI12.
"""

import subprocess
import time
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import sys

DATASET = "BPI12.csv"
BASE_RESULT_DIR = Path("results/mlp_tune_regularization_quick")
MAX_ITER = 50  # Reduced for faster testing

# Test fewer but representative configurations
DROPOUT_RATES = [0.0, 0.15, 0.3]
WEIGHT_DECAY_RATES = [0.0, 1e-4, 1e-3]

def run_configuration(dropout: float, weight_decay: float, config_id: str) -> Tuple[dict, bool]:
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
    
    print(f"dropout={dropout:.2f}, weight_decay={weight_decay:.0e}  ", end="", flush=True)
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"FAILED")
            return {"dropout": dropout, "weight_decay": f"{weight_decay:.0e}", "status": "FAILED", "r2_score": None}, False
        
        metrics_csv = result_dir / "mlp_metrics_all.csv"
        if not metrics_csv.exists():
            print(f"NO_METRICS")
            return {"dropout": dropout, "weight_decay": f"{weight_decay:.0e}", "status": "NO_METRICS", "r2_score": None}, False
        
        df_metrics = pd.read_csv(metrics_csv)
        if len(df_metrics) == 0:
            print(f"EMPTY_CSV")
            return {"dropout": dropout, "weight_decay": f"{weight_decay:.0e}", "status": "EMPTY_CSV", "r2_score": None}, False
        
        row = df_metrics.iloc[0]
        result_metrics = {
            "dropout": dropout,
            "weight_decay": f"{weight_decay:.0e}",
            "status": "SUCCESS",
            "r2_score": float(row["r2"]),
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "train_epochs": int(row["train_epochs"]),
            "stopped_early": bool(row["stopped_early"]),
            "elapsed_sec": f"{elapsed:.0f}s",
        }
        
        print(f"✓ R2={result_metrics['r2_score']:.4f}, MAE={result_metrics['mae']:.0f}, EP={result_metrics['train_epochs']}")
        return result_metrics, True
        
    except Exception as e:
        print(f"ERROR: {str(e)[:30]}")
        return {"dropout": dropout, "weight_decay": f"{weight_decay:.0e}", "status": "ERROR", "error": str(e)[:50]}, False


def main():
    print(f"\nQuick MLP Regularization Tuning")
    print(f"Dataset: {DATASET}, Max Epochs: {MAX_ITER}")
    print(f"Configs: {len(DROPOUT_RATES) * len(WEIGHT_DECAY_RATES)}\n")
    
    BASE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    
    for dropout in DROPOUT_RATES:
        for weight_decay in WEIGHT_DECAY_RATES:
            config_id = f"d{dropout:.2f}_wd{weight_decay:.0e}".replace("-0", "-")
            metrics, success = run_configuration(dropout, weight_decay, config_id)
            results.append(metrics)
    
    df_results = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}\n")
    
    successful = df_results[df_results["status"] == "SUCCESS"]
    if not successful.empty:
        successful = successful.sort_values("r2_score", ascending=False)
        print(successful[["dropout", "weight_decay", "r2_score", "mae", "train_epochs"]].to_string(index=False))
        
        summary_csv = BASE_RESULT_DIR / "results.csv"
        df_results.to_csv(summary_csv, index=False)
        print(f"\nResults saved to: {summary_csv}")
        
        best = successful.iloc[0]
        print(f"\nBEST: dropout={best['dropout']:.2f}, weight_decay={best['weight_decay']}")
        print(f"      R2={best['r2_score']:.4f}, MAE={best['mae']:.0f}")


if __name__ == "__main__":
    main()
