#!/usr/bin/env python3
"""Architecture tuning sweep: Test batch norm + layer width combinations."""

import subprocess
import time
from pathlib import Path
from typing import Tuple
import pandas as pd

DATASET = "BPI12.csv"
BASE_RESULT_DIR = Path("results/mlp_tune_architecture")
MAX_ITER = 50

# Test configurations: (batch_norm, hidden_layers_str)
CONFIGS = [
    (False, "256,128", "baseline"),
    (True, "256,128", "bn_default"),
    (True, "512,256,128", "bn_wide"),
    (True, "512,256,128,64", "bn_deeper"),
    (False, "512,256,128", "wide_nobn"),
    (False, "1024,512,256", "extra_wide_nobn"),
    (True, "1024,512,256", "bn_extra_wide"),
]

def run_config(use_bn: bool, hidden_layers: str, config_name: str) -> Tuple[dict, bool]:
    """Run a single architecture configuration."""
    
    result_dir = BASE_RESULT_DIR / config_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "./.venv/bin/python",
        "scripts/mlp.py",
        "--dataset", DATASET,
        "--max-iter", str(MAX_ITER),
        "--hidden-layers", hidden_layers,
        "--result-dir", str(result_dir),
    ]
    
    if use_bn:
        cmd.append("--use-batch-norm")
    
    print(f"{config_name:30s}  ", end="", flush=True)
    
    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"FAILED")
            return {"config": config_name, "status": "FAILED", "r2_score": None}, False
        
        metrics_csv = result_dir / "mlp_metrics_all.csv"
        if not metrics_csv.exists():
            print(f"NO_METRICS")
            return {"config": config_name, "status": "NO_METRICS", "r2_score": None}, False
        
        df = pd.read_csv(metrics_csv)
        if len(df) == 0:
            print(f"EMPTY")
            return {"config": config_name, "status": "EMPTY", "r2_score": None}, False
        
        row = df.iloc[0]
        result_metrics = {
            "config": config_name,
            "batch_norm": use_bn,
            "layers": hidden_layers,
            "status": "SUCCESS",
            "r2_score": float(row["r2"]),
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "epochs": int(row["train_epochs"]),
            "time_sec": f"{elapsed:.0f}",
        }
        
        print(f"✓ R2={result_metrics['r2_score']:.4f}, MAE={result_metrics['mae']:.0f}, E={result_metrics['epochs']}")
        return result_metrics, True
        
    except Exception as e:
        print(f"ERROR: {str(e)[:30]}")
        return {"config": config_name, "status": "ERROR", "r2_score": None}, False


def main():
    print(f"\nArchitecture Tuning Sweep (50 epochs per config)")
    print(f"Dataset: {DATASET}")
    print(f"Configs: {len(CONFIGS)}")
    print(f"Baseline R2 (no regularization, 100 epochs): 0.1771\n")
    
    BASE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    successful = 0
    
    for use_bn, hidden_layers, config_name in CONFIGS:
        metrics, success = run_config(use_bn, hidden_layers, config_name)
        results.append(metrics)
        if success:
            successful += 1
    
    print(f"\n{'=' * 100}")
    print("RESULTS (sorted by R2):")
    print(f"{'=' * 100}\n")
    
    df_results = pd.DataFrame(results)
    successful_df = df_results[df_results["status"] == "SUCCESS"].sort_values("r2_score", ascending=False)
    
    if not successful_df.empty:
        print(successful_df[["config", "r2_score", "mae", "epochs", "layers"]].to_string(index=False))
        
        best = successful_df.iloc[0]
        print(f"\n{'=' * 100}")
        print(f"BEST: {best['config']}")
        print(f"  R2: {best['r2_score']:.4f}")
        print(f"  MAE: {best['mae']:.0f}")
        print(f"  Batch Norm: {best['batch_norm']}")
        print(f"  Layers: {best['layers']}")
        print(f"  Epochs Trained: {best['epochs']}")
        
        if best['r2_score'] > 0.1771:
            improvement = ((best['r2_score'] - 0.1771) / 0.1771) * 100
            print(f"  Improvement over baseline: +{improvement:.1f}% ✨")
        else:
            gap = ((0.1771 - best['r2_score']) / 0.1771) * 100
            print(f"  Gap vs baseline: -{gap:.1f}%")
    
    # Save results
    summary_csv = BASE_RESULT_DIR / "results.csv"
    df_results.to_csv(summary_csv, index=False)
    print(f"\nResults saved to: {summary_csv}")


if __name__ == "__main__":
    main()
