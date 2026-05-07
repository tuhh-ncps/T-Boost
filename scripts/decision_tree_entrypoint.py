#!/usr/bin/env python3
"""Run the two-stage decision-tree pipeline.

Stage 1: Train/evaluate time_regime classifier using
- <dataset>_train.csv
- <dataset>_test.csv
and generate
- <dataset>_test_with_time_regime.csv

Stage 2: Train/evaluate next_time regressor using
- <dataset>_train.csv
- <dataset>_test_with_time_regime.csv

Input selection:
    Use --data-dir for the split CSV folder. Optionally use --dataset to run
    only one dataset stem.

Example:
    python scripts/decision_tree_entrypoint.py --data-dir data_csv --dataset helpdesk.csv

Usage:
    --data-dir PATH            Folder containing split CSV files (default: data_csv)
    --dataset NAME             Optional single dataset file to run
    --random-state N           Random seed (default: 42)
    --max-rows-per-file N      Optional row cap per file (default: 0)
    --preceding-k N            Number of prefix tokens to encode (default: 6)
    --max-depth-limit N        Upper bound for max_depth tuning (default: 10)
    --use-next-activity        Include next_activity in both stages
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATA_DIR = Path("data_csv")
DEFAULT_TIME_REGIME_RESULT_DIR = Path("results/decision_tree_time_regime")
DEFAULT_REGRESSION_RESULT_DIR = Path("results/decision_tree")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run decision_tree_time_regime.py then decision_tree.py")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder containing split CSV files")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset base name or filename (e.g., helpdesk.csv). If omitted, run all discovered pairs.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--max-rows-per-file", type=int, default=0, help="Optional row cap per file")
    parser.add_argument("--preceding-k", type=int, default=6, help="Number of prefix tokens to encode")
    parser.add_argument("--max-depth-limit", type=int, default=10, help="Upper bound for max_depth tuning")
    parser.add_argument(
        "--use-next-activity",
        action="store_true",
        help="Include next_activity as model input in both stages",
    )
    parser.add_argument(
        "--time-regime-result-dir",
        type=Path,
        default=DEFAULT_TIME_REGIME_RESULT_DIR,
        help="Output directory for time_regime classifier artifacts",
    )
    parser.add_argument(
        "--regression-result-dir",
        type=Path,
        default=DEFAULT_REGRESSION_RESULT_DIR,
        help="Output directory for next_time regressor artifacts",
    )
    return parser.parse_args()


def build_common_cli(args: argparse.Namespace) -> list[str]:
    cli = [
        "--data-dir",
        str(args.data_dir),
        "--random-state",
        str(args.random_state),
        "--max-rows-per-file",
        str(args.max_rows_per_file),
        "--preceding-k",
        str(args.preceding_k),
        "--max-depth-limit",
        str(args.max_depth_limit),
    ]
    if args.dataset:
        cli.extend(["--dataset", args.dataset])
    if args.use_next_activity:
        cli.append("--use-next-activity")
    return cli


def main() -> None:
    args = parse_args()
    common_cli = build_common_cli(args)

    stage1_cmd = [
        sys.executable,
        "scripts/decision_tree_time_regime.py",
        "--result-dir",
        str(args.time_regime_result_dir),
    ] + common_cli

    stage2_cmd = [
        sys.executable,
        "scripts/decision_tree.py",
        "--result-dir",
        str(args.regression_result_dir),
    ] + common_cli

    print("Stage 1/2: decision_tree_time_regime.py")
    subprocess.run(stage1_cmd, check=True)

    print("Stage 2/2: decision_tree.py")
    subprocess.run(stage2_cmd, check=True)

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
