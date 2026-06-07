# T-Boost: Temporal Boosting

Utilities for process-mining experiments on event logs. The scripts in `scripts/` cover preprocessing, exploratory analysis, model training, evaluation, and plotting.

## Recommended order

If you want to run the full pipeline from raw logs to models and plots, use this order:

1. Preprocess raw event logs into CSV files.
2. Train the time-regime classifier.
3. Train the next-time regressor.
4. Evaluate the time-regime classifier.
5. Plot prediction errors.
6. Run exploratory analysis on the raw XES logs.

You can also skip steps you do not need. For example, `decision_tree_entrypoint.py` runs the two training stages back to back.

## Folder vs file input

Most scripts are folder-based and optionally let you narrow to one dataset by name.

- Folder + optional dataset name: `data_preprocess.py`, `data_preprocess_per_unit.py`, `decision_tree_time_regime.py`, `decision_tree.py`, `mlp.py`, `decision_tree_entrypoint.py`, `decision_tree_time_regime_eval.py`
- Direct file path input: `decision_tree_plot.py` via `--input`
- Folder-only (no single-file flag): `data_analysis.py` scans all `.xes` files in `--data-dir`

## Scripts

### `scripts/data_preprocess.py`

Converts raw logs in `data/` into enriched CSV files in `data_csv/`.

Input selection:
- Use `--data-dir` to point to a folder.
- Optional: use `--dataset` to process one file inside that folder.

What it produces:
- one CSV per input file
- prefix-based features
- `next_time`, `next_activity`, and `time_regime`

Example:

```bash
python scripts/data_preprocess.py --data-dir data --output-dir data_csv
```

Common options:
- `--data-dir`: input directory containing `.xes`, `.zip`, or `.csv` files
- `--output-dir`: output directory for generated CSV files
- `--time-unit`: `seconds`, `minutes`, `hours`, or `days`
- `--time-bucket`: optional rounding bucket for temporal features
- `--dataset`: process only one dataset file
- `--test-size`: fraction written to the test split

### `scripts/data_preprocess_per_unit.py`

Creates multiple CSV outputs split by `next_time_seconds` range.

Input selection:
- Use `--data-dir` to point to a folder.
- Optional: use `--dataset` to process one file inside that folder.

What it produces:
- `<dataset>_second.csv`
- `<dataset>_minute.csv`
- `<dataset>_hour.csv`
- `<dataset>_day.csv`

Example:

```bash
python scripts/data_preprocess_per_unit.py --data-dir data --output-dir data_csv
```

Common options:
- `--data-dir`
- `--output-dir`
- `--time-unit`
- `--dataset`

### `scripts/decision_tree_time_regime.py`

Trains a classifier to predict `time_regime` from event context.

Input selection:
- Use `--data-dir` for the folder containing split CSV files.
- Optional: use `--dataset` to train only one dataset.

Input:
- split CSV files in `data_csv/`

Output:
- model files
- metrics tables
- training artifacts under `results/decision_tree_time_regime/`

Example:

```bash
python scripts/decision_tree_time_regime.py --data-dir data_csv --result-dir results/decision_tree_time_regime
```

### `scripts/decision_tree.py`

Trains a regressor to predict `next_time` from event context.

Input selection:
- Use `--data-dir` for the folder containing split CSV files.
- Optional: use `--dataset` to train only one dataset.

Input:
- split CSV files in `data_csv/`

Output:
- model files
- metrics tables
- predictions and plots under `results/decision_tree/`

Example:

```bash
python scripts/decision_tree.py --data-dir data_csv --result-dir results/decision_tree
```

### `scripts/mlp.py`

Trains a PyTorch MLP regressor to predict `next_time` from event context.

Input selection:
- Use `--data-dir` for the folder containing split CSV files.
- Optional: use `--dataset` to train only one dataset.
- Optional: use `--train-file` and `--test-file` for explicit split files.

Input:
- split CSV files in `data_csv/`
- required soft time-regime columns:
	- `predicted_time_regime_q1_pct`
	- `predicted_time_regime_q2_pct`
	- `predicted_time_regime_q3_pct`

Output:
- model checkpoints under `results/mlp/models/`
- prediction CSV files under `results/mlp/predictions/`
- aggregate metrics at `results/mlp/mlp_metrics_all.csv`

Example:

```bash
python scripts/mlp.py --data-dir data_csv --train-file helpdesk_train_with_time_regime.csv --test-file helpdesk_test_with_time_regime.csv
```

### `scripts/decision_tree_entrypoint.py`

Runs the two model-training stages in sequence:

1. `scripts/decision_tree_time_regime.py`
2. `scripts/decision_tree.py`

Use this if you want the full training flow with one command.

Input selection:
- Use `--data-dir` for the folder containing split CSV files.
- Optional: use `--dataset` to run one dataset.

Example:

```bash
python scripts/decision_tree_entrypoint.py --data-dir data_csv --dataset helpdesk.csv
```

### `scripts/decision_tree_time_regime_eval.py`

Evaluates the trained time-regime classifier and writes metrics plus a confusion matrix plot.

Input selection:
- Uses `--data-dir` and `--result-dir` folders.
- Optional: use `--dataset` to evaluate one dataset stem.

Example:

```bash
python scripts/decision_tree_time_regime_eval.py --data-dir data_csv --result-dir results/decision_tree_time_regime
```

### `scripts/decision_tree_plot.py`

Plots normalized MAE curves from regressor prediction output.

Input selection:
- Uses `--input` with a direct prediction CSV file path.

Default input:
- `results/decision_tree/predictions/BPI12_prediction.csv`

Example:

```bash
python scripts/decision_tree_plot.py --input results/decision_tree/predictions/BPI12_prediction.csv
```

### `scripts/data_analysis.py`

Runs exploratory process-mining analysis on `.xes` files in `data/`.

Input selection:
- Uses `--data-dir` only and processes all `.xes` files in that folder.
- No single-file flag is available in this script.

What it produces per dataset:
- directly-follows graph PDF
- last-activity summary CSV
- sequence-length and execution-time plots
- variant Pareto plot
- trace-vs-timespan scatter plot and label mapping CSV

Example:

```bash
python scripts/data_analysis.py --data-dir data --output-dir results/data_analysis
```

## Quick pipeline example

A typical run looks like this:

```bash
python scripts/data_preprocess.py --data-dir data --output-dir data_csv
python scripts/decision_tree_entrypoint.py --data-dir data_csv
python scripts/mlp.py --data-dir data_csv --train-file helpdesk_train_with_time_regime.csv --test-file helpdesk_test_with_time_regime.csv
python scripts/decision_tree_time_regime_eval.py --data-dir data_csv --result-dir results/decision_tree_time_regime
python scripts/decision_tree_plot.py --input results/decision_tree/predictions/BPI12_prediction.csv
python scripts/data_analysis.py --data-dir data --output-dir results/data_analysis
```

## Notes

- `data_preprocess.py` and `data_preprocess_per_unit.py` are alternative preprocessing paths. Use the one that matches the dataset format you want.
- `decision_tree_entrypoint.py` is the fastest way to run both training stages in order.
- `mlp.py` expects soft time-regime probability columns (`predicted_time_regime_q1_pct`, `predicted_time_regime_q2_pct`, `predicted_time_regime_q3_pct`) in both train and test CSV files.
- All scripts accept `--help` for the full CLI reference.
