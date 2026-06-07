#!/usr/bin/env python3
"""
MLP with simple sequence embedding for predicting next event time in process logs.

This script trains neural network models to predict the inter-event time (next_time)
for business process events based on:
  - Event context (activity, timestamp, numeric features)
    - Temporal patterns (prefix sequences with pooled embedding)
    - Soft time-regime probabilities (predicted_time_regime_q1_pct/q2_pct/q3_pct)

Key Features:
    - TorchMLP: Multi-layer perceptron with pooled prefix-sequence embedding
  - Multiple loss functions: Huber (robust), MSE, MAE, weighted MSE
  - Sqrt-space transformation: Normalizes skewed time distributions
  - Feature engineering: Mirrors decision_tree.py for consistency
  - Early stopping: Based on validation loss with configurable patience

Input: CSV files with columns [activity, timestamp, prefix_sequence,
    time_span, predicted_time_regime_q1_pct, predicted_time_regime_q2_pct,
    predicted_time_regime_q3_pct, next_time, ...]
Note: Soft regime columns are required in both train and test inputs.
Output: Trained models, predictions, and metrics to results/mlp/

Usage:
  python scripts/mlp.py --data-dir data_csv --train-file helpdesk_train.csv \\
      --test-file helpdesk_test.csv --hidden-layers 512,256,128
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_DATA_DIR = Path("data_csv")
DEFAULT_RESULT_DIR = Path("results/mlp")
DEFAULT_EMBEDDING_DIM = 16
SECONDS_PER_DAY = 86400.0
TIME_UNIT_TO_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
}
SOFT_REGIME_COLUMNS = [
    "predicted_time_regime_q1_pct",
    "predicted_time_regime_q2_pct",
    "predicted_time_regime_q3_pct"
]


def apply_soft_regime_median_override(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SOFT_REGIME_COLUMNS:
        if col not in out.columns or col not in reference_df.columns:
            continue
        median_value = pd.to_numeric(reference_df[col], errors="coerce").median()
        out[col] = float(median_value) if pd.notna(median_value) else np.nan
    return out


class DenseBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return x


class TorchMLP(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        activity_cardinality: int,
        prefix_cardinality: int,
        hidden_layers: Tuple[int, ...],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.activity_embedding = nn.Embedding(activity_cardinality + 1, embedding_dim, padding_idx=0)
        self.prefix_embedding = nn.Embedding(prefix_cardinality + 1, embedding_dim, padding_idx=0)
        input_dim = numeric_dim + embedding_dim + embedding_dim

        self.hidden_blocks = nn.ModuleList()
        prev = input_dim
        for h in hidden_layers:
            self.hidden_blocks.append(DenseBlock(prev, h))
            prev = h
        self.output_layer = nn.Linear(prev, 1)

    def forward(
        self,
        numeric_x: torch.Tensor,
        activity_x: torch.Tensor,
        prefix_x: torch.Tensor,
        prefix_len_x: torch.Tensor,
    ) -> torch.Tensor:
        activity_emb = self.activity_embedding(activity_x)
        prefix_embeddings = self.prefix_embedding(prefix_x)
        seq_len = int(prefix_embeddings.shape[1])
        prefix_len_x = prefix_len_x.to(prefix_embeddings.device)
        step_idx = torch.arange(seq_len, device=prefix_embeddings.device).unsqueeze(0)
        seq_mask = (step_idx < prefix_len_x.unsqueeze(1)).unsqueeze(-1).to(prefix_embeddings.dtype)
        masked_sum = (prefix_embeddings * seq_mask).sum(dim=1)
        lengths = prefix_len_x.clamp(min=1).unsqueeze(1).to(prefix_embeddings.dtype)
        prefix_repr = masked_sum / lengths


        x = torch.cat([numeric_x, prefix_repr, activity_emb], dim=1)

        for block in self.hidden_blocks:
            x = block(x)

        return self.output_layer(x).squeeze(-1)


def print_model_architecture(model: nn.Module) -> None:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    print("Model architecture:")
    print(model)
    print("Parameter tensors:")
    for name, parameter in model.named_parameters():
        shape = tuple(parameter.shape)
        status = "trainable" if parameter.requires_grad else "frozen"
        print(f"  {name}: shape={shape}, numel={parameter.numel()}, {status}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def export_model_to_onnx(
    model: TorchMLP,
    export_path: Path,
    X_num_np: np.ndarray,
    X_activity_np: np.ndarray,
    X_prefix_np: np.ndarray,
    X_prefix_len_np: np.ndarray,
) -> None:
    if len(X_num_np) == 0:
        raise ValueError("Cannot export ONNX model without at least one example row")

    device = next(model.parameters()).device
    model.eval()
    example_num = torch.as_tensor(X_num_np[:1], dtype=torch.float32, device=device)
    example_activity = torch.as_tensor(X_activity_np[:1], dtype=torch.long, device=device)
    example_prefix = torch.as_tensor(X_prefix_np[:1], dtype=torch.long, device=device)
    example_prefix_len = torch.as_tensor(X_prefix_len_np[:1], dtype=torch.long, device=device)

    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example_num, example_activity, example_prefix, example_prefix_len),
        export_path,
        input_names=["numeric_x", "activity_x", "prefix_x", "prefix_len_x"],
        output_names=["next_time_prediction"],
        dynamic_axes={
            "numeric_x": {0: "batch_size"},
            "activity_x": {0: "batch_size"},
            "prefix_x": {0: "batch_size", 1: "prefix_length"},
            "prefix_len_x": {0: "batch_size"},
            "next_time_prediction": {0: "batch_size"},
        },
        opset_version=17,
    )


def _stable_file_seed(base_seed: int, file_name: str) -> int:
    return int(base_seed + sum(file_name.encode("utf-8")))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prefer deterministic kernels for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one MLP regressor per CSV file to predict next_time."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder containing input CSV files")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Folder to store all models and reports (default: results/mlp)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional single CSV filename to train (e.g., BPI12.csv). Default trains all CSVs.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="",
        help="Optional explicit train split filename (e.g., helpdesk_train.csv).",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="",
        help="Optional explicit test split filename (e.g., helpdesk_test_with_time_regime.csv).",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=0,
        help="Optional row cap per file for faster experiments (0 = use all rows)",
    )
    parser.add_argument(
        "--max-train-cases",
        type=int,
        default=0,
        help="Use only the first N temporal training cases (0 = use all training cases)",
    )
    parser.add_argument(
        "--max-iter",
        "--epochs",
        dest="max_iter",
        type=int,
        default=200,
        help="Maximum training epochs / MLP iterations (default: 200)",
    )
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="128,128",
        help="Comma-separated hidden layer sizes (default: 512,256,128)",
    )
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM, help="Embedding size for categorical features (default: 12)")
    parser.add_argument(
        "--time-rounding-seconds",
        type=float,
        default=0.0,
        help="Optional online rounding in seconds for temporal values; keep 0 when using preprocessing buckets (default: 0)",
    )
    parser.add_argument(
        "--sqrt-pred-clip-multiplier",
        type=float,
        default=3.0,
        help="Multiplier applied to sqrt-space prediction clip threshold (default: 3.0)",
    )
    parser.add_argument(
        "--disable-sqrt-pred-clip",
        action="store_true",
        help="Disable sqrt-space prediction clipping before inverse transform",
    )
    parser.add_argument(
        "--soft-regime-median-override",
        action="store_true",
        help="Replace all soft regime probability values with the training-column median for testing",
    )
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split from training slice (default: 0.1)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size (default: 1024)")
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Huber delta parameter (default: 1.0)")
    parser.set_defaults(loss_name="huber")
    parser.add_argument(
        "--loss-huber",
        dest="loss_name",
        action="store_const",
        const="huber",
        help="Use Huber loss (default)",
    )
    parser.add_argument(
        "--loss-mse",
        dest="loss_name",
        action="store_const",
        const="mse",
        help="Use Mean Squared Error loss",
    )
    parser.add_argument(
        "--loss-mae",
        dest="loss_name",
        action="store_const",
        const="mae",
        help="Use Mean Absolute Error loss",
    )
    parser.add_argument(
        "--loss-weighted-mse",
        dest="loss_name",
        action="store_const",
        const="weighted_mse",
        help="Use weighted MSE loss: log1p(next_time) * (pred-target)^2",
    )
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience on validation loss (default: 6)")
    parser.add_argument("--min-delta", type=float, default=1e-3, help="Minimum validation-loss decrease to reset patience")
    return parser.parse_args()


def parse_hidden_layers(value: str) -> Tuple[int, ...]:
    sizes = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not sizes:
        raise ValueError("--hidden-layers must contain at least one positive integer")
    if any(v <= 0 for v in sizes):
        raise ValueError("--hidden-layers must only contain positive integers")
    return tuple(sizes)


def list_csv_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {data_dir}")

    files = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return files


def strip_known_suffixes(stem: str) -> str:
    for suffix in ("_train_with_time_regime", "_test_with_time_regime", "_train", "_test"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def build_dataset_pairs(
    data_dir: Path,
    dataset: str,
    train_file: str = "",
    test_file: str = "",
) -> List[Tuple[str, Path, Path]]:
    files = list_csv_files(data_dir)
    by_name = {p.name: p for p in files}
    pairs: List[Tuple[str, Path, Path]] = []

    if train_file or test_file:
        if not train_file or not test_file:
            raise ValueError("When using explicit file override, provide both --train-file and --test-file")
        if train_file not in by_name:
            raise FileNotFoundError(f"Train split not found in {data_dir}: {train_file}")
        if test_file not in by_name:
            raise FileNotFoundError(f"Test split not found in {data_dir}: {test_file}")
        stem = strip_known_suffixes(Path(train_file).stem)
        return [(stem, by_name[train_file], by_name[test_file])]

    if dataset:
        stem = strip_known_suffixes(Path(dataset).stem)
        train_name = f"{stem}_train_with_time_regime.csv"
        if train_name not in by_name:
            train_name = f"{stem}_train.csv"
        test_name = f"{stem}_test_with_time_regime.csv"
        if test_name not in by_name:
            test_name = f"{stem}_test.csv"
        if train_name not in by_name:
            raise FileNotFoundError(f"Train split not found in {data_dir}: {train_name}")
        if test_name not in by_name:
            raise FileNotFoundError(f"Test split not found in {data_dir}: {test_name}")
        return [(stem, by_name[train_name], by_name[test_name])]

    for path in files:
        if not path.stem.endswith("_train"):
            continue
        stem = strip_known_suffixes(path.stem)
        soft_train_name = f"{stem}_train_with_time_regime.csv"
        if soft_train_name in by_name:
            path = by_name[soft_train_name]
        test_name = f"{stem}_test_with_time_regime.csv"
        test_path = by_name.get(test_name)
        if test_path is None:
            test_path = by_name.get(f"{stem}_test.csv")
        if test_path is None:
            continue
        pairs.append((stem, path, test_path))

    if not pairs:
        raise FileNotFoundError(
            f"No train/test pairs found in {data_dir}. Expected <dataset>_train.csv and <dataset>_test_with_time_regime.csv"
        )

    return sorted(pairs, key=lambda item: item[0])

def load_single_dataset(path: Path, max_rows_per_file: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if max_rows_per_file > 0:
        df = df.head(max_rows_per_file)
    return df


def validate_required_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    feature_cols = [
        "activity",
        "timestamp",
        "prefix_sequence",
        "time_span",
    ]
    target_col = "next_time"

    missing = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return feature_cols, target_col


def split_prefix_sequence(value: object) -> List[str]:
    if pd.isna(value):
        parts: List[str] = []
    else:
        text = str(value).strip()
        if not text:
            parts = []
        else:
            parts = [p.strip() for p in re.split(r"\s*(?:->|>|,|;|\|)\s*", text) if p.strip()]

    return parts


def _round_nonnegative_seconds(value: object, quantum_seconds: float) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return np.nan
    numeric_f = float(numeric)
    if not np.isfinite(numeric_f):
        return np.nan
    numeric_f = max(0.0, numeric_f)
    return float(np.round(numeric_f / quantum_seconds) * quantum_seconds)


def detect_time_unit(df: pd.DataFrame) -> str:
    if "time_unit" not in df.columns:
        return "seconds"

    series = df["time_unit"].dropna().astype(str).str.strip().str.lower()
    if series.empty:
        return "seconds"

    candidate = series.iloc[0]
    return candidate if candidate in TIME_UNIT_TO_SECONDS else "seconds"


def seconds_per_time_unit(time_unit: str) -> float:
    return float(TIME_UNIT_TO_SECONDS.get(str(time_unit).strip().lower(), 1.0))


def apply_time_rounding(df: pd.DataFrame, rounding_seconds: float, unit_seconds: float = 1.0) -> pd.DataFrame:
    if rounding_seconds <= 0:
        return df

    safe_unit_seconds = max(float(unit_seconds), 1e-9)
    quantum_in_data_unit = float(rounding_seconds) / safe_unit_seconds
    if quantum_in_data_unit <= 0:
        return df

    out = df.copy()
    out["time_span"] = out["time_span"].apply(lambda value: _round_nonnegative_seconds(value, quantum_in_data_unit))
    out["next_time"] = out["next_time"].apply(lambda value: _round_nonnegative_seconds(value, quantum_in_data_unit))
    return out


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out["timestamp_epoch"] = ts.astype("int64") / 1e9
    out.loc[ts.isna(), "timestamp_epoch"] = np.nan
    out["ts_dayofweek"] = ts.dt.dayofweek
    out["ts_hour"] = ts.dt.hour
    out["ts_month"] = ts.dt.month

    out["time_span"] = pd.to_numeric(out["time_span"], errors="coerce")
    out["next_time"] = pd.to_numeric(out["next_time"], errors="coerce")

    for col in SOFT_REGIME_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    prefix_values = out["prefix_sequence"].apply(split_prefix_sequence)
    out["prefix_length"] = prefix_values.map(len).astype(float)

    numeric_cols = [
        "timestamp_epoch",
        "time_span",
        "ts_dayofweek",
        "ts_hour",
        "ts_month",
        "prefix_length",
        *SOFT_REGIME_COLUMNS,
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["activity", "prefix_sequence", "time_regime"]:
        if col not in out.columns:
            continue
        out[col] = out[col].astype("object")
        out.loc[out[col].isna(), col] = np.nan

    return out


def get_categorical_columns() -> List[str]:
    return ["activity", "prefix_sequence"]


def get_numeric_columns() -> List[str]:
    base_cols = ["prefix_length", "time_span", "ts_dayofweek", "ts_hour", "ts_month"]
    return [*base_cols, *SOFT_REGIME_COLUMNS]


def build_category_maps(df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Dict[str, int]]:
    category_maps: Dict[str, Dict[str, int]] = {}
    for col in categorical_columns:
        if col == "prefix_sequence":
            values: List[str] = []
            for sequence in df[col].dropna().tolist():
                values.extend(split_prefix_sequence(sequence))
            category_maps[col] = {value: idx + 1 for idx, value in enumerate(sorted(set(values)))}
        else:
            values = sorted(df[col].dropna().astype(str).unique().tolist())
            category_maps[col] = {value: idx + 1 for idx, value in enumerate(values)}
    return category_maps


def encode_categories(
    df: pd.DataFrame,
    categorical_columns: List[str],
    category_maps: Dict[str, Dict[str, int]],
    max_prefix_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    activity_map = category_maps["activity"]
    prefix_map = category_maps["prefix_sequence"]
    activity_encoded = (
        df["activity"].astype(str).map(activity_map).fillna(0).astype(np.int64).to_numpy(copy=True)
    )

    prefix_encoded_rows: List[List[int]] = []
    prefix_lengths: List[int] = []
    safe_max_prefix_len = max(1, int(max_prefix_len))
    for sequence in df["prefix_sequence"].tolist():
        tokens = split_prefix_sequence(sequence)
        encoded_tokens = [prefix_map.get(token, 0) for token in tokens][:safe_max_prefix_len]
        seq_len = len(encoded_tokens)
        if seq_len < safe_max_prefix_len:
            encoded_tokens.extend([0] * (safe_max_prefix_len - seq_len))
        prefix_encoded_rows.append(encoded_tokens)
        prefix_lengths.append(seq_len)

    prefix_encoded = np.asarray(prefix_encoded_rows, dtype=np.int64).copy()
    prefix_lengths_np = np.asarray(prefix_lengths, dtype=np.int64).copy()
    return activity_encoded, prefix_encoded, prefix_lengths_np

def get_model_feature_columns() -> List[str]:
    return ["activity", "prefix_sequence"] + get_numeric_columns()


def limit_train_cases(train_df: pd.DataFrame, max_train_cases: int) -> pd.DataFrame:
    if max_train_cases <= 0:
        return train_df

    if "case_index" not in train_df.columns:
        return train_df.iloc[:max_train_cases].copy()

    case_series = pd.to_numeric(train_df["case_index"], errors="coerce")
    valid_case_df = train_df.loc[case_series.notna()].copy()
    if valid_case_df.empty:
        return train_df.iloc[:max_train_cases].copy()

    valid_case_df["__case_num"] = pd.to_numeric(valid_case_df["case_index"], errors="coerce")
    ordered_cases = np.sort(valid_case_df["__case_num"].unique())
    cutoff_cases = ordered_cases[:max_train_cases]
    limited = valid_case_df[valid_case_df["__case_num"].isin(cutoff_cases)].copy()
    return limited.drop(columns=["__case_num"])


def temporal_train_val_split(train_part: pd.DataFrame, val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if val_size <= 0:
        return train_part, train_part.iloc[0:0].copy()

    n_rows = len(train_part)
    if n_rows < 2:
        return train_part, train_part.iloc[0:0].copy()

    n_val = max(1, int(round(n_rows * val_size)))
    if n_val >= n_rows:
        n_val = n_rows - 1
    split_idx = n_rows - n_val
    return train_part.iloc[:split_idx].copy(), train_part.iloc[split_idx:].copy()


def to_float_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def fit_numeric_transformer(df: pd.DataFrame, numeric_columns: List[str]) -> Tuple[SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    numeric = df[numeric_columns].astype(float).to_numpy(dtype=np.float32)
    numeric_imputed = imputer.fit_transform(numeric)
    scaler.fit(numeric_imputed)
    return imputer, scaler


def transform_numeric(
    df: pd.DataFrame,
    numeric_columns: List[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    numeric = df[numeric_columns].astype(float).to_numpy(dtype=np.float32)
    numeric_imputed = imputer.transform(numeric)
    numeric_scaled = scaler.transform(numeric_imputed)
    return numeric_scaled.astype(np.float32)


def invert_sqrt_predictions(sqrt_predictions: np.ndarray, sqrt_clip_max: float) -> np.ndarray:
    clipped = np.clip(np.asarray(sqrt_predictions, dtype=float), a_min=0.0, a_max=float(sqrt_clip_max))
    return clipped ** 2


def plot_learning_curve(history: Dict[str, List[float]], output_path: Path, title: str) -> None:
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if not train_loss and not val_loss:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if train_loss:
        ax.plot(np.arange(1, len(train_loss) + 1), train_loss, label="Train loss", linewidth=2.0)
    if val_loss:
        ax.plot(np.arange(1, len(val_loss) + 1), val_loss, label="Validation loss", linewidth=2.0)

    best_series = val_loss if val_loss else train_loss
    best_label = "Validation loss" if val_loss else "Train loss"
    best_epoch = int(np.argmin(best_series) + 1)
    best_loss = float(np.min(best_series))
    ax.scatter([best_epoch], [best_loss], color="tab:red", s=45, zorder=3, label="Best epoch")
    ax.annotate(
        f"{best_label} best={best_loss:.4f}",
        xy=(best_epoch, best_loss),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        color="tab:red",
    )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def train_torch_model(
    X_train_num_np: np.ndarray,
    X_train_activity_np: np.ndarray,
    X_train_prefix_np: np.ndarray,
    X_train_prefix_len_np: np.ndarray,
    y_train_np: np.ndarray,
    X_val_num_np: np.ndarray,
    X_val_activity_np: np.ndarray,
    X_val_prefix_np: np.ndarray,
    X_val_prefix_len_np: np.ndarray,
    y_val_np: np.ndarray,
    y_train_seconds_np: np.ndarray,
    y_val_seconds_np: np.ndarray,
    args: argparse.Namespace,
    activity_cardinality: int,
    prefix_cardinality: int,
    sqrt_pred_clip_max: float,
    train_seed: int,
) -> Tuple[TorchMLP, Dict[str, List[float]], int, bool]:
    set_global_seed(train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    model = TorchMLP(
        numeric_dim=X_train_num_np.shape[1],
        activity_cardinality=activity_cardinality,
        prefix_cardinality=prefix_cardinality,
        hidden_layers=hidden_layers,
        embedding_dim=args.embedding_dim,
    ).to(device)
    print_model_architecture(model)

    if args.loss_name == "mse":
        criterion = nn.MSELoss()
    elif args.loss_name == "mae":
        criterion = nn.L1Loss()
    elif args.loss_name == "weighted_mse":
        criterion = None
    else:
        criterion = nn.HuberLoss(delta=args.huber_delta)

    def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if args.loss_name == "weighted_mse":
            # target is sqrt(next_time), so log1p(next_time) == log1p(target^2).
            weights = torch.log1p(torch.clamp(target, min=0.0) ** 2)
            return torch.mean(weights * (pred - target) ** 2)
        assert criterion is not None
        return criterion(pred, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

    train_dataset = TensorDataset(
        to_float_tensor(X_train_num_np, device),
        torch.as_tensor(X_train_activity_np, dtype=torch.long, device=device),
        torch.as_tensor(X_train_prefix_np, dtype=torch.long, device=device),
        torch.as_tensor(X_train_prefix_len_np, dtype=torch.long, device=device),
        to_float_tensor(y_train_np, device),
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(train_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=True,
        generator=loader_generator,
    )

    X_val_num_t = to_float_tensor(X_val_num_np, device) if len(X_val_num_np) > 0 else None
    X_val_activity_t = torch.as_tensor(X_val_activity_np, dtype=torch.long, device=device) if len(X_val_activity_np) > 0 else None
    X_val_prefix_t = torch.as_tensor(X_val_prefix_np, dtype=torch.long, device=device) if len(X_val_prefix_np) > 0 else None
    X_val_prefix_len_t = torch.as_tensor(X_val_prefix_len_np, dtype=torch.long, device=device) if len(X_val_prefix_len_np) > 0 else None
    y_val_t = to_float_tensor(y_val_np, device) if len(y_val_np) > 0 else None
    y_train_seconds_np = np.asarray(y_train_seconds_np, dtype=float)
    y_val_seconds_np = np.asarray(y_val_seconds_np, dtype=float)
    X_train_num_t = to_float_tensor(X_train_num_np, device)
    X_train_activity_t = torch.as_tensor(X_train_activity_np, dtype=torch.long, device=device)
    X_train_prefix_t = torch.as_tensor(X_train_prefix_np, dtype=torch.long, device=device)
    X_train_prefix_len_t = torch.as_tensor(X_train_prefix_len_np, dtype=torch.long, device=device)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_mae_days": [], "val_mae_days": []}
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 1
    no_improve = 0
    stopped_early = False

    for epoch in range(1, args.max_iter + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb_num, xb_activity, xb_prefix, xb_prefix_len, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb_num, xb_activity, xb_prefix, xb_prefix_len)
            loss = compute_loss(pred, yb)
            loss.backward()
            optimizer.step()

            batch_size = int(xb_num.shape[0])
            running_loss += float(loss.detach().item()) * batch_size
            n_seen += batch_size

        train_loss = running_loss / max(1, n_seen)
        history["train_loss"].append(float(train_loss))

        if (
            X_val_num_t is not None
            and X_val_activity_t is not None
            and X_val_prefix_t is not None
            and X_val_prefix_len_t is not None
            and y_val_t is not None
            and len(X_val_num_np) > 0
        ):
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_num_t, X_val_activity_t, X_val_prefix_t, X_val_prefix_len_t)
                val_loss = float(compute_loss(val_pred, y_val_t).item())
        else:
            val_loss = float(train_loss)

        history["val_loss"].append(float(val_loss))

        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_num_t, X_train_activity_t, X_train_prefix_t, X_train_prefix_len_t)
            train_pred_days = invert_sqrt_predictions(train_pred.detach().cpu().numpy(), sqrt_pred_clip_max)
            train_mae_days = float(mean_absolute_error(y_train_seconds_np, train_pred_days) / SECONDS_PER_DAY)

            if (
                X_val_num_t is not None
                and X_val_activity_t is not None
                and X_val_prefix_t is not None
                and X_val_prefix_len_t is not None
                and y_val_t is not None
                and len(X_val_num_np) > 0
                and len(y_val_seconds_np) > 0
            ):
                val_pred_days = invert_sqrt_predictions(
                    model(X_val_num_t, X_val_activity_t, X_val_prefix_t, X_val_prefix_len_t)
                    .detach()
                    .cpu()
                    .numpy(),
                    sqrt_pred_clip_max,
                )
                val_mae_days = float(mean_absolute_error(y_val_seconds_np, val_pred_days) / SECONDS_PER_DAY)
            else:
                val_mae_days = float(train_mae_days)

        history["train_mae_days"].append(float(train_mae_days))
        history["val_mae_days"].append(float(val_mae_days))

        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                stopped_early = True
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch, stopped_early


def predict_torch_embeddings(
    model: TorchMLP,
    X_num_np: np.ndarray,
    X_activity_np: np.ndarray,
    X_prefix_np: np.ndarray,
    X_prefix_len_np: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    device = next(model.parameters()).device
    dataset = TensorDataset(
        torch.as_tensor(X_num_np, dtype=torch.float32),
        torch.as_tensor(X_activity_np, dtype=torch.long),
        torch.as_tensor(X_prefix_np, dtype=torch.long),
        torch.as_tensor(X_prefix_len_np, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False)
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for xb_num, xb_activity, xb_prefix, xb_prefix_len in loader:
            xb_num = xb_num.to(device)
            xb_activity = xb_activity.to(device)
            xb_prefix = xb_prefix.to(device)
            xb_prefix_len = xb_prefix_len.to(device)
            pred = model(xb_num, xb_activity, xb_prefix, xb_prefix_len).detach().cpu().numpy()
            preds.append(pred)
    if not preds:
        return np.array([], dtype=float)
    return np.concatenate(preds, axis=0)


def train_one_pair(train_csv_path: Path, test_csv_path: Path, dataset_stem: str, args: argparse.Namespace) -> Dict[str, object]:
    raw_train_data = load_single_dataset(train_csv_path, args.max_rows_per_file)
    raw_test_data = load_single_dataset(test_csv_path, args.max_rows_per_file)

    regime_feature_mode = "soft_probabilities"
    regime_feature_columns = ",".join(SOFT_REGIME_COLUMNS)

    missing_soft = [
        col
        for col in SOFT_REGIME_COLUMNS
        if col not in raw_train_data.columns or col not in raw_test_data.columns
    ]
    if missing_soft:
        raise ValueError(f"Missing required soft time-regime columns: {missing_soft}")

    _, target_col = validate_required_columns(raw_train_data)
    validate_required_columns(raw_test_data)
    dataset_time_unit = detect_time_unit(raw_train_data)
    dataset_unit_seconds = seconds_per_time_unit(dataset_time_unit)
    raw_train_data = apply_time_rounding(raw_train_data, args.time_rounding_seconds, dataset_unit_seconds)
    raw_test_data = apply_time_rounding(raw_test_data, args.time_rounding_seconds, dataset_unit_seconds)

    train_data = coerce_types(raw_train_data)
    test_data = coerce_types(raw_test_data)
    if args.soft_regime_median_override:
        train_data = apply_soft_regime_median_override(train_data, train_data)
        test_data = apply_soft_regime_median_override(test_data, train_data)
    train_data["__row_id"] = train_data.index
    test_data["__row_id"] = test_data.index

    train_df = train_data.dropna(subset=[target_col]).copy()
    test_df = test_data.dropna(subset=[target_col]).copy()
    if len(train_df) < 2 or len(test_df) < 1:
        return {
            "file": f"{dataset_stem}.csv",
            "train_file": train_csv_path.name,
            "test_file": test_csv_path.name,
            "rows_total": int(len(train_data) + len(test_data)),
            "rows_used_for_training": int(len(train_df) + len(test_df)),
            "rows_train": 0,
            "rows_test": 0,
            "mae": np.nan,
            "mae_days": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "model_path": "",
            "status": "skipped_not_enough_rows",
        }

    file_seed = _stable_file_seed(args.random_state, train_csv_path.name)

    train_part = limit_train_cases(train_df, args.max_train_cases)
    test_part = test_df
    if train_part.empty or test_part.empty:
        return {
            "file": f"{dataset_stem}.csv",
            "train_file": train_csv_path.name,
            "test_file": test_csv_path.name,
            "rows_total": int(len(train_data) + len(test_data)),
            "rows_used_for_training": int(len(train_df) + len(test_df)),
            "rows_train": int(len(train_part)),
            "rows_test": int(len(test_part)),
            "mae": np.nan,
            "mae_days": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "model_path": "",
            "status": "skipped_invalid_split_files",
        }

    feature_cols = get_model_feature_columns()
    X_train = train_part[feature_cols]
    X_test = test_part[feature_cols]
    y_test = test_part[target_col]

    train_fit_part, val_fit_part = temporal_train_val_split(train_part, args.val_size)
    if train_fit_part.empty:
        raise ValueError("Training split is empty after validation split")

    categorical_columns = get_categorical_columns()
    numeric_columns = get_numeric_columns()
    category_maps = build_category_maps(train_fit_part, categorical_columns)
    activity_cardinality = len(category_maps["activity"])
    prefix_cardinality = len(category_maps["prefix_sequence"])
    max_prefix_len = int(train_fit_part["prefix_length"].max()) if not train_fit_part.empty else 0
    numeric_imputer, numeric_scaler = fit_numeric_transformer(train_fit_part, numeric_columns)

    X_fit_num_np = transform_numeric(train_fit_part, numeric_columns, numeric_imputer, numeric_scaler)
    X_fit_activity_np, X_fit_prefix_np, X_fit_prefix_len_np = encode_categories(
        train_fit_part,
        categorical_columns,
        category_maps,
        max_prefix_len=max_prefix_len,
    )
    X_val_num_np = (
        transform_numeric(val_fit_part, numeric_columns, numeric_imputer, numeric_scaler)
        if not val_fit_part.empty
        else np.empty((0, len(numeric_columns)), dtype=np.float32)
    )
    X_val_activity_np, X_val_prefix_np, X_val_prefix_len_np = (
        encode_categories(
            val_fit_part,
            categorical_columns,
            category_maps,
            max_prefix_len=max_prefix_len,
        )
        if not val_fit_part.empty
        else (
            np.empty((0,), dtype=np.int64),
            np.empty((0, max(1, max_prefix_len)), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    )
    X_train_num_np = transform_numeric(X_train, numeric_columns, numeric_imputer, numeric_scaler)
    X_train_activity_np, X_train_prefix_np, X_train_prefix_len_np = encode_categories(
        X_train,
        categorical_columns,
        category_maps,
        max_prefix_len=max_prefix_len,
    )
    X_test_num_np = transform_numeric(X_test, numeric_columns, numeric_imputer, numeric_scaler)
    X_test_activity_np, X_test_prefix_np, X_test_prefix_len_np = encode_categories(
        X_test,
        categorical_columns,
        category_maps,
        max_prefix_len=max_prefix_len,
    )

    y_fit_sqrt = np.sqrt(np.clip(train_fit_part[target_col].to_numpy(dtype=float), a_min=0.0, a_max=None))
    y_val_sqrt = (
        np.sqrt(np.clip(val_fit_part[target_col].to_numpy(dtype=float), a_min=0.0, a_max=None))
        if not val_fit_part.empty
        else np.array([], dtype=float)
    )
    train_target_nonneg = np.clip(train_fit_part[target_col].to_numpy(dtype=float), a_min=0.0, a_max=None)
    max_train_target = float(np.max(train_target_nonneg)) if len(train_target_nonneg) > 0 else 0.0
    # Keep clipping configurable to allow controlled extrapolation when desired.
    base_sqrt_clip = np.sqrt(max_train_target) + 1.0
    clip_multiplier = max(1.0, float(args.sqrt_pred_clip_multiplier))
    sqrt_pred_clip_max = float("inf") if args.disable_sqrt_pred_clip else base_sqrt_clip * clip_multiplier

    model, history, best_epoch, stopped_early = train_torch_model(
        X_fit_num_np,
        X_fit_activity_np,
        X_fit_prefix_np,
        X_fit_prefix_len_np,
        y_fit_sqrt,
        X_val_num_np,
        X_val_activity_np,
        X_val_prefix_np,
        X_val_prefix_len_np,
        y_val_sqrt,
        train_fit_part[target_col].to_numpy(dtype=float),
        val_fit_part[target_col].to_numpy(dtype=float) if not val_fit_part.empty else np.array([], dtype=float),
        args,
        activity_cardinality,
        prefix_cardinality,
        sqrt_pred_clip_max,
        file_seed,
    )

    plots_dir = args.result_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    learning_curve_path = plots_dir / f"{dataset_stem}_learning_curve.png"
    plot_learning_curve(history, learning_curve_path, f"Learning Curve: {dataset_stem}")

    learning_curve_csv_path = plots_dir / f"{dataset_stem}_learning_curve.csv"
    history_df = pd.DataFrame({
        "epoch": np.arange(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_mae_days": history.get("train_mae_days", [np.nan] * len(history["train_loss"])),
        "val_mae_days": history.get("val_mae_days", [np.nan] * len(history["train_loss"])),
    })
    history_df.to_csv(learning_curve_csv_path, index=False)

    y_pred_test = invert_sqrt_predictions(
        predict_torch_embeddings(
            model,
            X_test_num_np,
            X_test_activity_np,
            X_test_prefix_np,
            X_test_prefix_len_np,
            batch_size=args.batch_size,
        ),
        sqrt_clip_max=sqrt_pred_clip_max,
    )
    y_pred_test = np.clip(y_pred_test, a_min=0.0, a_max=None)
    mae = mean_absolute_error(y_test, y_pred_test)
    mae_days = (float(mae) * dataset_unit_seconds) / SECONDS_PER_DAY
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    y_pred_train = invert_sqrt_predictions(
        predict_torch_embeddings(
            model,
            X_train_num_np,
            X_train_activity_np,
            X_train_prefix_np,
            X_train_prefix_len_np,
            batch_size=args.batch_size,
        ),
        sqrt_clip_max=sqrt_pred_clip_max,
    )
    y_pred_train = np.clip(y_pred_train, a_min=0.0, a_max=None)

    train_part = train_part.copy()
    test_part = test_part.copy()
    train_part["prediction_next_time"] = y_pred_train
    test_part["prediction_next_time"] = y_pred_test
    train_part["split"] = "train"
    test_part["split"] = "test"

    train_input_cols = list(raw_train_data.columns)
    raw_train_with_row_id = raw_train_data.copy()
    raw_train_with_row_id["__row_id"] = raw_train_with_row_id.index
    train_export = raw_train_with_row_id.merge(
        train_part[["__row_id", "split", "prediction_next_time"]],
        on="__row_id",
        how="inner",
    )
    train_export["actual_next_time"] = pd.to_numeric(train_export["next_time"], errors="coerce")
    train_export["prediction_error"] = train_export["actual_next_time"] - train_export["prediction_next_time"]
    train_export = train_export[
        train_input_cols + ["split", "actual_next_time", "prediction_next_time", "prediction_error"]
    ]

    test_input_cols = list(raw_test_data.columns)
    raw_test_with_row_id = raw_test_data.copy()
    raw_test_with_row_id["__row_id"] = raw_test_with_row_id.index
    test_export = raw_test_with_row_id.merge(
        test_part[["__row_id", "split", "prediction_next_time"]],
        on="__row_id",
        how="inner",
    )
    test_export["actual_next_time"] = pd.to_numeric(test_export["next_time"], errors="coerce")
    test_export["prediction_error"] = test_export["actual_next_time"] - test_export["prediction_next_time"]
    test_export = test_export[
        test_input_cols + ["split", "actual_next_time", "prediction_next_time", "prediction_error"]
    ]

    prediction_export = pd.concat([train_export, test_export], ignore_index=True)

    models_dir = args.result_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{dataset_stem}_mlp_next_time.pt"
    checkpoint = {
        "model_type": "pytorch_mlp",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "numeric_dim": int(X_fit_num_np.shape[1]),
        "activity_cardinality": int(activity_cardinality),
        "prefix_cardinality": int(prefix_cardinality),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "category_maps": category_maps,
        "numeric_imputer": numeric_imputer,
        "numeric_scaler": numeric_scaler,
        "hidden_layers": parse_hidden_layers(args.hidden_layers),
        "feature_cols": feature_cols,
        "history": history,
        "best_epoch": int(best_epoch),
        "stopped_early": bool(stopped_early),
        "target_transform": "sqrt",
        "sqrt_pred_clip_max": float(sqrt_pred_clip_max),
        "embedding_dim": int(args.embedding_dim),
        "time_unit": str(dataset_time_unit),
        "time_unit_seconds": float(dataset_unit_seconds),
        "time_rounding_seconds": float(args.time_rounding_seconds),
        "sqrt_pred_clip_multiplier": float(args.sqrt_pred_clip_multiplier),
        "disable_sqrt_pred_clip": bool(args.disable_sqrt_pred_clip),
        "max_prefix_len": int(max_prefix_len),
    }
    torch.save(checkpoint, model_path)

    onnx_path = models_dir / f"{dataset_stem}_mlp_next_time.onnx"
    try:
        export_model_to_onnx(
            model,
            onnx_path,
            X_fit_num_np,
            X_fit_activity_np,
            X_fit_prefix_np,
            X_fit_prefix_len_np,
        )
        print(f"Saved ONNX model: {onnx_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to export ONNX model for {dataset_stem}: {exc}")

    predictions_dir = args.result_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = predictions_dir / f"{dataset_stem}_prediction.csv"
    prediction_export.to_csv(prediction_path, index=False)

    return {
        "file": f"{dataset_stem}.csv",
        "train_file": train_csv_path.name,
        "test_file": test_csv_path.name,
        "rows_total": int(len(train_data) + len(test_data)),
        "rows_used_for_training": int(len(train_df) + len(test_df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "max_train_cases": int(args.max_train_cases),
        "train_cases_used": int(train_part["case_index"].nunique()) if "case_index" in train_part.columns else np.nan,
        "target_transform": "sqrt",
        "time_regime_feature_mode": regime_feature_mode,
        "time_regime_feature_columns": regime_feature_columns,
        "time_unit": str(dataset_time_unit),
        "time_unit_seconds": float(dataset_unit_seconds),
        "time_rounding_seconds": float(args.time_rounding_seconds),
        "loss_name": str(args.loss_name),
        "train_epochs": int(len(history["train_loss"])),
        "best_epoch": int(best_epoch),
        "stopped_early": bool(stopped_early),
        "train_loss_last": float(history["train_loss"][-1]) if history["train_loss"] else np.nan,
        "val_loss_last": float(history["val_loss"][-1]) if history["val_loss"] else np.nan,
        "train_mae_days_last": float(history["train_mae_days"][-1]) if history["train_mae_days"] else np.nan,
        "val_mae_days_last": float(history["val_mae_days"][-1]) if history["val_mae_days"] else np.nan,
        "mae": float(mae),
        "mae_days": float(mae_days),
        "rmse": float(rmse),
        "r2": float(r2),
        "model_path": str(model_path),
        "onnx_path": str(onnx_path),
        "learning_curve_path": str(learning_curve_path),
        "learning_curve_csv_path": str(learning_curve_csv_path),
        "prediction_path": str(prediction_path),
        "status": "trained",
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.random_state)
    pairs = build_dataset_pairs(
        args.data_dir,
        args.dataset,
        args.train_file,
        args.test_file,
    )

    args.result_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    for dataset_stem, train_path, test_path in pairs:
        try:
            row = train_one_pair(train_path, test_path, dataset_stem, args)
        except Exception as exc:  # noqa: BLE001
            row = {
                "file": f"{dataset_stem}.csv",
                "train_file": train_path.name,
                "test_file": test_path.name,
                "rows_total": np.nan,
                "rows_used_for_training": np.nan,
                "rows_train": np.nan,
                "rows_test": np.nan,
                "mae": np.nan,
                "mae_days": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "model_path": "",
                "status": f"failed: {exc}",
            }
        all_rows.append(row)
        print(f"{dataset_stem}: {row['status']}")

    new_metrics_df = pd.DataFrame(all_rows)
    report_path = args.result_dir / "mlp_metrics_all.csv"
    if report_path.exists():
        existing_metrics_df = pd.read_csv(report_path)
        metrics_df = pd.concat([existing_metrics_df, new_metrics_df], ignore_index=True, sort=False)
    else:
        metrics_df = new_metrics_df

    metrics_df.to_csv(report_path, index=False)

    trained_count = int((new_metrics_df["status"] == "trained").sum())
    print(f"Trained models this run: {trained_count}/{len(pairs)}")
    print(f"Saved aggregate metrics: {report_path}")


if __name__ == "__main__":
    main()
