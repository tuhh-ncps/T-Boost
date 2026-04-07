#!/usr/bin/env python3
"""Train MLP regressors to predict next_time from event context.

This script mirrors the feature engineering from decision_tree.py and writes
outputs to results/mlp by default.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_DATA_DIR = Path("data_csv")
DEFAULT_RESULT_DIR = Path("results/mlp")
DEFAULT_EMBEDDING_DIM = 16


class DenseBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_batch_norm: bool, dropout_rate: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features) if use_batch_norm else None
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class TorchMLP(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        activity_cardinality: int,
        next_activity_cardinality: int,
        prefix_cardinality: int,
        hidden_layers: Tuple[int, ...],
        embedding_dim: int,
        gru_hidden_dim: int,
        gru_num_layers: int,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        use_next_activity: bool = True,
    ) -> None:
        super().__init__()
        self.use_next_activity = use_next_activity
        self.activity_embedding = nn.Embedding(activity_cardinality + 1, embedding_dim, padding_idx=0)
        self.next_activity_embedding = nn.Embedding(next_activity_cardinality + 1, embedding_dim, padding_idx=0)
        self.prefix_embedding = nn.Embedding(prefix_cardinality + 1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, gru_hidden_dim, num_layers=gru_num_layers, batch_first=True)

        input_dim = numeric_dim + gru_hidden_dim + embedding_dim
        if self.use_next_activity:
            input_dim += embedding_dim
        
        self.input_batch_norm = nn.BatchNorm1d(input_dim) if use_batch_norm else None
        self.hidden_blocks = nn.ModuleList()
        prev = input_dim
        for h in hidden_layers:
            self.hidden_blocks.append(DenseBlock(prev, h, use_batch_norm, dropout_rate))
            prev = h
        self.output_layer = nn.Linear(prev, 1)

    def _set_module_trainable(self, module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = trainable
        module.train(trainable)

    def configure_client_training(self, freeze_embeddings: bool, trainable_hidden_layers: int) -> None:
        """Freeze embeddings and keep the GRU/trainable MLP blocks updateable.

        Args:
            freeze_embeddings: If True, keep the embeddings frozen.
            trainable_hidden_layers: Number of final hidden blocks to train.
                A negative value keeps all hidden blocks trainable.
        """
        embedding_modules = [
            self.activity_embedding,
            self.next_activity_embedding,
            self.prefix_embedding,
        ]

        for module in embedding_modules:
            self._set_module_trainable(module, not freeze_embeddings)

        # Keep GRU and input batch norm trainable so local clients can adapt sequence dynamics.
        self._set_module_trainable(self.gru, True)
        if self.input_batch_norm is not None:
            self._set_module_trainable(self.input_batch_norm, True)

        total_blocks = len(self.hidden_blocks)
        if trainable_hidden_layers < 0 or trainable_hidden_layers >= total_blocks:
            first_trainable_block = 0
        else:
            first_trainable_block = max(0, total_blocks - trainable_hidden_layers)

        for block_index, block in enumerate(self.hidden_blocks):
            self._set_module_trainable(block, block_index >= first_trainable_block)

        self._set_module_trainable(self.output_layer, True)

    def forward(
        self,
        numeric_x: torch.Tensor,
        activity_x: torch.Tensor,
        prefix_x: torch.Tensor,
        prefix_len_x: torch.Tensor,
        next_activity_x: torch.Tensor,
    ) -> torch.Tensor:
        activity_emb = self.activity_embedding(activity_x)
        next_activity_emb = self.next_activity_embedding(next_activity_x)
        prefix_embeddings = self.prefix_embedding(prefix_x)
        prefix_outputs, _ = self.gru(prefix_embeddings)

        # Use the hidden state at the last valid prefix token; if no prefix, use zeros.
        safe_prefix_len = torch.clamp(prefix_len_x, min=1)
        gather_idx = (safe_prefix_len - 1).view(-1, 1, 1).expand(-1, 1, prefix_outputs.size(2))
        prefix_repr = prefix_outputs.gather(1, gather_idx).squeeze(1)
        no_prefix_mask = prefix_len_x == 0
        if no_prefix_mask.any():
            prefix_repr = prefix_repr.clone()
            prefix_repr[no_prefix_mask] = 0.0

        if self.use_next_activity:
            x = torch.cat([numeric_x, prefix_repr, activity_emb, next_activity_emb], dim=1)
        else:
            x = torch.cat([numeric_x, prefix_repr, activity_emb], dim=1)
        
        if self.input_batch_norm is not None:
            x = self.input_batch_norm(x)
        
        for block in self.hidden_blocks:
            x = block(x)

        return self.output_layer(x).squeeze(-1)


def _stable_file_seed(base_seed: int, file_name: str) -> int:
    return int(base_seed + sum(file_name.encode("utf-8")))


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
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional single CSV filename to train (e.g., BPI12.csv). Default trains all CSVs.",
    )
    parser.add_argument(
        "--next-activity-noise-rate",
        type=float,
        default=0.05,
        help="Probability of replacing next_activity with a different value (default: 0.05)",
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
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum MLP iterations (default: 200)")
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="512,256,128",
        help="Comma-separated hidden layer sizes (default: 512,256,128)",
    )
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM, help="Embedding size for categorical features (default: 12)")
    parser.add_argument("--gru-hidden-dim", type=int, default=16, help="GRU hidden size for prefix encoding (default: 16)")
    parser.add_argument("--gru-num-layers", type=int, default=1, help="Number of GRU layers for prefix encoding (default: 1)")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split from training slice (default: 0.1)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size (default: 1024)")
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Huber delta parameter (default: 1.0)")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience on validation loss (default: 6)")
    parser.add_argument("--min-delta", type=float, default=1e-3, help="Minimum validation-loss decrease to reset patience")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in MLP layers (default: 0.0)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay in optimizer (default: 0.0)")
    parser.add_argument("--use-batch-norm", action="store_true", help="Enable batch normalization in MLP layers")
    parser.set_defaults(drop_next_activity=True)
    parser.add_argument(
        "--drop-next-activity",
        dest="drop_next_activity",
        action="store_true",
        help="Exclude next_activity embedding from model input (default behavior)",
    )
    parser.add_argument(
        "--use-next-activity",
        dest="drop_next_activity",
        action="store_false",
        help="Include next_activity embedding in model input",
    )
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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "next_actvity" in df.columns and "next_activity" not in df.columns:
        df = df.rename(columns={"next_actvity": "next_activity"})
    return df


def load_single_dataset(path: Path, max_rows_per_file: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)
    if max_rows_per_file > 0:
        df = df.head(max_rows_per_file)
    return df


def validate_required_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    feature_cols = ["activity", "timestamp", "prefix_sequence", "time_span", "next_activity"]
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

    prefix_values = out["prefix_sequence"].apply(split_prefix_sequence)
    out["prefix_length"] = prefix_values.map(len).astype(float)

    numeric_cols = ["timestamp_epoch", "time_span", "ts_dayofweek", "ts_hour", "ts_month", "prefix_length"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["activity", "prefix_sequence", "next_activity"]:
        out[col] = out[col].astype("object")
        out.loc[out[col].isna(), col] = np.nan

    return out


def get_categorical_columns() -> List[str]:
    return ["activity", "prefix_sequence", "next_activity"]


def get_numeric_columns() -> List[str]:
    return ["prefix_length", "time_span", "ts_dayofweek", "ts_hour", "ts_month"]


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    activity_map = category_maps["activity"]
    prefix_map = category_maps["prefix_sequence"]
    next_activity_map = category_maps["next_activity"]

    activity_encoded = (
        df["activity"].astype(str).map(activity_map).fillna(0).astype(np.int64).to_numpy(copy=True)
    )
    next_activity_encoded = (
        df["next_activity"].astype(str).map(next_activity_map).fillna(0).astype(np.int64).to_numpy(copy=True)
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
    return activity_encoded, prefix_encoded, next_activity_encoded, prefix_lengths_np


def encode_numeric(df: pd.DataFrame, numeric_columns: List[str]) -> np.ndarray:
    return df[numeric_columns].astype(float).to_numpy(dtype=np.float32)


def inject_next_activity_noise(df: pd.DataFrame, noise_rate: float, random_state: int) -> pd.DataFrame:
    out = df.copy()
    if noise_rate <= 0:
        return out

    values = [v for v in out["next_activity"].dropna().unique().tolist() if str(v).strip()]
    if len(values) < 2:
        return out

    rng = np.random.default_rng(random_state)
    mask = rng.random(len(out)) < noise_rate
    n_masked = int(mask.sum())
    if n_masked == 0:
        return out

    current_vals = out.loc[mask, "next_activity"].to_numpy(dtype=object)
    new_vals = rng.choice(values, size=n_masked)

    same = new_vals == current_vals
    tries = 0
    while same.any() and tries < 5:
        new_vals[same] = rng.choice(values, size=int(same.sum()))
        same = new_vals == current_vals
        tries += 1

    if same.any():
        idx_map = {v: i for i, v in enumerate(values)}
        for i, is_same in enumerate(same):
            if is_same:
                cur = current_vals[i]
                if cur in idx_map:
                    new_vals[i] = values[(idx_map[cur] + 1) % len(values)]

    out.loc[mask, "next_activity"] = new_vals
    return out


def get_model_feature_columns() -> List[str]:
    return get_categorical_columns() + get_numeric_columns()


def temporal_split(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("timestamp_epoch", kind="mergesort", na_position="last").reset_index(drop=True)
    n_rows = len(ordered)
    if n_rows < 2:
        return ordered, ordered.iloc[0:0]

    n_test = max(1, int(round(n_rows * test_size)))
    if n_test >= n_rows:
        n_test = n_rows - 1
    split_idx = n_rows - n_test

    train_df = ordered.iloc[:split_idx].copy()
    test_df = ordered.iloc[split_idx:].copy()
    return train_df, test_df


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


def invert_log1p_predictions(log_predictions: np.ndarray, log_clip_max: float) -> np.ndarray:
    clipped = np.clip(log_predictions, -20.0, float(log_clip_max))
    return np.expm1(clipped)


def compute_normalized_mae(
    y_true: Optional[np.ndarray],
    y_pred: Optional[np.ndarray] = None,
    mae: Optional[float] = None,
) -> Optional[float]:
    """Compute normalized MAE (MAE / median of y_true).
    
    Args:
        y_true: True values
        y_pred: Predicted values (not needed, included for consistency)
        mae: Pre-computed MAE (if not provided, will be computed from y_true and y_pred)
    
    Returns:
        Normalized MAE (MAE divided by median of y_true) or None if cannot compute
    """
    if y_true is None or len(y_true) == 0:
        return None
    
    y_median = float(np.median(y_true))
    if y_median == 0:
        return None
    
    if mae is None:
        if y_pred is None or len(y_pred) == 0:
            return None
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_true, y_pred)
    
    return float(mae / y_median)


def train_torch_model(
    X_train_num_np: np.ndarray,
    X_train_activity_np: np.ndarray,
    X_train_prefix_np: np.ndarray,
    X_train_prefix_len_np: np.ndarray,
    X_train_next_activity_np: np.ndarray,
    y_train_np: np.ndarray,
    X_val_num_np: np.ndarray,
    X_val_activity_np: np.ndarray,
    X_val_prefix_np: np.ndarray,
    X_val_prefix_len_np: np.ndarray,
    X_val_next_activity_np: np.ndarray,
    y_val_np: np.ndarray,
    args: argparse.Namespace,
    activity_cardinality: int,
    next_activity_cardinality: int,
    prefix_cardinality: int,
) -> Tuple[TorchMLP, Dict[str, List[float]], int, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    model = TorchMLP(
        numeric_dim=X_train_num_np.shape[1],
        activity_cardinality=activity_cardinality,
        next_activity_cardinality=next_activity_cardinality,
        prefix_cardinality=prefix_cardinality,
        hidden_layers=hidden_layers,
        embedding_dim=args.embedding_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_num_layers=args.gru_num_layers,
        dropout_rate=args.dropout,
        use_batch_norm=args.use_batch_norm,
        use_next_activity=not args.drop_next_activity,
    ).to(device)

    criterion = nn.HuberLoss(delta=args.huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_dataset = TensorDataset(
        to_float_tensor(X_train_num_np, device),
        torch.as_tensor(X_train_activity_np, dtype=torch.long, device=device),
        torch.as_tensor(X_train_prefix_np, dtype=torch.long, device=device),
        torch.as_tensor(X_train_prefix_len_np, dtype=torch.long, device=device),
        torch.as_tensor(X_train_next_activity_np, dtype=torch.long, device=device),
        to_float_tensor(y_train_np, device),
    )
    train_loader = DataLoader(train_dataset, batch_size=max(1, args.batch_size), shuffle=True)

    X_val_num_t = to_float_tensor(X_val_num_np, device) if len(X_val_num_np) > 0 else None
    X_val_activity_t = torch.as_tensor(X_val_activity_np, dtype=torch.long, device=device) if len(X_val_activity_np) > 0 else None
    X_val_prefix_t = torch.as_tensor(X_val_prefix_np, dtype=torch.long, device=device) if len(X_val_prefix_np) > 0 else None
    X_val_prefix_len_t = torch.as_tensor(X_val_prefix_len_np, dtype=torch.long, device=device) if len(X_val_prefix_len_np) > 0 else None
    X_val_next_activity_t = torch.as_tensor(X_val_next_activity_np, dtype=torch.long, device=device) if len(X_val_next_activity_np) > 0 else None
    y_val_t = to_float_tensor(y_val_np, device) if len(y_val_np) > 0 else None

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 1
    no_improve = 0
    stopped_early = False

    for epoch in range(1, args.max_iter + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb_num, xb_activity, xb_prefix, xb_prefix_len, xb_next_activity, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb_num, xb_activity, xb_prefix, xb_prefix_len, xb_next_activity)
            loss = criterion(pred, yb)
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
            and X_val_next_activity_t is not None
            and y_val_t is not None
            and len(X_val_num_np) > 0
        ):
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_num_t, X_val_activity_t, X_val_prefix_t, X_val_prefix_len_t, X_val_next_activity_t)
                val_loss = float(criterion(val_pred, y_val_t).item())
        else:
            val_loss = float(train_loss)

        history["val_loss"].append(float(val_loss))

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
    X_next_activity_np: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    device = next(model.parameters()).device
    dataset = TensorDataset(
        torch.as_tensor(X_num_np, dtype=torch.float32),
        torch.as_tensor(X_activity_np, dtype=torch.long),
        torch.as_tensor(X_prefix_np, dtype=torch.long),
        torch.as_tensor(X_prefix_len_np, dtype=torch.long),
        torch.as_tensor(X_next_activity_np, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False)
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for xb_num, xb_activity, xb_prefix, xb_prefix_len, xb_next_activity in loader:
            xb_num = xb_num.to(device)
            xb_activity = xb_activity.to(device)
            xb_prefix = xb_prefix.to(device)
            xb_prefix_len = xb_prefix_len.to(device)
            xb_next_activity = xb_next_activity.to(device)
            pred = model(xb_num, xb_activity, xb_prefix, xb_prefix_len, xb_next_activity).detach().cpu().numpy()
            preds.append(pred)
    if not preds:
        return np.array([], dtype=float)
    return np.concatenate(preds, axis=0)


def train_one_file(csv_path: Path, args: argparse.Namespace) -> Dict[str, object]:
    raw_data = load_single_dataset(csv_path, args.max_rows_per_file)
    _, target_col = validate_required_columns(raw_data)

    data = coerce_types(raw_data)
    data["__row_id"] = data.index

    train_df = data.dropna(subset=[target_col]).copy()
    if len(train_df) < 2:
        return {
            "file": csv_path.name,
            "rows_total": int(len(data)),
            "rows_used_for_training": int(len(train_df)),
            "rows_train": 0,
            "rows_test": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "model_path": "",
            "status": "skipped_not_enough_rows",
        }

    noise_seed = _stable_file_seed(args.random_state, csv_path.name)
    if not args.drop_next_activity:
        train_df = inject_next_activity_noise(train_df, args.next_activity_noise_rate, noise_seed)

    train_part, test_part = temporal_split(train_df, args.test_size)
    train_part = limit_train_cases(train_part, args.max_train_cases)
    if train_part.empty or test_part.empty:
        return {
            "file": csv_path.name,
            "rows_total": int(len(data)),
            "rows_used_for_training": int(len(train_df)),
            "rows_train": int(len(train_part)),
            "rows_test": int(len(test_part)),
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "model_path": "",
            "status": "skipped_invalid_temporal_split",
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
    next_activity_cardinality = len(category_maps["next_activity"])
    max_prefix_len = int(train_fit_part["prefix_length"].max()) if not train_fit_part.empty else 0

    numeric_imputer, numeric_scaler = fit_numeric_transformer(train_fit_part, numeric_columns)

    X_fit_num_np = transform_numeric(train_fit_part, numeric_columns, numeric_imputer, numeric_scaler)
    X_fit_activity_np, X_fit_prefix_np, X_fit_next_activity_np, X_fit_prefix_len_np = encode_categories(
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
    X_val_activity_np, X_val_prefix_np, X_val_next_activity_np, X_val_prefix_len_np = (
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
            np.empty((0,), dtype=np.int64),
        )
    )
    X_train_num_np = transform_numeric(X_train, numeric_columns, numeric_imputer, numeric_scaler)
    X_train_activity_np, X_train_prefix_np, X_train_next_activity_np, X_train_prefix_len_np = encode_categories(
        X_train,
        categorical_columns,
        category_maps,
        max_prefix_len=max_prefix_len,
    )
    X_test_num_np = transform_numeric(X_test, numeric_columns, numeric_imputer, numeric_scaler)
    X_test_activity_np, X_test_prefix_np, X_test_next_activity_np, X_test_prefix_len_np = encode_categories(
        X_test,
        categorical_columns,
        category_maps,
        max_prefix_len=max_prefix_len,
    )

    y_fit_log = np.log1p(np.clip(train_fit_part[target_col].to_numpy(dtype=float), a_min=0.0, a_max=None))
    y_val_log = (
        np.log1p(np.clip(val_fit_part[target_col].to_numpy(dtype=float), a_min=0.0, a_max=None))
        if not val_fit_part.empty
        else np.array([], dtype=float)
    )
    train_target_nonneg = np.clip(train_fit_part[target_col].to_numpy(dtype=float), a_min=0.0, a_max=None)
    max_train_target = float(np.max(train_target_nonneg)) if len(train_target_nonneg) > 0 else 0.0
    # Keep inverse-transform clipping near observed target scale to prevent extreme outlier predictions.
    log_pred_clip_max = min(20.0, np.log1p(max_train_target) + 0.5)

    model, history, best_epoch, stopped_early = train_torch_model(
        X_fit_num_np,
        X_fit_activity_np,
        X_fit_prefix_np,
        X_fit_prefix_len_np,
        X_fit_next_activity_np,
        y_fit_log,
        X_val_num_np,
        X_val_activity_np,
        X_val_prefix_np,
        X_val_prefix_len_np,
        X_val_next_activity_np,
        y_val_log,
        args,
        activity_cardinality,
        next_activity_cardinality,
        prefix_cardinality,
    )

    y_pred_test = invert_log1p_predictions(
        predict_torch_embeddings(
            model,
            X_test_num_np,
            X_test_activity_np,
            X_test_prefix_np,
            X_test_prefix_len_np,
            X_test_next_activity_np,
            batch_size=args.batch_size,
        ),
        log_clip_max=log_pred_clip_max,
    )
    y_pred_test = np.clip(y_pred_test, a_min=0.0, a_max=None)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    y_pred_train = invert_log1p_predictions(
        predict_torch_embeddings(
            model,
            X_train_num_np,
            X_train_activity_np,
            X_train_prefix_np,
            X_train_prefix_len_np,
            X_train_next_activity_np,
            batch_size=args.batch_size,
        ),
        log_clip_max=log_pred_clip_max,
    )
    y_pred_train = np.clip(y_pred_train, a_min=0.0, a_max=None)

    train_part = train_part.copy()
    test_part = test_part.copy()
    train_part["prediction_next_time"] = y_pred_train
    test_part["prediction_next_time"] = y_pred_test
    train_part["split"] = "train"
    test_part["split"] = "test"

    prediction_rows = pd.concat([train_part, test_part], ignore_index=True)
    prediction_rows = prediction_rows.sort_values("__row_id", kind="mergesort")

    input_cols = list(raw_data.columns)
    raw_with_row_id = raw_data.copy()
    raw_with_row_id["__row_id"] = raw_with_row_id.index

    prediction_export = raw_with_row_id.merge(
        prediction_rows[["__row_id", "split", "prediction_next_time"]],
        on="__row_id",
        how="inner",
    )
    prediction_export["actual_next_time"] = pd.to_numeric(prediction_export["next_time"], errors="coerce")
    prediction_export["prediction_error"] = (
        prediction_export["actual_next_time"] - prediction_export["prediction_next_time"]
    )
    prediction_export = prediction_export[
        input_cols + ["split", "actual_next_time", "prediction_next_time", "prediction_error"]
    ]

    models_dir = args.result_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{csv_path.stem}_mlp_next_time.pt"
    checkpoint = {
        "model_type": "pytorch_mlp",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "numeric_dim": int(X_fit_num_np.shape[1]),
        "activity_cardinality": int(activity_cardinality),
        "prefix_cardinality": int(prefix_cardinality),
        "next_activity_cardinality": int(next_activity_cardinality),
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
        "target_transform": "log1p",
        "log_pred_clip_max": float(log_pred_clip_max),
        "embedding_dim": int(args.embedding_dim),
        "gru_hidden_dim": int(args.gru_hidden_dim),
        "gru_num_layers": int(args.gru_num_layers),
        "max_prefix_len": int(max_prefix_len),
        "use_next_activity": bool(not args.drop_next_activity),
    }
    torch.save(checkpoint, model_path)

    predictions_dir = args.result_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = predictions_dir / f"{csv_path.stem}_prediction.csv"
    prediction_export.to_csv(prediction_path, index=False)

    return {
        "file": csv_path.name,
        "rows_total": int(len(data)),
        "rows_used_for_training": int(len(train_df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "max_train_cases": int(args.max_train_cases),
        "train_cases_used": int(train_part["case_index"].nunique()) if "case_index" in train_part.columns else np.nan,
        "target_transform": "log1p",
        "train_epochs": int(len(history["train_loss"])),
        "best_epoch": int(best_epoch),
        "stopped_early": bool(stopped_early),
        "train_loss_last": float(history["train_loss"][-1]) if history["train_loss"] else np.nan,
        "val_loss_last": float(history["val_loss"][-1]) if history["val_loss"] else np.nan,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
        "status": "trained",
        "use_next_activity": bool(not args.drop_next_activity),
    }


def main() -> None:
    args = parse_args()
    files = list_csv_files(args.data_dir)
    if args.dataset:
        files = [p for p in files if p.name == args.dataset]
        if not files:
            raise FileNotFoundError(f"Dataset not found in {args.data_dir}: {args.dataset}")

    args.result_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    for path in files:
        try:
            row = train_one_file(path, args)
        except Exception as exc:  # noqa: BLE001
            row = {
                "file": path.name,
                "rows_total": np.nan,
                "rows_used_for_training": np.nan,
                "rows_train": np.nan,
                "rows_test": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "model_path": "",
                "status": f"failed: {exc}",
            }
        all_rows.append(row)
        print(f"{path.name}: {row['status']}")

    metrics_df = pd.DataFrame(all_rows)
    report_path = args.result_dir / "mlp_metrics_all.csv"
    metrics_df.to_csv(report_path, index=False)

    trained_count = int((metrics_df["status"] == "trained").sum())
    print(f"Trained models: {trained_count}/{len(files)}")
    print(f"Saved aggregate metrics: {report_path}")


if __name__ == "__main__":
    main()
