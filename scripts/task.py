"""Task-specific logic: dataset loading, preprocessing, and model factory."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

try:
    from datasets import Dataset as HFDataset
except ImportError:
    HFDataset = None

try:
    from flwr_datasets.partitioner import NaturalIdPartitioner
except ImportError:
    NaturalIdPartitioner = None

import mlp

logger = logging.getLogger(__name__)


@dataclass
class EncodedSplit:
    """Encoded training/test split."""

    X_num: np.ndarray
    X_activity: np.ndarray
    X_prefix: np.ndarray
    X_prefix_len: np.ndarray
    X_next_activity: np.ndarray
    y_raw: np.ndarray
    y_log: np.ndarray


@dataclass
class ClientData:
    """Per-client data for federated training."""

    cid: str
    n_examples: int
    split: EncodedSplit


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Flower federated MLP baseline")

    parser.add_argument("--data-dir", type=Path, default=mlp.DEFAULT_DATA_DIR)
    parser.add_argument("--dataset", type=str, default="BPI12.csv")
    parser.add_argument("--result-dir", type=Path, default=Path("results/flwr_mlp"))

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--max-train-cases", type=int, default=0)

    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)

    parser.add_argument("--hidden-layers", type=str, default="512,256,128")
    parser.add_argument("--embedding-dim", type=int, default=mlp.DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--gru-hidden-dim", type=int, default=16)
    parser.add_argument("--gru-num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-batch-norm", action="store_true")

    parser.set_defaults(drop_next_activity=True)
    parser.add_argument("--drop-next-activity", dest="drop_next_activity", action="store_true")
    parser.add_argument("--use-next-activity", dest="drop_next_activity", action="store_false")

    parser.add_argument("--next-activity-noise-rate", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--central-pretrain-fraction", type=float, default=0.0)
    parser.add_argument("--central-pretrain-epochs", type=int, default=0)
    parser.add_argument("--central-pretrain-consume-fraction", action="store_true")

    parser.add_argument("--min-client-samples", type=int, default=128)
    parser.add_argument("--min-available-clients", type=int, default=2)
    parser.add_argument("--fraction-fit", type=float, default=1.0)
    parser.add_argument("--fraction-evaluate", type=float, default=1.0)
    parser.add_argument("--proximal-mu", type=float, default=0.01)
    parser.add_argument(
        "--client-freeze-embeddings",
        "--client-freeze-encoder",
        action="store_true",
        help="Freeze embeddings during client-side local training (GRU stays trainable)",
    )
    parser.add_argument(
        "--client-trainable-hidden-layers",
        type=int,
        default=-1,
        help="Number of final hidden MLP blocks to train on clients (-1 = all blocks)",
    )
    parser.add_argument(
        "--client-validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of each client's local split reserved for validation during best-round selection",
    )
    parser.add_argument(
        "--client-send-best-so-far",
        action="store_true",
        help="If enabled, client sends best-so-far local parameters (by selected metric) instead of latest round parameters",
    )
    parser.add_argument(
        "--client-best-metric",
        choices=["mae", "rmse"],
        default="mae",
        help="Metric used to choose client best-so-far parameters when --client-send-best-so-far is enabled",
    )
    parser.add_argument("--log-level", type=str, default="INFO")

    return parser.parse_args()


def _to_float_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy array to float32 tensor on device."""
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def build_encoded_split(
    df: pd.DataFrame,
    numeric_columns: List[str],
    categorical_columns: List[str],
    category_maps: Dict[str, Dict[str, int]],
    max_prefix_len: int,
    imputer: Any,
    scaler: Any,
) -> EncodedSplit:
    """Build an encoded data split from DataFrame."""
    X_num = mlp.transform_numeric(df, numeric_columns, imputer, scaler)
    X_activity, X_prefix, X_next_activity, X_prefix_len = mlp.encode_categories(
        df,
        categorical_columns,
        category_maps,
        max_prefix_len=max_prefix_len,
    )
    y_raw = np.clip(df["next_time"].to_numpy(dtype=float), a_min=0.0, a_max=None)
    y_log = np.log1p(y_raw)
    return EncodedSplit(
        X_num=X_num,
        X_activity=X_activity,
        X_prefix=X_prefix,
        X_prefix_len=X_prefix_len,
        X_next_activity=X_next_activity,
        y_raw=y_raw,
        y_log=y_log,
    )


def subset_encoded_split(split: EncodedSplit, idx: np.ndarray) -> EncodedSplit:
    """Extract a subset of an encoded split by indices."""
    return EncodedSplit(
        X_num=split.X_num[idx],
        X_activity=split.X_activity[idx],
        X_prefix=split.X_prefix[idx],
        X_prefix_len=split.X_prefix_len[idx],
        X_next_activity=split.X_next_activity[idx],
        y_raw=split.y_raw[idx],
        y_log=split.y_log[idx],
    )


def split_encoded_split(split: EncodedSplit, val_fraction: float) -> tuple[EncodedSplit, EncodedSplit]:
    """Split an encoded split into train and validation parts.

    The split is deterministic and keeps the earlier rows for training.
    """
    n_rows = len(split.y_raw)
    if n_rows <= 1 or val_fraction <= 0:
        empty_idx = np.array([], dtype=int)
        return split, subset_encoded_split(split, empty_idx)

    n_val = max(1, int(round(n_rows * val_fraction)))
    if n_val >= n_rows:
        n_val = n_rows - 1

    split_idx = n_rows - n_val
    train_idx = np.arange(0, split_idx)
    val_idx = np.arange(split_idx, n_rows)
    return subset_encoded_split(split, train_idx), subset_encoded_split(split, val_idx)


def build_partition_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for partitioner."""
    partition_df = df.reset_index(drop=True).copy()
    partition_df["activity"] = partition_df["activity"].astype(str)
    return partition_df


def build_activity_clients_with_partitioner(
    train_fl_df: pd.DataFrame,
    numeric_columns: List[str],
    category_maps: Dict[str, Dict[str, int]],
    max_prefix_len: int,
    imputer: Any,
    scaler: Any,
    min_client_samples: int,
) -> Dict[str, ClientData]:
    """Build clients using NaturalIdPartitioner (activity-based)."""
    if HFDataset is None or NaturalIdPartitioner is None:
        raise ImportError("flwr_datasets and datasets are required for NaturalIdPartitioner")

    logger.info("Building clients with NaturalIdPartitioner (by activity)")
    hf_dataset = HFDataset.from_pandas(build_partition_df(train_fl_df), preserve_index=False)
    partitioner = NaturalIdPartitioner(partition_by="activity")
    partitioner.dataset = hf_dataset

    clients: Dict[str, ClientData] = {}
    for partition_id in range(partitioner.num_partitions):
        partition = partitioner.load_partition(partition_id)
        partition_df = partition.to_pandas()
        if len(partition_df) < min_client_samples:
            continue

        natural_id = str(partitioner.partition_id_to_natural_id[partition_id])
        split = build_encoded_split(
            partition_df,
            numeric_columns,
            [],
            category_maps,
            max_prefix_len=max_prefix_len,
            imputer=imputer,
            scaler=scaler,
        )
        clients[natural_id] = ClientData(cid=natural_id, n_examples=len(partition_df), split=split)
        logger.debug(f"Client {natural_id}: {len(partition_df)} samples")

    return clients


def build_activity_clients_manually(
    train_fl_df: pd.DataFrame,
    train_fl_encoded: EncodedSplit,
    min_client_samples: int,
) -> Dict[str, ClientData]:
    """Build clients by manually grouping by activity."""
    logger.info("Building clients manually (by activity)")
    activity_values = train_fl_df["activity"].astype(str).to_numpy()
    unique_activities = sorted(set(activity_values.tolist()))

    clients: Dict[str, ClientData] = {}
    for activity_name in unique_activities:
        idx = np.where(activity_values == activity_name)[0]
        if len(idx) < min_client_samples:
            continue
        split = subset_encoded_split(train_fl_encoded, idx)
        clients[str(activity_name)] = ClientData(cid=str(activity_name), n_examples=len(idx), split=split)
        logger.debug(f"Client {activity_name}: {len(idx)} samples")

    return clients


def train_local_epoch(
    model: mlp.TorchMLP,
    split: EncodedSplit,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    huber_delta: float,
    device: torch.device,
    client_id: str = "central",
    round_num: int = 0,
    freeze_embeddings: bool = False,
    trainable_hidden_layers: int = -1,
    global_params: List[np.ndarray] | None = None,
    proximal_mu: float = 0.0,
) -> List[float]:
    """Train model for local_epochs on a data split.
    
    Returns:
        List of average losses per epoch
    """
    criterion = nn.HuberLoss(delta=huber_delta)

    dataset = TensorDataset(
        _to_float_tensor(split.X_num, device),
        torch.as_tensor(split.X_activity, dtype=torch.long, device=device),
        torch.as_tensor(split.X_prefix, dtype=torch.long, device=device),
        torch.as_tensor(split.X_prefix_len, dtype=torch.long, device=device),
        torch.as_tensor(split.X_next_activity, dtype=torch.long, device=device),
        _to_float_tensor(split.y_log, device),
    )
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=True)

    model.train()
    if hasattr(model, "configure_client_training"):
        model.configure_client_training(
            freeze_embeddings=freeze_embeddings,
            trainable_hidden_layers=trainable_hidden_layers,
        )

    all_parameters = list(model.parameters())
    trainable_parameters = [parameter for parameter in all_parameters if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after applying the client training policy")

    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    global_params_tensors: List[torch.Tensor] | None = None
    if global_params is not None and proximal_mu > 0:
        global_params_tensors = [
            torch.as_tensor(w, dtype=param.dtype, device=device)
            for w, param in zip(global_params, all_parameters)
        ]
    epoch_losses = []
    
    for epoch_idx in range(max(1, local_epochs)):
        epoch_loss = 0.0
        batch_count = 0
        for xb_num, xb_activity, xb_prefix, xb_prefix_len, xb_next_activity, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb_num, xb_activity, xb_prefix, xb_prefix_len, xb_next_activity)
            loss = criterion(pred, yb)
            if global_params_tensors is not None:
                prox_term = 0.0
                for w, w_global in zip(all_parameters, global_params_tensors):
                    prox_term += ((w - w_global) ** 2).sum()
                loss = loss + proximal_mu * prox_term
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        epoch_losses.append(avg_epoch_loss)
        logger.debug(f"Client {client_id} round {round_num} epoch {epoch_idx + 1}/{local_epochs}: loss={avg_epoch_loss:.4f}")
    
    return epoch_losses


@torch.no_grad()
def evaluate_split(
    model: mlp.TorchMLP,
    split: EncodedSplit,
    batch_size: int,
    log_pred_clip_max: float,
    device: torch.device,
    phase: str = "evaluation",
) -> tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model on a data split, returning metrics and predictions.
    
    Returns:
        (metrics_dict, y_true, y_pred)
    """
    model.eval()

    dataset = TensorDataset(
        torch.as_tensor(split.X_num, dtype=torch.float32),
        torch.as_tensor(split.X_activity, dtype=torch.long),
        torch.as_tensor(split.X_prefix, dtype=torch.long),
        torch.as_tensor(split.X_prefix_len, dtype=torch.long),
        torch.as_tensor(split.X_next_activity, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False)

    preds: List[np.ndarray] = []
    for xb_num, xb_activity, xb_prefix, xb_prefix_len, xb_next_activity in loader:
        xb_num = xb_num.to(device)
        xb_activity = xb_activity.to(device)
        xb_prefix = xb_prefix.to(device)
        xb_prefix_len = xb_prefix_len.to(device)
        xb_next_activity = xb_next_activity.to(device)
        p = model(xb_num, xb_activity, xb_prefix, xb_prefix_len, xb_next_activity).cpu().numpy()
        preds.append(p)

    y_pred_log = np.concatenate(preds, axis=0) if preds else np.array([], dtype=float)
    y_pred = mlp.invert_log1p_predictions(y_pred_log, log_clip_max=log_pred_clip_max)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    y_true = split.y_raw
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    logger.debug(f"{phase}: mae={mae:.4f}, rmse={rmse:.4f}, r2={r2:.4f}")
    
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}, y_true, y_pred


def create_model_factory(
    train_encoded: EncodedSplit,
    category_maps: Dict[str, Dict[str, int]],
    args: argparse.Namespace,
) -> Callable[[], mlp.TorchMLP]:
    """Create a model factory function."""
    activity_cardinality = len(category_maps["activity"])
    prefix_cardinality = len(category_maps["prefix_sequence"])
    next_activity_cardinality = len(category_maps["next_activity"])
    hidden_layers = mlp.parse_hidden_layers(args.hidden_layers)

    def factory() -> mlp.TorchMLP:
        return mlp.TorchMLP(
            numeric_dim=train_encoded.X_num.shape[1],
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
        )

    return factory


def setup_data(args: argparse.Namespace) -> tuple:
    """Load and preprocess data, return all task-specific data structures.
    
    Returns:
        (clients, train_encoded, test_encoded, model_factory, log_pred_clip_max)
    """
    data_path = args.data_dir / args.dataset
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    logger.info(f"Loading dataset from {data_path}")
    raw_data = mlp.load_single_dataset(data_path, args.max_rows_per_file)
    mlp.validate_required_columns(raw_data)

    data = mlp.coerce_types(raw_data)
    train_df = data.dropna(subset=["next_time"]).copy()
    if not args.drop_next_activity:
        noise_seed = args.random_state + sum(args.dataset.encode("utf-8"))
        train_df = mlp.inject_next_activity_noise(train_df, args.next_activity_noise_rate, noise_seed)

    train_part, test_part = mlp.temporal_split(train_df, args.test_size)
    train_part = mlp.limit_train_cases(train_part, args.max_train_cases)
    if train_part.empty or test_part.empty:
        raise ValueError("Temporal split failed: train/test empty")

    logger.info(f"Train split: {len(train_part)} rows, Test split: {len(test_part)} rows")

    categorical_columns = mlp.get_categorical_columns()
    numeric_columns = mlp.get_numeric_columns()

    category_maps = mlp.build_category_maps(train_part, categorical_columns)
    max_prefix_len = int(train_part["prefix_length"].max()) if not train_part.empty else 0
    numeric_imputer, numeric_scaler = mlp.fit_numeric_transformer(train_part, numeric_columns)

    train_encoded = build_encoded_split(
        train_part,
        numeric_columns,
        categorical_columns,
        category_maps,
        max_prefix_len=max_prefix_len,
        imputer=numeric_imputer,
        scaler=numeric_scaler,
    )
    test_encoded = build_encoded_split(
        test_part,
        numeric_columns,
        categorical_columns,
        category_maps,
        max_prefix_len=max_prefix_len,
        imputer=numeric_imputer,
        scaler=numeric_scaler,
    )

    max_train_target = float(np.max(train_encoded.y_raw)) if len(train_encoded.y_raw) > 0 else 0.0
    log_pred_clip_max = min(20.0, np.log1p(max_train_target) + 0.5)

    # Optional central pretraining on first N% temporal rows.
    n_pre = int(round(len(train_part) * args.central_pretrain_fraction))
    n_pre = max(0, min(len(train_part), n_pre))

    if args.central_pretrain_consume_fraction:
        train_fl_start = n_pre
    else:
        train_fl_start = 0

    train_fl_df = train_part.iloc[train_fl_start:].copy()
    train_fl_encoded = build_encoded_split(
        train_fl_df,
        numeric_columns,
        categorical_columns,
        category_maps,
        max_prefix_len=max_prefix_len,
        imputer=numeric_imputer,
        scaler=numeric_scaler,
    )

    # Build clients by activity value
    try:
        clients = build_activity_clients_with_partitioner(
            train_fl_df=train_fl_df,
            numeric_columns=numeric_columns,
            category_maps=category_maps,
            max_prefix_len=max_prefix_len,
            imputer=numeric_imputer,
            scaler=numeric_scaler,
            min_client_samples=args.min_client_samples,
        )
    except Exception as exc:
        logger.warning(
            f"NaturalIdPartitioner unavailable or failed ({exc}); falling back to manual activity grouping."
        )
        clients = build_activity_clients_manually(
            train_fl_df=train_fl_df,
            train_fl_encoded=train_fl_encoded,
            min_client_samples=args.min_client_samples,
        )

    if len(clients) < args.min_available_clients:
        raise ValueError(
            f"Not enough clients after filtering. Have {len(clients)}, need {args.min_available_clients}."
        )

    logger.info(f"Created {len(clients)} clients")

    # Central pretraining
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_factory = create_model_factory(train_encoded, category_maps, args)
    global_model = model_factory().to(device)

    if n_pre > 0 and args.central_pretrain_epochs > 0:
        logger.info(f"Central pretraining on {n_pre} samples for {args.central_pretrain_epochs} epochs")
        pre_df = train_part.iloc[:n_pre].copy()
        pre_encoded = build_encoded_split(
            pre_df,
            numeric_columns,
            categorical_columns,
            category_maps,
            max_prefix_len=max_prefix_len,
            imputer=numeric_imputer,
            scaler=numeric_scaler,
        )
        epoch_losses = train_local_epoch(
            model=global_model,
            split=pre_encoded,
            local_epochs=args.central_pretrain_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            huber_delta=args.huber_delta,
            device=device,
            client_id="central-pretraining",
            round_num=0,
        )
        logger.info(f"Central pretraining completed with final epoch loss: {epoch_losses[-1] if epoch_losses else 'N/A':.4f}")

    return clients, train_encoded, test_encoded, model_factory, log_pred_clip_max, global_model, device, (n_pre, train_part, test_part)
