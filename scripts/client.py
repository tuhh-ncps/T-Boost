"""Client-side federated learning logic."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List

import flwr as fl
import numpy as np
import torch

from logging_utils import get_logger, initialize_logger
from task import ClientData, EncodedSplit, evaluate_split, split_encoded_split, train_local_epoch

logger = logging.getLogger(__name__)


def get_model_parameters(model) -> List[np.ndarray]:
    """Extract model parameters as numpy arrays."""
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_model_parameters(model, parameters: List[np.ndarray]) -> None:
    """Load model parameters from numpy arrays."""
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {k: torch.as_tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(new_state, strict=True)


class ActivityClient(fl.client.NumPyClient):
    """Federated learning client for activity-based models."""

    def __init__(
        self,
        client_data: ClientData,
        model_factory,
        device: torch.device,
        args,
        log_pred_clip_max: float,
        result_dir: Path,
        round_num: int = 0,
    ) -> None:
        self.client_data = client_data
        self.model_factory = model_factory
        self.device = device
        self.args = args
        self.log_pred_clip_max = log_pred_clip_max
        self.round_num = round_num
        self.result_dir = Path(result_dir)
        self.best_val_metric: float | None = None
        self.best_val_params: List[np.ndarray] | None = None
        
        # Initialize logger in this process if not already done
        try:
            get_logger()
        except RuntimeError:
            initialize_logger(self.result_dir)

        logger.info(f"Initialized client {client_data.cid} with {client_data.n_examples} samples")

    def _infer_round_from_history(self, phase: str) -> int:
        """Infer next round index for this client from existing metric logs."""
        if phase == "fit":
            history_file = self.result_dir / "epoch_metrics.csv"
            phase_filter = None
        else:
            history_file = self.result_dir / "client_metrics.csv"
            phase_filter = "evaluate"

        if not history_file.exists():
            return 1

        max_round = 0
        try:
            with open(history_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("client_id") != self.client_data.cid:
                        continue
                    if phase_filter is not None and row.get("phase") != phase_filter:
                        continue
                    try:
                        round_val = int(float(row.get("round", "0")))
                    except (TypeError, ValueError):
                        continue
                    max_round = max(max_round, round_val)
        except OSError:
            return 1

        return max_round + 1

    def _update_round_from_config(self, config, phase: str) -> None:
        """Update local round label from Flower config; fallback to local counter."""
        if not isinstance(config, dict):
            self.round_num = self._infer_round_from_history(phase)
            return
        logger.debug("Client %s: %s config keys=%s", self.client_data.cid, phase, sorted(config.keys()))

        server_round = None
        for key in ("server_round", "current_round", "round", "rnd"):
            if key in config:
                server_round = config.get(key)
                break
        if server_round is None:
            self.round_num = self._infer_round_from_history(phase)
            return
        try:
            self.round_num = int(server_round)
        except (TypeError, ValueError):
            logger.debug("Client %s: invalid server_round in config: %r", self.client_data.cid, server_round)
            self.round_num = self._infer_round_from_history(phase)

    def get_parameters(self, config) -> List[np.ndarray]:
        """Get client's local model parameters."""
        logger.debug(f"Client {self.client_data.cid}: get_parameters() called")
        model = self.model_factory().to(self.device)
        return get_model_parameters(model)

    def fit(self, parameters: List[np.ndarray], config) -> tuple:
        """Train client's local model with received global parameters."""
        self._update_round_from_config(config, phase="fit")
        logger.info(
            f"Client {self.client_data.cid}: round {self.round_num} training starting "
            f"(local_epochs={self.args.local_epochs})"
        )
        
        model = self.model_factory().to(self.device)
        set_model_parameters(model, parameters)

        epoch_losses = train_local_epoch(
            model=model,
            split=self.client_data.split,
            local_epochs=self.args.local_epochs,
            batch_size=self.args.batch_size,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            huber_delta=self.args.huber_delta,
            device=self.device,
            client_id=self.client_data.cid,
            round_num=self.round_num,
            freeze_embeddings=self.args.client_freeze_embeddings,
            trainable_hidden_layers=self.args.client_trainable_hidden_layers,
            global_params=parameters,
            proximal_mu=self.args.proximal_mu,
        )

        # Log epoch metrics
        try:
            fl_logger = get_logger()
            for epoch_idx, loss in enumerate(epoch_losses):
                fl_logger.log_epoch_metrics(
                    round_num=self.round_num,
                    client_id=self.client_data.cid,
                    epoch=epoch_idx + 1,
                    loss=loss,
                    avg_batch_loss=loss,
                )
        except RuntimeError:
            pass  # Logger not initialized yet
        
        updated_params = get_model_parameters(model)
        send_params = updated_params

        if self.args.client_send_best_so_far:
            train_split, val_split = split_encoded_split(
                self.client_data.split,
                self.args.client_validation_fraction,
            )
            selection_split = val_split if len(val_split.y_raw) > 0 else train_split
            fit_metrics, _, _ = evaluate_split(
                model=model,
                split=selection_split,
                batch_size=self.args.batch_size,
                log_pred_clip_max=self.log_pred_clip_max,
                device=self.device,
                phase=f"client-{self.client_data.cid}-fit-selection",
            )
            metric_name = self.args.client_best_metric
            current_metric = float(fit_metrics[metric_name])

            is_first = self.best_val_metric is None or self.best_val_params is None
            is_better = not np.isnan(current_metric) and (
                is_first or current_metric < float(self.best_val_metric)
            )

            if is_better:
                self.best_val_metric = current_metric
                self.best_val_params = [np.array(p, copy=True) for p in updated_params]
                logger.info(
                    "Client %s: round %s new best validation %s=%.6f; sending current params",
                    self.client_data.cid,
                    self.round_num,
                    metric_name,
                    current_metric,
                )
            elif self.best_val_params is not None:
                send_params = [np.array(p, copy=True) for p in self.best_val_params]
                logger.info(
                    "Client %s: round %s validation %s=%.6f worse than best %.6f; sending best-so-far params",
                    self.client_data.cid,
                    self.round_num,
                    metric_name,
                    current_metric,
                    float(self.best_val_metric),
                )

        logger.info(f"Client {self.client_data.cid}: round {self.round_num} training completed")

        return send_params, self.client_data.n_examples, {}

    def evaluate(self, parameters: List[np.ndarray], config) -> tuple:
        """Evaluate client's model with received global parameters."""
        self._update_round_from_config(config, phase="evaluate")
        logger.info(f"Client {self.client_data.cid}: round {self.round_num} evaluation starting")
        
        model = self.model_factory().to(self.device)
        set_model_parameters(model, parameters)
        
        metrics, y_true, y_pred = evaluate_split(
            model=model,
            split=self.client_data.split,
            batch_size=self.args.batch_size,
            log_pred_clip_max=self.log_pred_clip_max,
            device=self.device,
            phase=f"client-{self.client_data.cid}",
        )
        
        logger.info(
            f"Client {self.client_data.cid}: round {self.round_num} evaluation metrics - "
            f"mae={metrics['mae']:.4f}, rmse={metrics['rmse']:.4f}, r2={metrics['r2']:.4f}"
        )
        
        # Log client metrics
        try:
            fl_logger = get_logger()
            fl_logger.log_client_metrics(
                round_num=self.round_num,
                client_id=self.client_data.cid,
                phase="evaluate",
                n_samples=self.client_data.n_examples,
                metrics=metrics,
                y_true=y_true,
                y_pred=y_pred,
            )
        except RuntimeError:
            pass  # Logger not initialized yet
        
        # Flower expects (loss, num_examples, metrics); use rmse as optimization loss proxy.
        return metrics["rmse"], self.client_data.n_examples, metrics


def client_fn(cid: str, clients_dict, model_factory, device, args, log_pred_clip_max, result_dir):
    """Factory function to create a client instance.
    
    Args:
        cid: Client ID (string representation of numeric ID)
        clients_dict: Dictionary of client data
        model_factory: Factory function for creating models
        device: Torch device
        args: Command-line arguments
        log_pred_clip_max: Log prediction clipping maximum
        result_dir: Results directory for logging
    
    Returns:
        ActivityClient instance converted to Flower Client
    """
    # Map numeric CID to actual client key
    client_id = int(cid)
    client_keys = sorted(clients_dict.keys())
    client_key = client_keys[client_id]
    client_data = clients_dict[client_key]
    
    logger.info(f"Creating client instance {client_id} (activity={client_key})")
    
    return ActivityClient(
        client_data=client_data,
        model_factory=model_factory,
        device=device,
        args=args,
        log_pred_clip_max=log_pred_clip_max,
        result_dir=result_dir,
        round_num=0,  # Will be updated server-side
    ).to_client()
