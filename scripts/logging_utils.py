"""Comprehensive logging utilities for federated learning training."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from mlp import compute_normalized_mae


class FederatedLearningLogger:
    """Centralized logger for federated learning experiments."""

    def __init__(self, result_dir: Path):
        """Initialize logger with result directory.
        
        Args:
            result_dir: Directory to save all logs and CSV files
        """
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.log_file = self.result_dir / "training.log"
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        
        # CSV files for structured logging
        self.server_metrics_file = self.result_dir / "server_metrics.csv"
        self.client_metrics_file = self.result_dir / "client_metrics.csv"
        self.epoch_metrics_file = self.result_dir / "epoch_metrics.csv"
        
        # CSV writers
        self.server_csv_writer = None
        self.client_csv_writer = None
        self.epoch_csv_writer = None
        
        self._init_csv_files()
    
    def add_logger(self, logger: logging.Logger) -> None:
        """Add file handler to a logger.
        
        Args:
            logger: Python logger instance to augment with file handler
        """
        logger.addHandler(self.file_handler)
    
    def _init_csv_files(self) -> None:
        """Initialize CSV files with headers."""
        # Server metrics: round-level global model evaluation
        with open(self.server_metrics_file, "w", newline="") as f:
            self.server_csv_writer = csv.DictWriter(
                f,
                fieldnames=[
                    "round",
                    "phase",
                    "mae",
                    "rmse",
                    "r2",
                    "normalized_mae",
                    "loss",
                ],
            )
            self.server_csv_writer.writeheader()
        
        # Client metrics: per-client, per-round results
        with open(self.client_metrics_file, "w", newline="") as f:
            self.client_csv_writer = csv.DictWriter(
                f,
                fieldnames=[
                    "round",
                    "client_id",
                    "phase",
                    "n_samples",
                    "mae",
                    "rmse",
                    "r2",
                    "normalized_mae",
                    "loss",
                ],
            )
            self.client_csv_writer.writeheader()
        
        # Epoch metrics: per-client training epoch details
        with open(self.epoch_metrics_file, "w", newline="") as f:
            self.epoch_csv_writer = csv.DictWriter(
                f,
                fieldnames=[
                    "round",
                    "client_id",
                    "epoch",
                    "loss",
                    "avg_batch_loss",
                ],
            )
            self.epoch_csv_writer.writeheader()
    
    def log_server_metrics(
        self,
        round_num: int,
        phase: str,
        metrics: Dict[str, float],
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
    ) -> None:
        """Log server-side evaluation metrics.
        
        Args:
            round_num: Round number
            phase: Phase name (e.g., "initial", "evaluation")
            metrics: Dict with mae, rmse, r2
            y_true: True values for normalized MAE calculation
            y_pred: Predicted values for normalized MAE calculation
        """
        normalized_mae = self._compute_normalized_mae(y_true, y_pred, metrics.get("mae"))
        
        row = {
            "round": round_num,
            "phase": phase,
            "mae": f"{metrics.get('mae', float('nan')):.4f}",
            "rmse": f"{metrics.get('rmse', float('nan')):.4f}",
            "r2": f"{metrics.get('r2', float('nan')):.4f}",
            "normalized_mae": f"{normalized_mae:.4f}" if normalized_mae is not None else "nan",
            "loss": f"{metrics.get('rmse', float('nan')):.4f}",  # RMSE as loss
        }
        
        with open(self.server_metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    def log_client_metrics(
        self,
        round_num: int,
        client_id: str,
        phase: str,
        n_samples: int,
        metrics: Dict[str, float],
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
    ) -> None:
        """Log per-client evaluation metrics.
        
        Args:
            round_num: Round number
            client_id: Client identifier
            phase: Phase name (e.g., "fit", "evaluate")
            n_samples: Number of samples processed
            metrics: Dict with mae, rmse, r2
            y_true: True values for normalized MAE calculation
            y_pred: Predicted values for normalized MAE calculation
        """
        normalized_mae = self._compute_normalized_mae(y_true, y_pred, metrics.get("mae"))
        
        row = {
            "round": round_num,
            "client_id": client_id,
            "phase": phase,
            "n_samples": n_samples,
            "mae": f"{metrics.get('mae', float('nan')):.4f}",
            "rmse": f"{metrics.get('rmse', float('nan')):.4f}",
            "r2": f"{metrics.get('r2', float('nan')):.4f}",
            "normalized_mae": f"{normalized_mae:.4f}" if normalized_mae is not None else "nan",
            "loss": f"{metrics.get('rmse', float('nan')):.4f}",
        }
        
        with open(self.client_metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    def log_epoch_metrics(
        self,
        round_num: int,
        client_id: str,
        epoch: int,
        loss: float,
        avg_batch_loss: float,
    ) -> None:
        """Log per-epoch training metrics.
        
        Args:
            round_num: Round number
            client_id: Client identifier
            epoch: Epoch number
            loss: Total epoch loss
            avg_batch_loss: Average batch loss for epoch
        """
        row = {
            "round": round_num,
            "client_id": client_id,
            "epoch": epoch,
            "loss": f"{loss:.4f}",
            "avg_batch_loss": f"{avg_batch_loss:.4f}",
        }
        
        with open(self.epoch_metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    @staticmethod
    def _compute_normalized_mae(
        y_true: Optional[np.ndarray],
        y_pred: Optional[np.ndarray],
        mae: Optional[float] = None,
    ) -> Optional[float]:
        """Compute normalized MAE (MAE / median of y_true). Uses shared function from mlp.py."""
        return compute_normalized_mae(y_true, y_pred, mae)
    
    def log_message(self, logger_name: str, level: str, message: str) -> None:
        """Log a message using the specified logger.
        
        Args:
            logger_name: Name of logger (e.g., "__main__", "client", "server")
            level: Log level (e.g., "INFO", "DEBUG", "WARNING")
            message: Message to log
        """
        logger = logging.getLogger(logger_name)
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, message)


# Global logger instance (will be set in main)
_global_logger: Optional[FederatedLearningLogger] = None


def get_logger() -> FederatedLearningLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        raise RuntimeError("Logger not initialized. Call initialize_logger() first.")
    return _global_logger


def initialize_logger(result_dir: Path) -> FederatedLearningLogger:
    """Initialize the global logger instance.
    
    Args:
        result_dir: Directory to save logs
    
    Returns:
        Initialized FederatedLearningLogger instance
    """
    global _global_logger
    _global_logger = FederatedLearningLogger(result_dir)
    
    # Add file handler to all loggers
    logging.getLogger("task").addHandler(_global_logger.file_handler)
    logging.getLogger("client").addHandler(_global_logger.file_handler)
    logging.getLogger("server").addHandler(_global_logger.file_handler)
    logging.getLogger("__main__").addHandler(_global_logger.file_handler)
    
    return _global_logger
