"""Server-side federated learning logic."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import flwr as fl
import numpy as np
import torch

from logging_utils import get_logger
from task import evaluate_split

logger = logging.getLogger(__name__)


def create_evaluate_fn(
    model_factory,
    test_encoded,
    batch_size: int,
    log_pred_clip_max: float,
    device: torch.device,
):
    """Create a server-side evaluation function.
    
    Args:
        model_factory: Factory function to create model instances
        test_encoded: Encoded test split
        batch_size: Batch size for evaluation
        log_pred_clip_max: Log prediction clipping max
        device: Torch device (cpu/cuda)
    
    Returns:
        evaluate_fn callable for use in strategy
    """

    def evaluate_fn(server_round: int, parameters, config):
        """Server-side evaluation function called each round."""
        logger.info(f"[SERVER] Round {server_round}: Evaluating global model on test set")
        
        model = model_factory().to(device)
        
        # FedAvg/FedProx pass ndarray parameters to evaluate_fn in newer Flower versions.
        # Keep compatibility with both ndarray-list and Parameters payloads.
        if isinstance(parameters, list):
            model_params = parameters
        else:
            model_params = fl.common.parameters_to_ndarrays(parameters)
        
        # Set model parameters
        state_dict = model.state_dict()
        keys = list(state_dict.keys())
        new_state = {k: torch.as_tensor(v) for k, v in zip(keys, model_params)}
        model.load_state_dict(new_state, strict=True)
        
        metrics, y_true, y_pred = evaluate_split(
            model=model,
            split=test_encoded,
            batch_size=batch_size,
            log_pred_clip_max=log_pred_clip_max,
            device=device,
            phase=f"server-round-{server_round}",
        )
        
        logger.info(
            f"[SERVER] Round {server_round}: Global model metrics - "
            f"mae={metrics['mae']:.4f}, rmse={metrics['rmse']:.4f}, r2={metrics['r2']:.4f}"
        )
        
        # Log server metrics
        try:
            fl_logger = get_logger()
            fl_logger.log_server_metrics(
                round_num=server_round,
                phase="evaluation",
                metrics=metrics,
                y_true=y_true,
                y_pred=y_pred,
            )
        except RuntimeError:
            pass  # Logger not initialized yet
        
        return metrics["rmse"], metrics

    return evaluate_fn


def create_strategy(
    initial_parameters,
    evaluate_fn,
    num_clients: int,
    min_available_clients: int,
    fraction_fit: float,
    fraction_evaluate: float,
    proximal_mu: float,
):
    """Create FedProx strategy with custom settings.
    
    Args:
        initial_parameters: Initial model parameters
        evaluate_fn: Server-side evaluation function
        num_clients: Total number of clients
        min_available_clients: Minimum clients required
        fraction_fit: Fraction of clients to use for training
        fraction_evaluate: Fraction of clients to use for evaluation
        proximal_mu: FedProx proximal regularization coefficient
    
    Returns:
        FedProx strategy instance
    """
    min_fit = min(max(2, int(np.ceil(fraction_fit * num_clients))), num_clients)
    min_eval = min(max(2, int(np.ceil(fraction_evaluate * num_clients))), num_clients)
    min_available = min(min_available_clients, num_clients)
    
    logger.info(
        f"[STRATEGY] Creating FedProx with: "
        f"total_clients={num_clients}, "
        f"min_fit={min_fit}, "
        f"min_evaluate={min_eval}, "
        f"min_available={min_available}, "
        f"proximal_mu={proximal_mu}"
    )

    def on_fit_config_fn(server_round: int) -> Dict[str, int]:
        return {"server_round": int(server_round)}

    def on_evaluate_config_fn(server_round: int) -> Dict[str, int]:
        return {"server_round": int(server_round)}
    
    strategy = fl.server.strategy.FedProx(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit,
        min_evaluate_clients=min_eval,
        min_available_clients=min_available,
        proximal_mu=proximal_mu,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
    )
    
    return strategy
