#!/usr/bin/env python3
"""Federated learning baseline for next_time prediction using Flower + TorchMLP.

Design:
- Clients are partitioned by activity with NaturalIdPartitioner when available.
- A central model can be pretrained on the first N% of temporal training rows.
- Federated rounds run FedProx over remaining client data.
- Evaluation is done on a held-out temporal test split.

Architecture:
- task.py: Data loading, preprocessing, model factory
- client.py: Client-side federated logic with modern Context API
- server.py: Server-side strategy and evaluation
- logging_utils.py: Comprehensive logging and metrics tracking
- flwr_federated_mlp.py: Main entry point and orchestration
"""

from __future__ import annotations

import logging

import flwr as fl
import pandas as pd
import torch

from client import client_fn, get_model_parameters
from logging_utils import initialize_logger
from server import create_evaluate_fn, create_strategy
from task import parse_args, setup_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main() -> None:
    """Main entry point for federated learning training."""
    args = parse_args()
    
    # Configure logging level
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    
    # Initialize file logging
    initialize_logger(args.result_dir)
    logger.info("=== Federated Learning Training Started ===")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Rounds: {args.rounds}, Local epochs: {args.local_epochs}")
    logger.info(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    logger.info(
        f"Client training policy: freeze_embeddings={args.client_freeze_embeddings}, "
        f"trainable_hidden_layers={args.client_trainable_hidden_layers}, "
        f"send_best_so_far={args.client_send_best_so_far}, "
        f"best_metric={args.client_best_metric}, "
        f"validation_fraction={args.client_validation_fraction}"
    )
    logger.info(f"Logs saved to: {args.result_dir}")
    
    # Create result directory
    args.result_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    logger.info("Setting up data...")
    clients, train_encoded, test_encoded, model_factory, log_pred_clip_max, global_model, device, metadata = setup_data(args)
    n_pre, train_part, test_part = metadata
    
    client_ids = sorted(clients.keys())
    logger.info(f"Initialized {len(client_ids)} clients")
    
    # Create initial parameters
    initial_parameters = fl.common.ndarrays_to_parameters(get_model_parameters(global_model))
    logger.info("Created initial model parameters")
    
    # Create server-side evaluation function
    logger.info("Creating evaluation function...")
    evaluate_fn = create_evaluate_fn(
        model_factory=model_factory,
        test_encoded=test_encoded,
        batch_size=args.batch_size,
        log_pred_clip_max=log_pred_clip_max,
        device=device,
    )
    
    # Create strategy
    logger.info("Creating FedProx strategy...")
    strategy = create_strategy(
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        num_clients=len(client_ids),
        min_available_clients=args.min_available_clients,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        proximal_mu=args.proximal_mu,
    )
    
    # Create client function with legacy API (compatible with simulation)
    def client_fn_wrapper(cid: str):
        """Wrapper to pass additional arguments to client_fn."""
        return client_fn(
            cid=cid,
            clients_dict=clients,
            model_factory=model_factory,
            device=device,
            args=args,
            log_pred_clip_max=log_pred_clip_max,
            result_dir=args.result_dir,
        )
    
    # Run simulation
    logger.info(f"Starting Flower simulation with {len(client_ids)} clients for {args.rounds} rounds...")
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn_wrapper,
            num_clients=len(client_ids),
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
        
        logger.info("Simulation completed successfully")
        
        # Extract final metrics from history
        def _last_metric(metric_name: str) -> float:
            pairs = history.metrics_centralized.get(metric_name, [])
            if not pairs:
                return float("nan")
            return float(pairs[-1][1])
        
        final_metrics = {
            "mae": _last_metric("mae"),
            "rmse": _last_metric("rmse"),
            "r2": _last_metric("r2"),
        }
        
        # Log round-by-round metrics
        logger.info("Centralized evaluation metrics by round:")
        for metric_name in ["mae", "rmse", "r2"]:
            pairs = history.metrics_centralized.get(metric_name, [])
            for round_num, value in pairs:
                logger.info(f"  Round {round_num}: {metric_name}={value:.4f}")
    
    except Exception as exc:
        logger.error(f"Simulation failed: {exc}")
        raise
    
    # Save results
    summary = {
        "dataset": args.dataset,
        "num_clients": len(client_ids),
        "train_rows": int(len(train_part)),
        "test_rows": int(len(test_part)),
        "central_pretrain_fraction": float(args.central_pretrain_fraction),
        "central_pretrain_rows": int(n_pre),
        "central_pretrain_consume_fraction": bool(args.central_pretrain_consume_fraction),
        "drop_next_activity": bool(args.drop_next_activity),
        "hidden_layers": args.hidden_layers,
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "proximal_mu": float(args.proximal_mu),
        "client_freeze_embeddings": bool(args.client_freeze_embeddings),
        "client_trainable_hidden_layers": int(args.client_trainable_hidden_layers),
        "client_send_best_so_far": bool(args.client_send_best_so_far),
        "client_best_metric": str(args.client_best_metric),
        "client_validation_fraction": float(args.client_validation_fraction),
        "mae": float(final_metrics["mae"]),
        "rmse": float(final_metrics["rmse"]),
        "r2": float(final_metrics["r2"]),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = args.result_dir / "federated_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    
    logger.info("Federated training complete")
    logger.info(summary_df.to_string(index=False))
    logger.info(f"Results saved to {summary_path}")
    logger.info(f"Detailed logs:")
    logger.info(f"  - Training log: {args.result_dir}/training.log")
    logger.info(f"  - Server metrics: {args.result_dir}/server_metrics.csv")
    logger.info(f"  - Client metrics: {args.result_dir}/client_metrics.csv")
    logger.info(f"  - Epoch metrics: {args.result_dir}/epoch_metrics.csv")


if __name__ == "__main__":
    main()
