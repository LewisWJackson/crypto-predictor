#!/usr/bin/env python3
"""
TFT Training Script for Crypto Price Prediction.

Loads config, creates datasets/dataloaders, instantiates the TFT model,
trains with PyTorch Lightning, evaluates on the test set, and saves
the best checkpoint.

Usage:
    python scripts/train_tft.py
    python scripts/train_tft.py --config configs/tft_config.yaml
    python scripts/train_tft.py --fast-dev-run  # Quick test (1 batch)
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.dataset import create_datasets, create_dataloaders
from src.metrics import compute_metrics, format_metrics_report
from src.model import create_model, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TFT model for crypto price prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "tft_config.yaml"),
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a single batch for testing (no full training)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # 1. Load configuration
    # -----------------------------------------------------------------------
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})

    # -----------------------------------------------------------------------
    # 2. Create datasets and dataloaders
    # -----------------------------------------------------------------------
    print("Creating datasets...")

    # Build processed features path from config
    data_cfg = config.get("data", {})
    processed_dir = PROJECT_ROOT / data_cfg.get("processed_dir", "data/processed")
    pair = data_cfg.get("pair", "BTC_USDT")
    processed_path = str(processed_dir / f"{pair}_features.parquet")

    target_col = dataset_cfg.get("target_col", "log_return_15")

    training_dataset, validation_dataset, test_dataset, norm_stats = create_datasets(
        processed_path=processed_path,
        train_frac=dataset_cfg.get("train_fraction", 0.70),
        val_frac=dataset_cfg.get("val_fraction", 0.15),
        purge_gap=dataset_cfg.get("purge_gap", 500),
        encoder_length=dataset_cfg.get("encoder_length", 256),
        decoder_length=dataset_cfg.get("decoder_length", 15),
        group_name=pair,
        target_col=target_col,
    )

    print(
        f"  Training samples:   {len(training_dataset)}\n"
        f"  Validation samples: {len(validation_dataset)}\n"
        f"  Test samples:       {len(test_dataset)}"
    )

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        training_dataset,
        validation_dataset,
        test_dataset,
        batch_size=dataset_cfg.get("batch_size", 64),
        num_workers=dataset_cfg.get("num_workers", 4),
    )

    # -----------------------------------------------------------------------
    # 3. Create model
    # -----------------------------------------------------------------------
    print("Creating TFT model...")
    model = create_model(training_dataset, config)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -----------------------------------------------------------------------
    # 4. Configure trainer
    # -----------------------------------------------------------------------
    model_dir = PROJECT_ROOT / "models" / "tft"
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=training_cfg.get("early_stopping_patience", 10),
            min_delta=training_cfg.get("early_stopping_min_delta", 1e-5),
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=str(model_dir),
            monitor="val_loss",
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            save_top_k=training_cfg.get("checkpoint_top_k", 3),
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=training_cfg.get("max_epochs", 100),
        gradient_clip_val=training_cfg.get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        fast_dev_run=args.fast_dev_run,
        default_root_dir=str(model_dir / "training_logs"),
    )

    # -----------------------------------------------------------------------
    # 5. Train
    # -----------------------------------------------------------------------
    print("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    if args.fast_dev_run:
        print("Fast dev run complete. Skipping evaluation.")
        return

    # -----------------------------------------------------------------------
    # 6. Evaluate on test set
    # -----------------------------------------------------------------------
    print("\nLoading best checkpoint for evaluation...")
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"  Best model: {best_model_path}")

    from pytorch_forecasting import TemporalFusionTransformer

    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    print("Running predictions on test set...")
    predictions = best_model.predict(
        test_dataloader, mode="prediction", return_x=False
    )

    # Get actual targets from the test dataloader
    actuals = []
    for batch in test_dataloader:
        x, y = batch
        # y is a tuple: (target, weight) — we need the target
        if isinstance(y, (tuple, list)):
            actuals.append(y[0].numpy())
        else:
            actuals.append(y.numpy())

    y_true = np.concatenate(actuals, axis=0)
    y_pred = predictions.numpy()

    # Ensure shapes match (trim if needed due to batching)
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # Handle single-step predictions: expand to 2D if needed
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    print(f"  Predictions shape: {y_pred.shape}")
    print(f"  Actuals shape:     {y_true.shape}")

    # -----------------------------------------------------------------------
    # 7. Compute and print metrics
    # -----------------------------------------------------------------------
    metrics = compute_metrics(y_true, y_pred)
    report = format_metrics_report(metrics)
    print(report)

    # Save metrics summary
    metrics_path = model_dir / "training_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(report)
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
