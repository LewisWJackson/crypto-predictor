#!/usr/bin/env python3
"""
Evaluation Script for TFT Crypto Predictor.

Loads a trained model checkpoint, runs predictions on the test set,
computes all metrics, compares against baselines, and prints a
formatted evaluation report.

Usage:
    python scripts/evaluate.py --checkpoint models/tft/best.ckpt
    python scripts/evaluate.py --checkpoint models/tft/best.ckpt --save-report
    python scripts/evaluate.py --checkpoint models/tft/best.ckpt --config configs/tft_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.dataset import create_datasets, create_dataloaders
from src.metrics import (
    compute_baseline_metrics,
    compute_metrics,
    format_metrics_report,
)
from src.model import load_config, load_trained_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TFT model on the test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "tft_config.yaml"),
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save evaluation report to a file alongside the checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for the report (overrides default)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # 1. Load config and model
    # -----------------------------------------------------------------------
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    print(f"Loading model from: {args.checkpoint}")
    model = load_trained_model(args.checkpoint)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -----------------------------------------------------------------------
    # 2. Create test dataset and dataloader
    # -----------------------------------------------------------------------
    print("Creating datasets...")
    dataset_cfg = config.get("dataset", {})
    data_cfg = config.get("data", {})
    processed_dir = PROJECT_ROOT / data_cfg.get("processed_dir", "data/processed")
    pair = data_cfg.get("pair", "BTC_USDT")
    timeframe = data_cfg.get("timeframe", "1m")
    if timeframe == "1m":
        processed_path = str(processed_dir / f"{pair}_features.parquet")
    else:
        processed_path = str(processed_dir / f"{pair}_{timeframe}_features.parquet")

    target_col = dataset_cfg.get("target_col", "forward_return_15")

    training_dataset, validation_dataset, test_dataset, norm_stats = create_datasets(
        processed_path=processed_path,
        train_frac=dataset_cfg.get("train_fraction", 0.70),
        val_frac=dataset_cfg.get("val_fraction", 0.15),
        purge_gap=dataset_cfg.get("purge_gap", 500),
        encoder_length=dataset_cfg.get("encoder_length", 256),
        decoder_length=dataset_cfg.get("decoder_length", 15),
        group_name=pair,
        target_col=target_col,
        start_date=data_cfg.get("start_date"),
    )

    _, _, test_dataloader = create_dataloaders(
        training_dataset,
        validation_dataset,
        test_dataset,
        batch_size=dataset_cfg.get("batch_size", 64) * 2,  # Larger batch for eval
        num_workers=dataset_cfg.get("num_workers", 0),
    )

    print(f"  Test samples: {len(test_dataset)}")

    # -----------------------------------------------------------------------
    # 3. Run predictions
    # -----------------------------------------------------------------------
    print("Running predictions on test set...")
    predictions = model.predict(
        test_dataloader, mode="prediction", return_x=True
    )

    # predictions is a tuple (output, x) when return_x=True
    y_pred = predictions[0].numpy() if isinstance(predictions, tuple) else predictions.numpy()

    # Collect actual targets
    actuals = []
    for batch in test_dataloader:
        x, y = batch
        if isinstance(y, (tuple, list)):
            actuals.append(y[0].numpy())
        else:
            actuals.append(y.numpy())

    y_true = np.concatenate(actuals, axis=0)

    # Ensure shapes match
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # Ensure 2D
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    print(f"  Predictions shape: {y_pred.shape}")
    print(f"  Actuals shape:     {y_true.shape}")

    # -----------------------------------------------------------------------
    # 4. Compute model metrics
    # -----------------------------------------------------------------------
    print("\nComputing metrics...")
    model_metrics = compute_metrics(y_true, y_pred)

    # -----------------------------------------------------------------------
    # 5. Compute baseline metrics
    # -----------------------------------------------------------------------
    print("Computing baseline metrics...")

    # Last observed return for persistence baseline
    # Use the first column (step 0) of y_true shifted by 1 as approximation
    last_returns = y_true[:-1, 0]  # Previous step's first-horizon return
    y_true_for_baselines = y_true[1:]  # Align with last_returns
    y_pred_for_baselines = y_pred[1:]

    baseline_metrics = compute_baseline_metrics(
        y_true=y_true_for_baselines,
        last_returns=last_returns,
        close_prices=None,  # Would need raw price data for MA crossover
    )

    # -----------------------------------------------------------------------
    # 6. Print report
    # -----------------------------------------------------------------------
    report = format_metrics_report(model_metrics, baseline_metrics)
    print(report)

    # -----------------------------------------------------------------------
    # 7. Check against thresholds
    # -----------------------------------------------------------------------
    eval_cfg = config.get("evaluation", {})
    min_da = eval_cfg.get("min_directional_accuracy", 0.52)
    min_sharpe = eval_cfg.get("min_sharpe_ratio", 0.5)
    max_dd = eval_cfg.get("max_drawdown", 0.20)

    print("\n--- Threshold Check ---")
    da = model_metrics["directional_accuracy_15"]
    sharpe = model_metrics["sharpe_ratio"]
    dd = model_metrics["max_drawdown"]

    pass_da = da >= min_da
    pass_sharpe = sharpe >= min_sharpe
    pass_dd = dd <= max_dd

    print(f"  Directional Acc >= {min_da:.2f}: {da:.4f}  {'PASS' if pass_da else 'FAIL'}")
    print(f"  Sharpe Ratio   >= {min_sharpe:.2f}: {sharpe:.4f}  {'PASS' if pass_sharpe else 'FAIL'}")
    print(f"  Max Drawdown   <= {max_dd:.2f}: {dd:.4f}  {'PASS' if pass_dd else 'FAIL'}")

    all_pass = pass_da and pass_sharpe and pass_dd
    print(f"\n  Overall: {'ALL THRESHOLDS MET' if all_pass else 'SOME THRESHOLDS NOT MET'}")

    # -----------------------------------------------------------------------
    # 8. Optionally save report
    # -----------------------------------------------------------------------
    if args.save_report or args.output:
        if args.output:
            report_path = Path(args.output)
        else:
            ckpt_path = Path(args.checkpoint)
            report_path = ckpt_path.parent / f"{ckpt_path.stem}_evaluation.txt"

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)
            f.write(f"\n\nCheckpoint: {args.checkpoint}\n")
            f.write(f"Config: {args.config}\n")

        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
