#!/usr/bin/env python3
"""
Backtesting Script for TFT Crypto Predictor.

Loads a trained TFT checkpoint, runs predictions on the test set,
simulates a long/flat trading strategy, computes P&L metrics, and
plots an equity curve.

Strategy:
    - Go LONG when model predicts UP (positive return at step 15)
    - Go FLAT when model predicts DOWN (negative return at step 15)

Usage:
    python scripts/backtest.py --checkpoint models/tft/tft-best.ckpt
    python scripts/backtest.py --checkpoint models/tft/tft-best.ckpt --config configs/tft_config.yaml
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer

from src.dataset import create_datasets, create_dataloaders
from src.metrics import compute_metrics
from src.model import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest a trained TFT model on the test set"
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
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "tft"),
        help="Directory to save backtest outputs",
    )
    return parser.parse_args()


def run_predictions(model, test_dataloader):
    """Run model predictions and collect actuals from the test dataloader.

    Returns:
        y_true: np.ndarray of actual log returns
        y_pred: np.ndarray of predicted log returns
    """
    predictions = model.predict(
        test_dataloader, mode="prediction", return_x=False
    )
    y_pred = predictions.numpy() if hasattr(predictions, "numpy") else np.array(predictions)

    actuals = []
    for batch in test_dataloader:
        x, y = batch
        if isinstance(y, (tuple, list)):
            actuals.append(y[0].numpy())
        else:
            actuals.append(y.numpy())

    y_true = np.concatenate(actuals, axis=0)

    # Align lengths
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # Ensure 2D: (N, horizon)
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    return y_true, y_pred


def simulate_strategy(y_true, y_pred):
    """Simulate a long/flat strategy based on final-step predictions.

    Strategy: go long when predicted return > 0, go flat when <= 0.

    Args:
        y_true: Actual log returns, shape (N, horizon)
        y_pred: Predicted log returns, shape (N, horizon)

    Returns:
        dict with strategy arrays and summary metrics
    """
    # Use the final step (step 15) for both signal and realized return
    y_true_final = y_true[:, -1]
    y_pred_final = y_pred[:, -1]

    # Long/flat: position = 1 when predicted UP, 0 when predicted DOWN
    positions = (y_pred_final > 0).astype(float)

    # Per-trade returns (only earn when positioned long)
    strategy_returns = positions * y_true_final

    # Cumulative P&L (log-return based)
    cumulative_pnl = np.cumsum(strategy_returns)

    # --- Metrics ---
    total_trades = int(np.sum(positions > 0))  # only count trades where we are long
    wins = int(np.sum((strategy_returns > 0) & (positions > 0)))
    losses = int(np.sum((strategy_returns < 0) & (positions > 0)))

    win_rate = wins / total_trades if total_trades > 0 else 0.0

    # Sharpe ratio (annualized, 15-min bars -> ~35,040 bars/year)
    trade_returns = strategy_returns[positions > 0]
    if len(trade_returns) > 1:
        mean_ret = np.mean(trade_returns)
        std_ret = np.std(trade_returns) + 1e-8
        sharpe = float(mean_ret / std_ret * np.sqrt(35040))
    else:
        sharpe = 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Profit factor
    gross_profit = float(np.sum(strategy_returns[strategy_returns > 0]))
    gross_loss = float(np.abs(np.sum(strategy_returns[strategy_returns < 0]))) + 1e-8
    profit_factor = gross_profit / gross_loss

    # Buy & hold benchmark (just sum all realized returns)
    bh_cumulative = np.cumsum(y_true_final)

    return {
        "positions": positions,
        "strategy_returns": strategy_returns,
        "cumulative_pnl": cumulative_pnl,
        "bh_cumulative": bh_cumulative,
        "total_return": float(cumulative_pnl[-1]) if len(cumulative_pnl) > 0 else 0.0,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "n_samples": len(y_true_final),
    }


def plot_equity_curve(results, output_path):
    """Plot and save the equity curve with buy & hold comparison."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})

    # --- Equity curve ---
    ax1 = axes[0]
    ax1.plot(results["cumulative_pnl"], label="Strategy (Long/Flat)", color="#2196F3", linewidth=1.2)
    ax1.plot(results["bh_cumulative"], label="Buy & Hold", color="#9E9E9E", linewidth=0.8, alpha=0.7)
    ax1.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_title("Backtest: TFT Long/Flat Strategy vs Buy & Hold", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Cumulative Log Return")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Annotate final P&L
    final_pnl = results["cumulative_pnl"][-1]
    ax1.annotate(
        f"Final P&L: {final_pnl:.4f}",
        xy=(len(results["cumulative_pnl"]) - 1, final_pnl),
        fontsize=10, fontweight="bold",
        color="#2196F3",
    )

    # --- Drawdown ---
    ax2 = axes[1]
    running_max = np.maximum.accumulate(results["cumulative_pnl"])
    drawdown = running_max - results["cumulative_pnl"]
    ax2.fill_between(range(len(drawdown)), drawdown, color="#F44336", alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdown", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # --- Positions ---
    ax3 = axes[2]
    ax3.fill_between(range(len(results["positions"])), results["positions"], color="#4CAF50", alpha=0.3, step="post")
    ax3.set_ylabel("Position")
    ax3.set_xlabel("Trade Index (15-min bars)")
    ax3.set_title("Position (1=Long, 0=Flat)", fontsize=10)
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Equity curve saved to: {output_path}")


def print_backtest_report(results, model_metrics):
    """Print a formatted backtest report."""
    print("\n" + "=" * 70)
    print("BACKTEST REPORT — TFT Long/Flat Strategy")
    print("=" * 70)

    print(f"\n  Test samples:       {results['n_samples']}")
    print(f"  Total trades (long): {results['total_trades']}")
    print(f"  Wins / Losses:       {results['wins']} / {results['losses']}")

    print(f"\n  --- P&L Metrics ---")
    print(f"  Cumulative P&L:      {results['total_return']:.6f} (log return)")
    print(f"  Sharpe Ratio (ann.): {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:        {results['max_drawdown']:.6f}")
    print(f"  Win Rate:            {results['win_rate']:.4f} ({results['win_rate']*100:.1f}%)")
    print(f"  Profit Factor:       {results['profit_factor']:.4f}")

    print(f"\n  --- Model Quality Metrics ---")
    print(f"  Directional Acc (step 15): {model_metrics['directional_accuracy_15']:.4f}")
    print(f"  Directional Acc (all):     {model_metrics['directional_accuracy_all_steps']:.4f}")
    print(f"  RMSE:                      {model_metrics['rmse']:.6f}")
    print(f"  MAE:                       {model_metrics['mae']:.6f}")

    print("\n" + "=" * 70)


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load config and model
    # ------------------------------------------------------------------
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    print(f"Loading model from: {args.checkpoint}")
    model = TemporalFusionTransformer.load_from_checkpoint(args.checkpoint)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # ------------------------------------------------------------------
    # 2. Create test dataset and dataloader
    # ------------------------------------------------------------------
    print("Creating datasets...")
    dataset_cfg = config.get("dataset", {})
    processed_path = (
        Path(config.get("data", {}).get("processed_dir", "data/processed"))
        / f"{config['data']['pair']}_features.parquet"
    )
    # Make path absolute relative to project root
    if not processed_path.is_absolute():
        processed_path = PROJECT_ROOT / processed_path

    target_col = dataset_cfg.get("target_col", "log_return_15")

    training_dataset, validation_dataset, test_dataset, norm_stats = create_datasets(
        processed_path=str(processed_path),
        train_frac=dataset_cfg.get("train_fraction", 0.70),
        val_frac=dataset_cfg.get("val_fraction", 0.15),
        purge_gap=dataset_cfg.get("purge_gap", 500),
        encoder_length=dataset_cfg.get("encoder_length", 256),
        decoder_length=dataset_cfg.get("decoder_length", 15),
        target_col=target_col,
    )

    batch_size = dataset_cfg.get("batch_size", 64) * 2
    _, _, test_dataloader = create_dataloaders(
        training_dataset, validation_dataset, test_dataset,
        batch_size=batch_size,
        num_workers=dataset_cfg.get("num_workers", 0),
    )

    print(f"  Test samples: {len(test_dataset)}")

    # ------------------------------------------------------------------
    # 3. Run predictions
    # ------------------------------------------------------------------
    print("Running predictions on test set...")
    with torch.no_grad():
        y_true, y_pred = run_predictions(model, test_dataloader)
    print(f"  Predictions shape: {y_pred.shape}")
    print(f"  Actuals shape:     {y_true.shape}")

    # ------------------------------------------------------------------
    # 4. Compute model quality metrics (reuse src/metrics)
    # ------------------------------------------------------------------
    print("Computing metrics...")
    model_metrics = compute_metrics(y_true, y_pred)

    # ------------------------------------------------------------------
    # 5. Simulate trading strategy
    # ------------------------------------------------------------------
    print("Simulating long/flat strategy...")
    results = simulate_strategy(y_true, y_pred)

    # ------------------------------------------------------------------
    # 6. Plot equity curve
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "backtest_equity.png"
    plot_equity_curve(results, str(plot_path))

    # ------------------------------------------------------------------
    # 7. Print report
    # ------------------------------------------------------------------
    print_backtest_report(results, model_metrics)


if __name__ == "__main__":
    main()
