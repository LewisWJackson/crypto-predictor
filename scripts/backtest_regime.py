#!/usr/bin/env python3
"""
Regime-aware backtesting script.

Compares:
1. Vanilla long/flat strategy (existing baseline)
2. Regime-gated strategy (position sizing + thresholds per regime)

Outputs side-by-side metrics and dual equity curves color-coded by regime.

Usage:
    python scripts/backtest_regime.py --checkpoint models/tft/tft-best.ckpt
    python scripts/backtest_regime.py --checkpoint models/tft/tft-best.ckpt --regime-model models/regime/hmm_4state.pkl
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

from src.dataset import create_datasets, create_dataloaders, split_data
from src.metrics import compute_metrics
from src.model import load_config
from src.regime import RegimeDetector, REGIME_NAMES
from src.strategy import (
    simulate_regime_strategy,
    load_regime_configs,
    DEFAULT_REGIME_CONFIGS,
)
from scripts.backtest import run_predictions, simulate_strategy


def parse_args():
    parser = argparse.ArgumentParser(description="Regime-aware backtest")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to TFT model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "tft_config.yaml"),
        help="Path to TFT config YAML",
    )
    parser.add_argument(
        "--regime-model", type=str,
        default=str(PROJECT_ROOT / "models" / "regime" / "hmm_4state.pkl"),
        help="Path to fitted HMM regime model",
    )
    parser.add_argument(
        "--regime-config", type=str,
        default=str(PROJECT_ROOT / "configs" / "regime_config.yaml"),
        help="Path to regime strategy config",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "models" / "regime"),
        help="Directory to save backtest outputs",
    )
    return parser.parse_args()


def get_test_regime_labels(
    processed_path: str,
    regime_model_path: str,
    train_frac: float,
    val_frac: float,
    purge_gap: int,
    start_date: str | None = None,
):
    """Load the test split and compute regime labels on it.

    Returns the regime labels/probs aligned with the test set.
    """
    import pandas as pd

    df = pd.read_parquet(processed_path)

    if start_date:
        ts = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[ts >= pd.Timestamp(start_date)].reset_index(drop=True)

    _, _, test_df = split_data(df, train_frac, val_frac, purge_gap)

    detector = RegimeDetector.load(regime_model_path)
    regime_result = detector.predict(test_df)

    return regime_result, test_df


def plot_dual_equity(vanilla, regime, regime_labels, output_path):
    """Plot side-by-side equity curves with regime coloring."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14),
                             gridspec_kw={"height_ratios": [3, 3, 1, 1]})

    colors = {
        0: "#2196F3",  # accumulation
        1: "#4CAF50",  # markup
        2: "#FF9800",  # distribution
        3: "#F44336",  # markdown
    }

    n = min(len(vanilla["cumulative_pnl"]), len(regime["cumulative_pnl"]))
    x = np.arange(n)

    # Trim regime labels to match prediction count
    labels = regime_labels.labels[:n] if len(regime_labels.labels) >= n else regime_labels.labels

    # --- Panel 1: Vanilla strategy ---
    ax1 = axes[0]
    ax1.plot(x, vanilla["cumulative_pnl"][:n], color="#9E9E9E", linewidth=1.2, label="Vanilla Long/Flat")
    ax1.plot(x, vanilla["bh_cumulative"][:n], color="#BDBDBD", linewidth=0.8, alpha=0.5, label="Buy & Hold")
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_title("Vanilla Long/Flat Strategy", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cumulative Log Return")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Regime-gated strategy ---
    ax2 = axes[1]
    ax2.plot(x, regime["cumulative_pnl"][:n], color="#2196F3", linewidth=1.2, label="Regime-Gated")
    ax2.plot(x, regime["bh_cumulative"][:n], color="#BDBDBD", linewidth=0.8, alpha=0.5, label="Buy & Hold")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_title("Regime-Gated Strategy", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cumulative Log Return")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Effective position size ---
    ax3 = axes[2]
    ax3.fill_between(x, regime["effective_position_size"][:n],
                     color="#4CAF50", alpha=0.4, step="post")
    ax3.set_ylabel("Position Size")
    ax3.set_title("Effective Position Size (Regime-Blended)", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Regime timeline ---
    ax4 = axes[3]
    if len(labels) == n:
        for regime_idx, color in colors.items():
            mask = labels == regime_idx
            ax4.fill_between(x, 0, 1, where=mask, color=color, alpha=0.7, step="post")
    ax4.set_ylabel("Regime")
    ax4.set_xlabel("Trade Index (15-min bars)")
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])

    # Add legend for regime colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.7, label=REGIME_NAMES[i].capitalize())
                       for i in range(4)]
    ax4.legend(handles=legend_elements, loc="upper left", ncol=4)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dual equity curve saved to: {output_path}")


def print_comparison_report(vanilla, regime):
    """Print side-by-side comparison of vanilla vs regime-gated strategy."""
    print("\n" + "=" * 75)
    print("BACKTEST COMPARISON — Vanilla vs Regime-Gated Strategy")
    print("=" * 75)

    metrics = [
        ("Total Return (log)", "total_return", ".6f"),
        ("Sharpe Ratio (ann.)", "sharpe_ratio", ".4f"),
        ("Max Drawdown", "max_drawdown", ".6f"),
        ("Win Rate", "win_rate", ".4f"),
        ("Profit Factor", "profit_factor", ".4f"),
        ("Total Trades", "total_trades", ",d"),
        ("Wins", "wins", ",d"),
        ("Losses", "losses", ",d"),
    ]

    header = f"{'Metric':<25s} {'Vanilla':>15s} {'Regime-Gated':>15s} {'Delta':>12s}"
    print(f"\n{header}")
    print("-" * 70)

    for label, key, fmt in metrics:
        v_val = vanilla[key]
        r_val = regime[key]

        v_str = f"{v_val:{fmt}}"
        r_str = f"{r_val:{fmt}}"

        if isinstance(v_val, int):
            delta = r_val - v_val
            d_str = f"{delta:+,d}"
        else:
            delta = r_val - v_val
            d_str = f"{delta:+.4f}"

        print(f"  {label:<23s} {v_str:>15s} {r_str:>15s} {d_str:>12s}")

    # Drawdown improvement
    if vanilla["max_drawdown"] > 0:
        dd_reduction = (vanilla["max_drawdown"] - regime["max_drawdown"]) / vanilla["max_drawdown"] * 100
        print(f"\n  Drawdown reduction: {dd_reduction:.1f}%")

    print("\n" + "=" * 75)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load TFT model and config
    # ------------------------------------------------------------------
    print(f"Loading TFT config from: {args.config}")
    config = load_config(args.config)

    print(f"Loading TFT model from: {args.checkpoint}")
    model = TemporalFusionTransformer.load_from_checkpoint(args.checkpoint)
    model.eval()

    # ------------------------------------------------------------------
    # 2. Create test dataset
    # ------------------------------------------------------------------
    dataset_cfg = config.get("dataset", {})
    data_cfg = config.get("data", {})
    timeframe = data_cfg.get("timeframe", "1m")
    pair = data_cfg.get("pair", "BTC_USDT")
    if timeframe == "1m":
        processed_path = (
            Path(data_cfg.get("processed_dir", "data/processed"))
            / f"{pair}_features.parquet"
        )
    else:
        processed_path = (
            Path(data_cfg.get("processed_dir", "data/processed"))
            / f"{pair}_{timeframe}_features.parquet"
        )
    if not processed_path.is_absolute():
        processed_path = PROJECT_ROOT / processed_path

    target_col = dataset_cfg.get("target_col", "forward_return_15")

    training_dataset, validation_dataset, test_dataset, norm_stats = create_datasets(
        processed_path=str(processed_path),
        train_frac=dataset_cfg.get("train_fraction", 0.70),
        val_frac=dataset_cfg.get("val_fraction", 0.15),
        purge_gap=dataset_cfg.get("purge_gap", 500),
        encoder_length=dataset_cfg.get("encoder_length", 256),
        decoder_length=dataset_cfg.get("decoder_length", 15),
        target_col=target_col,
        start_date=data_cfg.get("start_date"),
    )

    batch_size = dataset_cfg.get("batch_size", 64) * 2
    _, _, test_dataloader = create_dataloaders(
        training_dataset, validation_dataset, test_dataset,
        batch_size=batch_size,
        num_workers=dataset_cfg.get("num_workers", 0),
    )

    # ------------------------------------------------------------------
    # 3. Run TFT predictions
    # ------------------------------------------------------------------
    print("Running TFT predictions on test set...")
    with torch.no_grad():
        y_true, y_pred = run_predictions(model, test_dataloader)
    print(f"  Predictions shape: {y_pred.shape}")

    # ------------------------------------------------------------------
    # 4. Compute regime labels for test set
    # ------------------------------------------------------------------
    print(f"Loading regime model from: {args.regime_model}")
    regime_result, test_raw_df = get_test_regime_labels(
        processed_path=str(processed_path),
        regime_model_path=args.regime_model,
        train_frac=dataset_cfg.get("train_fraction", 0.70),
        val_frac=dataset_cfg.get("val_fraction", 0.15),
        purge_gap=dataset_cfg.get("purge_gap", 500),
        start_date=data_cfg.get("start_date"),
    )

    # Align regime labels with prediction count
    # The test dataloader may produce fewer samples than the raw test split
    # due to encoder_length requirements
    n_preds = len(y_true)
    n_regime = len(regime_result.labels)

    # Regime labels correspond to the END of the test split (predictions
    # start encoder_length bars into the test split)
    offset = n_regime - n_preds
    if offset < 0:
        offset = 0
    regime_labels_aligned = regime_result.labels[offset:offset + n_preds]
    regime_probs_aligned = regime_result.probabilities[offset:offset + n_preds]

    print(f"  Regime labels aligned: {len(regime_labels_aligned)} "
          f"(offset={offset}, test_raw={n_regime}, preds={n_preds})")

    # Print regime distribution in test predictions
    for i, name in enumerate(REGIME_NAMES):
        count = int(np.sum(regime_labels_aligned == i))
        frac = count / len(regime_labels_aligned) if len(regime_labels_aligned) > 0 else 0
        print(f"    {name:15s}: {count:>6,} ({frac:.1%})")

    # ------------------------------------------------------------------
    # 5. Run both strategies
    # ------------------------------------------------------------------
    print("\nRunning vanilla long/flat strategy...")
    vanilla_results = simulate_strategy(y_true, y_pred)

    print("Running regime-gated strategy...")
    # Load regime configs
    regime_config_path = Path(args.regime_config)
    if regime_config_path.exists():
        regime_configs = load_regime_configs(regime_config_path)
    else:
        regime_configs = DEFAULT_REGIME_CONFIGS

    y_true_final = y_true[:, -1]
    y_pred_final = y_pred[:, -1]
    regime_results = simulate_regime_strategy(
        y_true_final, y_pred_final, regime_probs_aligned, regime_configs
    )

    # ------------------------------------------------------------------
    # 6. Compare and report
    # ------------------------------------------------------------------
    print_comparison_report(vanilla_results, regime_results)

    # Model quality metrics
    model_metrics = compute_metrics(y_true, y_pred)
    print(f"\n  Model Directional Accuracy (step 15): {model_metrics['directional_accuracy_15']:.4f}")

    # ------------------------------------------------------------------
    # 7. Plot dual equity curves
    # ------------------------------------------------------------------
    from src.regime import RegimeLabels
    aligned_regime = RegimeLabels(
        labels=regime_labels_aligned,
        probabilities=regime_probs_aligned,
        names=REGIME_NAMES,
    )
    plot_path = output_dir / "backtest_regime_comparison.png"
    plot_dual_equity(vanilla_results, regime_results, aligned_regime, str(plot_path))

    print("\nDone.")


if __name__ == "__main__":
    main()
