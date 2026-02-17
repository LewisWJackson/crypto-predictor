#!/usr/bin/env python3
"""
Fit the HMM regime detector on training data and produce diagnostics.

Usage:
    python scripts/fit_regime.py
    python scripts/fit_regime.py --config configs/regime_config.yaml
    python scripts/fit_regime.py --processed data/processed/btc_usdt_features.parquet
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from src.regime import RegimeDetector, HMM_FEATURES, REGIME_NAMES
from src.dataset import split_data


def parse_args():
    parser = argparse.ArgumentParser(description="Fit HMM regime detector")
    parser.add_argument(
        "--processed",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "btc_usdt_features.parquet"),
        help="Path to processed features parquet",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "regime_config.yaml"),
        help="Path to regime config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "regime"),
        help="Directory to save model and diagnostics",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.70,
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.15,
    )
    parser.add_argument(
        "--purge-gap", type=int, default=500,
    )
    return parser.parse_args()


def plot_regime_timeline(df, regime_labels, output_path):
    """Plot BTC price colored by regime."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    colors = {
        0: "#2196F3",  # accumulation - blue
        1: "#4CAF50",  # markup - green
        2: "#FF9800",  # distribution - orange
        3: "#F44336",  # markdown - red
    }

    ts = pd.to_datetime(df["timestamp"], unit="ms")
    close = df["close"].values

    # Plot price with regime coloring
    for regime_idx, color in colors.items():
        mask = regime_labels.labels == regime_idx
        ax1.scatter(
            ts[mask], close[mask],
            c=color, s=0.3, alpha=0.6,
            label=REGIME_NAMES[regime_idx],
        )

    ax1.set_ylabel("BTC Price (USD)")
    ax1.set_title("Market Regime Detection — BTC/USDT", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", markerscale=10)
    ax1.grid(True, alpha=0.3)

    # Plot regime labels as colored bar
    for regime_idx, color in colors.items():
        mask = regime_labels.labels == regime_idx
        ax2.fill_between(
            ts, 0, 1, where=mask,
            color=color, alpha=0.7, step="post",
        )

    ax2.set_ylabel("Regime")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Regime timeline saved to: {output_path}")


def plot_transition_matrix(model, output_path):
    """Plot the HMM transition probability matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    trans = model.model.transmat_

    sns.heatmap(
        trans,
        annot=True, fmt=".3f", cmap="Blues",
        xticklabels=REGIME_NAMES,
        yticklabels=REGIME_NAMES,
        ax=ax,
    )
    ax.set_title("Regime Transition Probabilities", fontsize=12, fontweight="bold")
    ax.set_xlabel("To")
    ax.set_ylabel("From")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Transition matrix saved to: {output_path}")


def plot_regime_durations(regime_labels, output_path):
    """Plot histogram of regime durations."""
    labels = regime_labels.labels
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for idx, (name, color) in enumerate(zip(REGIME_NAMES, colors)):
        # Compute run lengths for this regime
        runs = []
        count = 0
        for label in labels:
            if label == idx:
                count += 1
            else:
                if count > 0:
                    runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)

        ax = axes[idx]
        if runs:
            ax.hist(runs, bins=min(50, len(runs)), color=color, alpha=0.7, edgecolor="black")
            ax.axvline(np.median(runs), color="black", linestyle="--", linewidth=1,
                       label=f"Median: {np.median(runs):.0f}")
            ax.legend()
        ax.set_title(f"{name.capitalize()} (n={len(runs)} episodes)", fontsize=11)
        ax.set_xlabel("Duration (bars)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Regime Duration Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Duration histogram saved to: {output_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    hmm_cfg = config.get("hmm", {})

    # Load data
    print(f"Loading data from: {args.processed}")
    df = pd.read_parquet(args.processed)
    print(f"  Total rows: {len(df):,}")

    # Split using same boundaries as dataset.py
    train_df, val_df, test_df = split_data(
        df, args.train_frac, args.val_frac, args.purge_gap
    )
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # Fit HMM on training data only
    print("Fitting HMM regime detector on training data...")
    detector = RegimeDetector(
        n_states=hmm_cfg.get("n_states", 4),
        covariance_type=hmm_cfg.get("covariance_type", "full"),
        n_iter=hmm_cfg.get("n_iter", 200),
        min_duration=hmm_cfg.get("min_duration", 30),
        random_state=hmm_cfg.get("random_state", 42),
    )
    detector.fit(train_df)
    print("  HMM fitted successfully.")

    # Predict on full dataset for diagnostics
    print("Generating regime labels for full dataset...")
    full_labels = detector.predict(df)

    # State statistics
    print("\n--- Regime Statistics (Full Dataset) ---")
    stats = detector.get_state_stats(df)
    for _, row in stats.iterrows():
        print(
            f"  {row['regime']:15s}: {row['count']:>8,} bars "
            f"({row['fraction']:.1%})  "
            f"mean_ret={row['mean_return_60']:.6f}  "
            f"mean_vol={row['mean_volatility_20']:.6f}"
        )

    # Transition matrix
    print("\n--- Transition Matrix ---")
    trans = detector.model.transmat_
    for i, name_from in enumerate(REGIME_NAMES):
        row_str = "  ".join(f"{trans[detector._state_map[i] if hasattr(detector, '_state_map') else i, j]:.3f}" for j in range(4))
        # Actually use the raw matrix since we need HMM state order
    # Just print the remapped matrix
    remapped_trans = np.zeros((4, 4))
    inv_map = {v: k for k, v in detector._state_map.items()}
    for wi in range(4):
        for wj in range(4):
            remapped_trans[wi, wj] = trans[inv_map[wi], inv_map[wj]]

    for i, name in enumerate(REGIME_NAMES):
        row_str = "  ".join(f"{remapped_trans[i, j]:.3f}" for j in range(4))
        print(f"  {name:15s} -> {row_str}")

    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")
    plot_regime_timeline(df, full_labels, output_dir / "regime_timeline.png")
    plot_transition_matrix(detector, output_dir / "transition_matrix.png")
    plot_regime_durations(full_labels, output_dir / "regime_durations.png")

    # Save model
    model_path = output_dir / "hmm_4state.pkl"
    detector.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Also predict on test set only and print stats
    print("\n--- Regime Statistics (Test Set Only) ---")
    test_stats = detector.get_state_stats(test_df)
    for _, row in test_stats.iterrows():
        print(
            f"  {row['regime']:15s}: {row['count']:>8,} bars "
            f"({row['fraction']:.1%})"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
