#!/usr/bin/env python3
"""
Multi-Model Comparison Dashboard for TFT Crypto Predictor.

Loads 3 experiment checkpoints (baseline 15-min, 5-min horizon, classification),
runs predictions on each, computes metrics, and generates a side-by-side
comparison dashboard.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --output-dir figures/comparison
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.dataset import create_datasets, create_dataloaders
from src.metrics import compute_metrics, compute_baseline_metrics
from src.model import load_config, load_trained_model


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "cyan":       "#00d4ff",
    "magenta":    "#ff2e97",
    "green":      "#00ff88",
    "orange":     "#ff9f1c",
    "purple":     "#b388ff",
    "yellow":     "#ffe66d",
    "red":        "#ff4444",
    "grey":       "#888888",
    "light_grey": "#bbbbbb",
    "dark_grey":  "#333333",
    "bg":         "#0e1117",
    "grid":       "#1e2330",
    "white":      "#e0e0e0",
}

MODEL_COLORS = {
    "Baseline (15-min)":     "#00d4ff",   # cyan
    "5-Min Horizon":         "#ff9f1c",   # orange
    "Classification":        "#b388ff",   # purple
    "Random Baseline":       "#888888",   # grey
}


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "name": "Baseline (15-min)",
        "checkpoint": "models/tft/tft-epoch=01-val_loss=0.2113.ckpt",
        "config": "configs/tft_config.yaml",
        "eval_file": "models/tft/tft-epoch=01-val_loss=0.2113_evaluation.txt",
    },
    {
        "name": "5-Min Horizon",
        "checkpoint": "models/tft/tft-epoch=02-val_loss=0.2202.ckpt",
        "config": "configs/experiment_5min.yaml",
        "eval_file": "models/tft/tft-epoch=02-val_loss=0.2202_evaluation.txt",
    },
    {
        "name": "Classification",
        "checkpoint": "models/tft/tft-epoch=01-val_loss=13.8759.ckpt",
        "config": "configs/experiment_classification.yaml",
        "eval_file": "models/tft/tft-epoch=01-val_loss=13.8759_evaluation.txt",
    },
]


def load_experiment_data(experiment):
    """Load model, run predictions, compute metrics for one experiment."""
    config_path = str(PROJECT_ROOT / experiment["config"])
    ckpt_path = str(PROJECT_ROOT / experiment["checkpoint"])

    print(f"\n{'='*60}")
    print(f"Loading: {experiment['name']}")
    print(f"  Config: {experiment['config']}")
    print(f"  Checkpoint: {experiment['checkpoint']}")
    print(f"{'='*60}")

    config = load_config(config_path)
    model = load_trained_model(ckpt_path)

    dataset_cfg = config.get("dataset", {})
    data_cfg = config.get("data", {})
    processed_dir = PROJECT_ROOT / data_cfg.get("processed_dir", "data/processed")
    pair = data_cfg.get("pair", "BTC_USDT")
    processed_path = str(processed_dir / f"{pair}_features.parquet")
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
    )

    _, _, test_dataloader = create_dataloaders(
        training_dataset, validation_dataset, test_dataset,
        batch_size=dataset_cfg.get("batch_size", 64) * 2,
        num_workers=dataset_cfg.get("num_workers", 0),
    )

    print(f"  Test samples: {len(test_dataset)}")
    print("  Running predictions...")

    predictions = model.predict(test_dataloader, mode="prediction", return_x=False)
    y_pred = predictions.numpy()

    actuals = []
    for batch in test_dataloader:
        x, y = batch
        if isinstance(y, (tuple, list)):
            actuals.append(y[0].numpy())
        else:
            actuals.append(y.numpy())
    y_true = np.concatenate(actuals, axis=0)

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    print(f"  Predictions shape: {y_pred.shape}")
    print(f"  Actuals shape:     {y_true.shape}")

    metrics = compute_metrics(y_true, y_pred)

    # Compute cumulative P&L for final step
    y_true_final = y_true[:, -1]
    y_pred_final = y_pred[:, -1]
    positions = np.sign(y_pred_final)
    strategy_returns = positions * y_true_final
    cum_pnl = np.cumsum(strategy_returns)

    # Rolling directional accuracy
    correct = (np.sign(y_true_final) == np.sign(y_pred_final)).astype(float)
    window = 500
    kernel = np.ones(window) / window
    rolling_da = np.convolve(correct, kernel, mode="valid")

    return {
        "name": experiment["name"],
        "config": experiment["config"],
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "cum_pnl": cum_pnl,
        "rolling_da": rolling_da,
        "strategy_returns": strategy_returns,
        "decoder_length": dataset_cfg.get("decoder_length", 15),
    }


def load_from_eval_files():
    """Fallback: load metrics from evaluation text files without running inference."""
    import re

    results = []
    for exp in EXPERIMENTS:
        eval_path = PROJECT_ROOT / exp["eval_file"]
        if not eval_path.exists():
            print(f"  Warning: {eval_path} not found, skipping {exp['name']}")
            continue

        text = eval_path.read_text()
        metrics = {}

        patterns = {
            "mse": r"MSE:\s+([\d.]+)",
            "rmse": r"RMSE:\s+([\d.]+)",
            "mae": r"MAE:\s+([\d.]+)",
            "directional_accuracy_all_steps": r"Directional Acc \(all steps\):\s+([\d.]+)",
            "directional_accuracy_15": r"Directional Acc \(step 15\):\s+([\d.]+)",
            "win_rate": r"Win Rate:\s+([\d.]+)",
            "sharpe_ratio": r"Sharpe Ratio \(annualized\):\s+([-\d.]+)",
            "total_return": r"Total Return:\s+([-\d.]+)",
            "max_drawdown": r"Max Drawdown:\s+([\d.]+)",
            "profit_factor": r"Profit Factor:\s+([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                metrics[key] = float(match.group(1))

        results.append({
            "name": exp["name"],
            "config": exp["config"],
            "metrics": metrics,
        })

    return results


def create_comparison_dashboard(results, output_dir, dpi=150):
    """Create a multi-model comparison dashboard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor":    COLORS["bg"],
        "figure.facecolor":  COLORS["bg"],
        "savefig.facecolor": COLORS["bg"],
        "axes.edgecolor":    COLORS["grid"],
        "axes.grid":         True,
        "grid.color":        COLORS["grid"],
        "grid.linewidth":    0.4,
        "grid.alpha":        0.6,
        "font.family":       "sans-serif",
        "font.size":         9,
        "axes.labelcolor":   COLORS["light_grey"],
        "xtick.color":       COLORS["light_grey"],
        "ytick.color":       COLORS["light_grey"],
        "text.color":        COLORS["white"],
        "legend.facecolor":  COLORS["bg"],
        "legend.edgecolor":  COLORS["dark_grey"],
    })

    has_predictions = "cum_pnl" in results[0]

    if has_predictions:
        fig = plt.figure(figsize=(22, 16))
        gs = gridspec.GridSpec(3, 2, figure=fig,
                               left=0.06, right=0.96,
                               top=0.91, bottom=0.04,
                               hspace=0.35, wspace=0.25)
    else:
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               left=0.06, right=0.96,
                               top=0.91, bottom=0.06,
                               hspace=0.35, wspace=0.25)

    # --- Title banner ---
    fig.text(
        0.5, 0.96,
        "TFT Crypto Predictor — Multi-Model Comparison Dashboard",
        fontsize=16, fontweight="bold", color=COLORS["cyan"],
        ha="center", va="center", family="monospace",
    )
    fig.text(
        0.5, 0.935,
        "Baseline (15-min quantile) vs 5-Min Horizon vs Classification (combined_trading loss)",
        fontsize=10, color=COLORS["light_grey"],
        ha="center", va="center",
    )

    # ---- Plot 1: Metric Comparison Bar Chart ----
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_metric_bars(ax1, results)

    # ---- Plot 2: Metrics Table ----
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_metrics_table(ax2, results)

    if has_predictions:
        # ---- Plot 3: Cumulative P&L Overlay ----
        ax3 = fig.add_subplot(gs[1, 0])
        _plot_cumulative_pnl_comparison(ax3, results)

        # ---- Plot 4: Rolling Directional Accuracy ----
        ax4 = fig.add_subplot(gs[1, 1])
        _plot_rolling_da_comparison(ax4, results)

        # ---- Plot 5: Return Distribution Comparison ----
        ax5 = fig.add_subplot(gs[2, 0])
        _plot_return_distributions(ax5, results)

        # ---- Plot 6: Strategy Return Scatter ----
        ax6 = fig.add_subplot(gs[2, 1])
        _plot_strategy_summary(ax6, results)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "model_comparison_dashboard.png"
    fig.savefig(str(png_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDashboard saved: {png_path}")
    return png_path


def _plot_metric_bars(ax, results):
    """Grouped bar chart comparing key metrics across models."""
    metrics_to_compare = [
        ("directional_accuracy_15", "Dir. Accuracy\n(final step)", 1.0),
        ("win_rate", "Win Rate", 1.0),
        ("sharpe_ratio", "Sharpe Ratio\n(annualized)", 1.0),
        ("profit_factor", "Profit Factor", 1.0),
    ]

    n_metrics = len(metrics_to_compare)
    n_models = len(results)
    x = np.arange(n_metrics)
    width = 0.22

    for i, result in enumerate(results):
        color = list(MODEL_COLORS.values())[i]
        values = []
        for key, _, scale in metrics_to_compare:
            val = result["metrics"].get(key, 0) * scale
            values.append(val)

        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=result["name"],
                      color=color, alpha=0.85, edgecolor="none", zorder=3)

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6,
                    color=color, fontweight="bold")

    # Reference lines
    ax.axhline(y=0.5, color=COLORS["grey"], linewidth=0.8, linestyle="--",
               alpha=0.5, label="50% DA baseline")
    ax.axhline(y=1.0, color=COLORS["grey"], linewidth=0.5, linestyle=":",
               alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics_to_compare], fontsize=8)
    ax.set_ylabel("Value", fontsize=9, color=COLORS["light_grey"])
    ax.set_title("Key Metrics Comparison", fontsize=12, color=COLORS["white"], pad=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.5)
    ax.tick_params(colors=COLORS["light_grey"], labelsize=7)


def _plot_metrics_table(ax, results):
    """Render a comparison table of all metrics."""
    ax.set_axis_off()
    ax.set_title("Full Metrics Comparison", fontsize=12, color=COLORS["white"], pad=10)

    metric_rows = [
        ("MSE", "mse", "{:.6f}", False),
        ("RMSE", "rmse", "{:.6f}", False),
        ("MAE", "mae", "{:.6f}", False),
        ("Dir. Acc (all steps)", "directional_accuracy_all_steps", "{:.4f}", True),
        ("Dir. Acc (final step)", "directional_accuracy_15", "{:.4f}", True),
        ("Win Rate", "win_rate", "{:.4f}", True),
        ("Sharpe Ratio", "sharpe_ratio", "{:.4f}", True),
        ("Total Return", "total_return", "{:.2f}", True),
        ("Max Drawdown", "max_drawdown", "{:.2f}", False),
        ("Profit Factor", "profit_factor", "{:.4f}", True),
    ]

    n_rows = len(metric_rows) + 1  # +1 for header
    row_height = 0.85 / n_rows
    col_x = [0.02, 0.32, 0.54, 0.78]

    # Header
    headers = ["Metric"] + [r["name"].split(" (")[0] if "(" in r["name"] else r["name"][:12]
                            for r in results]
    y = 0.92
    for c, (text, xpos) in enumerate(zip(headers, col_x)):
        color = COLORS["cyan"] if c == 0 else list(MODEL_COLORS.values())[c - 1]
        ax.text(xpos, y, text, transform=ax.transAxes, fontsize=8,
                fontweight="bold", color=color, va="center", ha="left",
                family="monospace")

    # Separator line
    ax.plot([0.01, 0.99], [y - row_height * 0.4, y - row_height * 0.4],
            color=COLORS["grid"], linewidth=0.5, transform=ax.transAxes)

    # Data rows
    for r, (label, key, fmt, higher_better) in enumerate(metric_rows):
        y = 0.92 - (r + 1) * row_height

        # Metric name
        ax.text(col_x[0], y, label, transform=ax.transAxes, fontsize=7,
                color=COLORS["light_grey"], va="center", ha="left",
                family="monospace")

        # Values for each model
        values = [res["metrics"].get(key, float("nan")) for res in results]
        if higher_better:
            best_idx = int(np.nanargmax(values))
        else:
            best_idx = int(np.nanargmin(values))

        for i, (val, xpos) in enumerate(zip(values, col_x[1:])):
            text = fmt.format(val) if not np.isnan(val) else "N/A"
            color = list(MODEL_COLORS.values())[i]
            fontweight = "bold" if i == best_idx else "normal"

            # Highlight best value
            if i == best_idx:
                text = f"*{text}"

            ax.text(xpos, y, text, transform=ax.transAxes, fontsize=7,
                    fontweight=fontweight, color=color, va="center", ha="left",
                    family="monospace")


def _plot_cumulative_pnl_comparison(ax, results):
    """Overlay cumulative P&L curves for all models."""
    for i, result in enumerate(results):
        color = list(MODEL_COLORS.values())[i]
        cum_pnl = result.get("cum_pnl")
        if cum_pnl is not None:
            x = np.arange(len(cum_pnl))
            ax.plot(x, cum_pnl, color=color, linewidth=1.2,
                    label=f"{result['name']} ({cum_pnl[-1]:+.1f})",
                    alpha=0.9, zorder=3 + i)

    ax.axhline(y=0, color=COLORS["dark_grey"], linewidth=0.6, zorder=1)
    ax.set_title("Cumulative P&L Comparison (Log Return)", fontsize=12,
                 color=COLORS["white"], pad=10)
    ax.set_xlabel("Trade Index", fontsize=9, color=COLORS["light_grey"])
    ax.set_ylabel("Cumulative Log Return", fontsize=9, color=COLORS["light_grey"])
    ax.legend(loc="upper left", fontsize=7, framealpha=0.5)
    ax.tick_params(colors=COLORS["light_grey"], labelsize=7)


def _plot_rolling_da_comparison(ax, results):
    """Rolling directional accuracy for all models."""
    for i, result in enumerate(results):
        color = list(MODEL_COLORS.values())[i]
        rolling_da = result.get("rolling_da")
        if rolling_da is not None:
            x = np.arange(len(rolling_da))
            ax.plot(x, rolling_da, color=color, linewidth=1.0,
                    label=result["name"], alpha=0.85, zorder=3 + i)

    ax.axhline(y=0.5, color=COLORS["grey"], linewidth=0.8, linestyle="--",
               alpha=0.6, label="50% baseline", zorder=2)

    ax.set_title("Rolling Directional Accuracy (500-bar window)", fontsize=12,
                 color=COLORS["white"], pad=10)
    ax.set_xlabel("Bar Index", fontsize=9, color=COLORS["light_grey"])
    ax.set_ylabel("Accuracy", fontsize=9, color=COLORS["light_grey"])
    ax.set_ylim(0.35, 0.65)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.5)
    ax.tick_params(colors=COLORS["light_grey"], labelsize=7)


def _plot_return_distributions(ax, results):
    """Overlaid histograms of strategy returns for all models."""
    bins = 80

    all_returns = []
    for result in results:
        sr = result.get("strategy_returns")
        if sr is not None:
            all_returns.append(sr)

    if all_returns:
        combined = np.concatenate(all_returns)
        lo, hi = np.percentile(combined, [1, 99])
        bin_edges = np.linspace(lo, hi, bins + 1)

        for i, result in enumerate(results):
            color = list(MODEL_COLORS.values())[i]
            sr = result.get("strategy_returns")
            if sr is not None:
                ax.hist(sr, bins=bin_edges, color=color, alpha=0.35,
                        density=True, label=result["name"], edgecolor="none",
                        zorder=2 + i)

                # KDE smoothing
                counts, edges = np.histogram(sr, bins=bin_edges, density=True)
                centers = (edges[:-1] + edges[1:]) / 2
                if len(counts) > 5:
                    kernel_size = 5
                    k = np.ones(kernel_size) / kernel_size
                    smoothed = np.convolve(counts, k, mode="same")
                else:
                    smoothed = counts
                ax.plot(centers, smoothed, color=color, linewidth=1.2,
                        alpha=0.9, zorder=5 + i)

    ax.set_title("Strategy Return Distributions", fontsize=12,
                 color=COLORS["white"], pad=10)
    ax.set_xlabel("Per-Trade Log Return", fontsize=9, color=COLORS["light_grey"])
    ax.set_ylabel("Density", fontsize=9, color=COLORS["light_grey"])
    ax.legend(loc="upper right", fontsize=7, framealpha=0.5)
    ax.tick_params(colors=COLORS["light_grey"], labelsize=7)


def _plot_strategy_summary(ax, results):
    """Summary text panel with key findings."""
    ax.set_axis_off()
    ax.set_title("Experiment Summary", fontsize=12, color=COLORS["white"], pad=10)

    lines = []
    lines.append("KEY FINDINGS")
    lines.append("=" * 45)
    lines.append("")

    # Sort by directional accuracy
    sorted_results = sorted(results,
                           key=lambda r: r["metrics"].get("directional_accuracy_15", 0),
                           reverse=True)

    lines.append("RANKING BY DIRECTIONAL ACCURACY:")
    for i, r in enumerate(sorted_results):
        da = r["metrics"].get("directional_accuracy_15", 0)
        sharpe = r["metrics"].get("sharpe_ratio", 0)
        medal = ["1st", "2nd", "3rd"][i] if i < 3 else f"{i+1}th"
        lines.append(f"  {medal}: {r['name']}")
        lines.append(f"        DA={da:.4f}  Sharpe={sharpe:.2f}")
    lines.append("")

    # Best model analysis
    best = sorted_results[0]
    best_da = best["metrics"].get("directional_accuracy_15", 0)
    lines.append(f"BEST MODEL: {best['name']}")
    lines.append(f"  Directional Accuracy: {best_da:.2%}")
    lines.append(f"  Sharpe Ratio: {best['metrics'].get('sharpe_ratio', 0):.2f}")
    lines.append(f"  Total Return: {best['metrics'].get('total_return', 0):.1f}")
    lines.append("")

    # Improvement over random
    lines.append("VS RANDOM BASELINE (50%):")
    for r in sorted_results:
        da = r["metrics"].get("directional_accuracy_15", 0)
        diff = da - 0.5
        lines.append(f"  {r['name'][:20]:20s}: {diff:+.2%}")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=7.5,
            color=COLORS["light_grey"], va="top", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=COLORS["bg"],
                      alpha=0.9, edgecolor=COLORS["grid"]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple TFT experiment results"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "figures" / "comparison"),
        help="Output directory for dashboard",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="DPI for PNG output",
    )
    parser.add_argument(
        "--from-files", action="store_true",
        help="Load metrics from evaluation text files (no inference)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("TFT Crypto Predictor - Multi-Model Comparison")
    print("=" * 60)

    if args.from_files:
        print("\nLoading metrics from evaluation files...")
        results = load_from_eval_files()
        for r in results:
            print(f"\n  {r['name']}:")
            for k, v in r["metrics"].items():
                print(f"    {k}: {v}")
    else:
        results = []
        for exp in EXPERIMENTS:
            ckpt_path = PROJECT_ROOT / exp["checkpoint"]
            if not ckpt_path.exists():
                print(f"\n  Warning: checkpoint not found: {ckpt_path}")
                print(f"  Skipping {exp['name']}")
                continue
            result = load_experiment_data(exp)
            results.append(result)

    if not results:
        print("\nNo results to compare!")
        return

    print(f"\n\nLoaded {len(results)} experiments. Generating dashboard...")
    png_path = create_comparison_dashboard(results, args.output_dir, dpi=args.dpi)

    # Print summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<30s}", end="")
    for r in results:
        print(f"{r['name']:>20s}", end="")
    print()
    print("-" * (30 + 20 * len(results)))

    for key, label in [
        ("directional_accuracy_15", "Dir. Accuracy (final)"),
        ("win_rate", "Win Rate"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("total_return", "Total Return"),
        ("max_drawdown", "Max Drawdown"),
        ("profit_factor", "Profit Factor"),
        ("mse", "MSE"),
        ("rmse", "RMSE"),
    ]:
        print(f"{label:<30s}", end="")
        for r in results:
            val = r["metrics"].get(key, float("nan"))
            print(f"{val:>20.4f}", end="")
        print()

    print(f"\nDashboard saved to: {png_path}")


if __name__ == "__main__":
    main()
