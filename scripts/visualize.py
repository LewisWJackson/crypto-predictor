#!/usr/bin/env python3
"""
Publication-quality evaluation visualizations for TFT Crypto Predictor.

Generates six chart types:
  1. Predicted vs Actual log returns with confidence intervals
  2. Directional accuracy over time (rolling window)
  3. Cumulative P&L curve with drawdown annotation
  4. Feature importance bar chart (top 20)
  5. Prediction distribution histogram
  6. Summary dashboard (all subplots combined)

Usage:
    # Demo mode with synthetic data (layout testing)
    python scripts/visualize.py --demo

    # From saved numpy arrays
    python scripts/visualize.py --predictions results/y_pred.npy --actuals results/y_true.npy

    # From a model checkpoint (runs inference on test set)
    python scripts/visualize.py --checkpoint models/tft/best.ckpt

    # Customize output
    python scripts/visualize.py --demo --output-dir figures/ --dpi 200
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so `from src.metrics import ...` works
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Color palette  (professional dark-background palette)
# ---------------------------------------------------------------------------
COLORS = {
    "cyan":        "#00d4ff",
    "magenta":     "#ff2e97",
    "green":       "#00ff88",
    "orange":      "#ff9f1c",
    "purple":      "#b388ff",
    "yellow":      "#ffe66d",
    "red":         "#ff4444",
    "grey":        "#888888",
    "light_grey":  "#bbbbbb",
    "dark_grey":   "#333333",
    "bg":          "#0e1117",
    "grid":        "#1e2330",
    "white":       "#e0e0e0",
}

# Quantile band alphas (outer to inner)
BAND_ALPHAS = [0.10, 0.18, 0.28]


# ---------------------------------------------------------------------------
# Synthetic data generator (--demo mode)
# ---------------------------------------------------------------------------
def generate_demo_data(
    n_samples: int = 2000,
    horizon: int = 15,
    seed: int = 42,
):
    """
    Generate realistic-looking synthetic predictions, actuals, quantiles,
    and feature importances for layout testing.

    Returns:
        dict with keys:
            y_true       (n_samples, horizon)
            y_pred       (n_samples, horizon)
            quantiles    (n_samples, horizon, 7)   -- 7 quantile levels
            feature_names  list[str]
            feature_importances  np.ndarray  (43,)
    """
    rng = np.random.RandomState(seed)

    # --- Actual returns: mean-reverting process with volatility clustering ---
    vol = np.zeros(n_samples)
    vol[0] = 0.001
    for i in range(1, n_samples):
        vol[i] = 0.0002 + 0.85 * vol[i - 1] + 0.10 * abs(rng.randn()) * 0.001
    vol = np.clip(vol, 0.0001, 0.01)

    base_returns = rng.randn(n_samples) * vol
    # Mild autocorrelation
    for i in range(1, n_samples):
        base_returns[i] += 0.05 * base_returns[i - 1]

    # Build multi-horizon actuals: cumulative sums with noise
    y_true = np.zeros((n_samples, horizon))
    for h in range(horizon):
        noise = rng.randn(n_samples) * vol * np.sqrt(h + 1)
        y_true[:, h] = np.cumsum(base_returns)[: n_samples] * (h + 1) / n_samples + noise

    # Rescale to realistic log-return magnitudes
    y_true *= 0.3

    # --- Predictions: actuals + noise, with model having ~57% directional acc ---
    pred_noise_scale = np.std(y_true) * 0.7
    y_pred = y_true + rng.randn(*y_true.shape) * pred_noise_scale
    # Bias toward correct direction to get ~57% accuracy
    sign_mask = np.sign(y_true) != np.sign(y_pred)
    flip_mask = sign_mask & (rng.rand(*y_true.shape) < 0.14)
    y_pred[flip_mask] *= -1

    # --- Quantile predictions (7 quantiles: 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98) ---
    quantile_levels = np.array([0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98])
    quantiles = np.zeros((n_samples, horizon, len(quantile_levels)))
    for qi, q in enumerate(quantile_levels):
        spread = (q - 0.5) * 2.0  # ranges from -0.96 to +0.96
        quantiles[:, :, qi] = y_pred + spread * pred_noise_scale * 1.2

    # --- Feature importances (43 features from architecture doc) ---
    feature_names = [
        # Price-derived (6)
        "log_return_1", "log_return_5", "log_return_15", "log_return_60",
        "high_low_range", "close_open_range",
        # Trend (8)
        "sma_20", "sma_50", "sma_200", "ema_9", "ema_21",
        "macd_line", "macd_signal", "macd_histogram",
        # Momentum (7)
        "rsi_14", "stoch_k", "stoch_d", "williams_r",
        "cci_20", "roc_10", "momentum_10",
        # Volatility (7)
        "bb_upper", "bb_lower", "bb_bandwidth", "bb_pctb",
        "atr_14", "rolling_volatility_20", "rolling_volatility_60",
        # Volume (6)
        "volume_sma_ratio", "obv_normalized", "vwap_distance",
        "mfi_14", "volume_roc", "ad_line_normalized",
        # Rolling stats (3)
        "rolling_skew_60", "rolling_kurtosis_60", "autocorr_lag1",
        # Temporal (6)
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "minute_of_hour_sin", "minute_of_hour_cos",
    ]
    # Importance: heavy-tailed distribution with a few dominant features
    raw_imp = rng.exponential(scale=1.0, size=len(feature_names))
    # Boost some features to look realistic
    boost_indices = [0, 2, 14, 26, 28]  # log_return_1, log_return_15, rsi_14, atr_14, vol_sma
    for idx in boost_indices:
        raw_imp[idx] *= rng.uniform(2.5, 4.0)
    feature_importances = raw_imp / raw_imp.sum()

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "quantiles": quantiles,
        "feature_names": feature_names,
        "feature_importances": feature_importances,
    }


# ---------------------------------------------------------------------------
# Plot 1: Predicted vs Actual
# ---------------------------------------------------------------------------
def plot_predicted_vs_actual(
    ax_full,
    ax_zoom,
    y_true_final,
    y_pred_final,
    quantiles_final=None,
    zoom_start=400,
    zoom_len=200,
):
    """
    Time series of actual vs predicted log returns (final horizon step).
    Top: full series. Bottom: zoomed 200-bar segment with confidence bands.

    Args:
        ax_full:  Axes for full time series
        ax_zoom:  Axes for zoomed segment
        y_true_final: 1-D array of actual returns (final step)
        y_pred_final: 1-D array of predicted returns (final step)
        quantiles_final: Optional (N, 7) quantile predictions for final step
        zoom_start: Start index for zoomed view
        zoom_len:  Number of bars in zoomed view
    """
    n = len(y_true_final)
    x = np.arange(n)

    # ---- Full series ----
    ax_full.plot(x, y_true_final, color=COLORS["cyan"], linewidth=0.5,
                 alpha=0.8, label="Actual", zorder=3)
    ax_full.plot(x, y_pred_final, color=COLORS["magenta"], linewidth=0.5,
                 alpha=0.7, label="Predicted", zorder=2)

    # Mark the zoom region
    zoom_end = min(zoom_start + zoom_len, n)
    ax_full.axvspan(zoom_start, zoom_end, alpha=0.12, color=COLORS["yellow"],
                    label="Zoom region")

    ax_full.set_title("Predicted vs Actual Log Returns (Step 15)", fontsize=11,
                      color=COLORS["white"], pad=8)
    ax_full.set_ylabel("Log Return", fontsize=9, color=COLORS["light_grey"])
    ax_full.legend(loc="upper right", fontsize=7, framealpha=0.4)
    ax_full.tick_params(colors=COLORS["light_grey"], labelsize=7)

    # ---- Zoomed segment ----
    zs, ze = zoom_start, zoom_end
    xz = np.arange(ze - zs)

    ax_zoom.plot(xz, y_true_final[zs:ze], color=COLORS["cyan"], linewidth=1.0,
                 alpha=0.9, label="Actual", zorder=4)
    ax_zoom.plot(xz, y_pred_final[zs:ze], color=COLORS["magenta"], linewidth=1.0,
                 alpha=0.85, label="Predicted", zorder=3)

    # Confidence bands from quantiles (if available)
    if quantiles_final is not None:
        # quantile indices: 0->0.02, 1->0.10, 2->0.25, 3->0.50, 4->0.75, 5->0.90, 6->0.98
        bands = [
            (0, 6, BAND_ALPHAS[0], "2%-98%"),   # widest
            (1, 5, BAND_ALPHAS[1], "10%-90%"),
            (2, 4, BAND_ALPHAS[2], "25%-75%"),   # narrowest
        ]
        for lo_idx, hi_idx, alpha, lbl in bands:
            ax_zoom.fill_between(
                xz,
                quantiles_final[zs:ze, lo_idx],
                quantiles_final[zs:ze, hi_idx],
                alpha=alpha,
                color=COLORS["magenta"],
                label=lbl,
                zorder=1,
            )

    ax_zoom.set_title(f"Zoomed View (bars {zs}-{ze})", fontsize=10,
                      color=COLORS["white"], pad=6)
    ax_zoom.set_xlabel("Bar Index", fontsize=9, color=COLORS["light_grey"])
    ax_zoom.set_ylabel("Log Return", fontsize=9, color=COLORS["light_grey"])
    ax_zoom.legend(loc="upper right", fontsize=6, framealpha=0.4, ncol=2)
    ax_zoom.tick_params(colors=COLORS["light_grey"], labelsize=7)


# ---------------------------------------------------------------------------
# Plot 2: Directional Accuracy Over Time
# ---------------------------------------------------------------------------
def plot_directional_accuracy(ax, y_true_final, y_pred_final, window=100):
    """
    Rolling directional accuracy with color-coded regions above/below 50%.

    Args:
        ax: Matplotlib Axes
        y_true_final: 1-D actual returns
        y_pred_final: 1-D predicted returns
        window: Rolling window size
    """
    correct = (np.sign(y_true_final) == np.sign(y_pred_final)).astype(float)
    n = len(correct)

    # Rolling mean via convolution
    kernel = np.ones(window) / window
    rolling_acc = np.convolve(correct, kernel, mode="valid")
    x = np.arange(len(rolling_acc)) + window // 2

    # Color regions above/below 50%
    above = rolling_acc >= 0.5
    below = ~above

    ax.fill_between(x, 0.5, rolling_acc, where=above,
                    color=COLORS["green"], alpha=0.25, interpolate=True)
    ax.fill_between(x, 0.5, rolling_acc, where=below,
                    color=COLORS["red"], alpha=0.25, interpolate=True)

    ax.plot(x, rolling_acc, color=COLORS["cyan"], linewidth=1.0, zorder=3)

    # 50% baseline
    ax.axhline(y=0.5, color=COLORS["grey"], linewidth=0.8, linestyle="--",
               alpha=0.7, label="50% baseline", zorder=2)

    # Overall accuracy annotation
    overall_acc = np.mean(correct)
    ax.axhline(y=overall_acc, color=COLORS["yellow"], linewidth=0.8,
               linestyle=":", alpha=0.8, label=f"Overall: {overall_acc:.1%}",
               zorder=2)

    ax.set_title(f"Directional Accuracy (Rolling {window}-bar Window)", fontsize=11,
                 color=COLORS["white"], pad=8)
    ax.set_xlabel("Bar Index", fontsize=9, color=COLORS["light_grey"])
    ax.set_ylabel("Accuracy", fontsize=9, color=COLORS["light_grey"])
    ax.set_ylim(0.30, 0.75)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.4)
    ax.tick_params(colors=COLORS["light_grey"], labelsize=7)


# ---------------------------------------------------------------------------
# Plot 3: Cumulative P&L Curve
# ---------------------------------------------------------------------------
def plot_cumulative_pnl(ax, y_true_final, y_pred_final, seed=42):
    """
    Cumulative P&L for model strategy, random baseline, and buy-and-hold.
    Annotates the maximum drawdown on the model curve.

    Args:
        ax: Matplotlib Axes
        y_true_final: 1-D actual returns
        y_pred_final: 1-D predicted returns
        seed: Random seed for the random baseline
    """
    n = len(y_true_final)

    # --- Model strategy: position = sign(prediction) ---
    positions_model = np.sign(y_pred_final)
    returns_model = positions_model * y_true_final
    cum_model = np.cumsum(returns_model)

    # --- Random baseline ---
    rng = np.random.RandomState(seed)
    positions_random = rng.choice([-1.0, 1.0], size=n)
    returns_random = positions_random * y_true_final
    cum_random = np.cumsum(returns_random)

    # --- Buy-and-hold ---
    cum_bah = np.cumsum(y_true_final)

    x = np.arange(n)

    # Plot curves
    ax.plot(x, cum_model, color=COLORS["cyan"], linewidth=1.2,
            label="TFT Model", zorder=4)
    ax.plot(x, cum_random, color=COLORS["grey"], linewidth=0.8,
            alpha=0.7, linestyle="--", label="Random Baseline", zorder=2)
    ax.plot(x, cum_bah, color=COLORS["orange"], linewidth=0.8,
            alpha=0.8, linestyle="-.", label="Buy & Hold", zorder=3)

    # Zero line
    ax.axhline(y=0, color=COLORS["dark_grey"], linewidth=0.6, zorder=1)

    # --- Mark maximum drawdown on model curve ---
    running_max = np.maximum.accumulate(cum_model)
    drawdowns = running_max - cum_model
    max_dd_idx = int(np.argmax(drawdowns))
    max_dd_val = drawdowns[max_dd_idx]

    # Find the peak before this trough
    peak_idx = int(np.argmax(cum_model[:max_dd_idx + 1]))

    # Shade the drawdown region
    ax.fill_between(
        x[peak_idx:max_dd_idx + 1],
        cum_model[peak_idx:max_dd_idx + 1],
        running_max[peak_idx:max_dd_idx + 1],
        color=COLORS["red"], alpha=0.25, zorder=2,
    )
    # Arrow annotation
    mid_dd = (peak_idx + max_dd_idx) // 2
    ax.annotate(
        f"Max DD: {max_dd_val:.4f}",
        xy=(max_dd_idx, cum_model[max_dd_idx]),
        xytext=(max_dd_idx + n * 0.05, cum_model[max_dd_idx] - max_dd_val * 0.5),
        fontsize=7,
        color=COLORS["red"],
        arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=0.8),
        zorder=5,
    )

    # Final return annotations
    for label, cum, color in [
        ("TFT", cum_model, COLORS["cyan"]),
        ("Random", cum_random, COLORS["grey"]),
        ("B&H", cum_bah, COLORS["orange"]),
    ]:
        ax.annotate(
            f"{cum[-1]:+.4f}",
            xy=(n - 1, cum[-1]),
            fontsize=6,
            color=color,
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
        )

    ax.set_title("Cumulative P&L (Log Return)", fontsize=11,
                 color=COLORS["white"], pad=8)
    ax.set_xlabel("Trade Index", fontsize=9, color=COLORS["light_grey"])
    ax.set_ylabel("Cumulative Log Return", fontsize=9, color=COLORS["light_grey"])
    ax.legend(loc="upper left", fontsize=7, framealpha=0.4)
    ax.tick_params(colors=COLORS["light_grey"], labelsize=7)


# ---------------------------------------------------------------------------
# Plot 4: Feature Importance
# ---------------------------------------------------------------------------
def plot_feature_importance(ax, feature_names, importances, top_n=20):
    """
    Horizontal bar chart of TFT variable importance (top N features).

    Args:
        ax: Matplotlib Axes
        feature_names: list of feature name strings
        importances: array of importance scores (same length as feature_names)
        top_n: Number of top features to display
    """
    import matplotlib.pyplot as plt

    # Sort by importance descending, take top N
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    # Reverse for horizontal bar chart (top feature at top)
    sorted_idx = sorted_idx[::-1]

    names = [feature_names[i] for i in sorted_idx]
    vals = importances[sorted_idx]

    # Color gradient from dim to bright
    norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
    cmap = plt.colormaps.get_cmap("cool")
    bar_colors = [cmap(0.3 + 0.7 * v) for v in norm_vals]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, vals, color=bar_colors, height=0.7, edgecolor="none",
                   zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=6.5, color=COLORS["light_grey"])
    ax.set_xlabel("Importance Score", fontsize=9, color=COLORS["light_grey"])
    ax.set_title(f"TFT Variable Importance (Top {top_n})", fontsize=11,
                 color=COLORS["white"], pad=8)
    ax.tick_params(colors=COLORS["light_grey"], labelsize=7)

    # Value labels on bars
    max_val = vals.max()
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=5.5,
            color=COLORS["light_grey"],
        )

    ax.set_xlim(0, max_val * 1.15)


# ---------------------------------------------------------------------------
# Plot 5: Prediction Distribution
# ---------------------------------------------------------------------------
def plot_prediction_distribution(ax, y_true_final, y_pred_final, bins=80):
    """
    Overlaid histograms of actual vs predicted return distributions.

    Args:
        ax: Matplotlib Axes
        y_true_final: 1-D actual returns
        y_pred_final: 1-D predicted returns
        bins: Number of histogram bins
    """
    # Compute common bin edges
    all_vals = np.concatenate([y_true_final, y_pred_final])
    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    bin_edges = np.linspace(lo, hi, bins + 1)

    ax.hist(y_true_final, bins=bin_edges, color=COLORS["cyan"], alpha=0.5,
            density=True, label="Actual", edgecolor="none", zorder=2)
    ax.hist(y_pred_final, bins=bin_edges, color=COLORS["magenta"], alpha=0.5,
            density=True, label="Predicted", edgecolor="none", zorder=3)

    # KDE-like smooth overlay using numpy histogram + interpolation
    for arr, color, lbl in [
        (y_true_final, COLORS["cyan"], None),
        (y_pred_final, COLORS["magenta"], None),
    ]:
        counts, edges = np.histogram(arr, bins=bin_edges, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        # Simple moving average smoothing
        if len(counts) > 5:
            kernel = np.ones(5) / 5
            smoothed = np.convolve(counts, kernel, mode="same")
        else:
            smoothed = counts
        ax.plot(centers, smoothed, color=color, linewidth=1.2, alpha=0.9, zorder=4)

    # Statistics annotation
    stats_text = (
        f"Actual:  \u03bc={np.mean(y_true_final):.5f}  \u03c3={np.std(y_true_final):.5f}\n"
        f"Pred:    \u03bc={np.mean(y_pred_final):.5f}  \u03c3={np.std(y_pred_final):.5f}"
    )
    ax.text(
        0.97, 0.95, stats_text,
        transform=ax.transAxes, fontsize=6.5, color=COLORS["light_grey"],
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["bg"], alpha=0.8,
                  edgecolor=COLORS["dark_grey"]),
        family="monospace",
    )

    ax.set_title("Return Distribution: Actual vs Predicted", fontsize=11,
                 color=COLORS["white"], pad=8)
    ax.set_xlabel("Log Return", fontsize=9, color=COLORS["light_grey"])
    ax.set_ylabel("Density", fontsize=9, color=COLORS["light_grey"])
    ax.legend(loc="upper left", fontsize=7, framealpha=0.4)
    ax.tick_params(colors=COLORS["light_grey"], labelsize=7)


# ---------------------------------------------------------------------------
# Plot 6: Summary Dashboard (single figure, 6 subplots)
# ---------------------------------------------------------------------------
def create_summary_dashboard(data, output_dir, dpi=150):
    """
    Create a single publication-quality figure with 6 subplots.
    Saves as both PNG and PDF.

    Args:
        data: dict from generate_demo_data() or loaded results
        output_dir: Path to save figures
        dpi: Resolution for PNG output
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.style.use("dark_background")

    # Override some rcParams for polish
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
        "font.size":         8,
        "axes.labelcolor":   COLORS["light_grey"],
        "xtick.color":       COLORS["light_grey"],
        "ytick.color":       COLORS["light_grey"],
        "text.color":        COLORS["white"],
        "legend.facecolor":  COLORS["bg"],
        "legend.edgecolor":  COLORS["dark_grey"],
    })

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_true_final = y_true[:, -1]
    y_pred_final = y_pred[:, -1]
    quantiles = data.get("quantiles")
    quantiles_final = quantiles[:, -1, :] if quantiles is not None else None
    feature_names = data.get("feature_names")
    feature_importances = data.get("feature_importances")

    # Compute metrics for the title bar
    metrics = compute_metrics(y_true, y_pred)

    # ---- Create figure with gridspec ----
    fig = plt.figure(figsize=(20, 14))

    # Top banner for metrics summary
    gs_top = gridspec.GridSpec(1, 1, figure=fig,
                               left=0.04, right=0.96, top=0.98, bottom=0.93)
    ax_banner = fig.add_subplot(gs_top[0])
    ax_banner.set_axis_off()

    banner_text = (
        f"TFT Crypto Predictor  |  "
        f"DA: {metrics['directional_accuracy_15']:.1%}   "
        f"Win Rate: {metrics['win_rate']:.1%}   "
        f"Sharpe: {metrics['sharpe_ratio']:.2f}   "
        f"Profit Factor: {metrics['profit_factor']:.2f}   "
        f"Max DD: {metrics['max_drawdown']:.4f}   "
        f"RMSE: {metrics['rmse']:.6f}"
    )
    ax_banner.text(
        0.5, 0.5, banner_text,
        transform=ax_banner.transAxes,
        fontsize=11, fontweight="bold",
        color=COLORS["cyan"],
        ha="center", va="center",
        family="monospace",
    )

    # Main grid: 3 rows x 2 cols
    gs_main = gridspec.GridSpec(3, 2, figure=fig,
                                left=0.06, right=0.96,
                                top=0.91, bottom=0.04,
                                hspace=0.38, wspace=0.28)

    # --- Subplot 1 & 2: Predicted vs Actual (full + zoom) ---
    gs_sub1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0, 0],
                                               hspace=0.35)
    ax_pva_full = fig.add_subplot(gs_sub1[0])
    ax_pva_zoom = fig.add_subplot(gs_sub1[1])

    zoom_start = min(400, max(0, len(y_true_final) - 250))
    plot_predicted_vs_actual(
        ax_pva_full, ax_pva_zoom,
        y_true_final, y_pred_final,
        quantiles_final=quantiles_final,
        zoom_start=zoom_start,
        zoom_len=200,
    )

    # --- Subplot 3: Directional Accuracy ---
    ax_da = fig.add_subplot(gs_main[0, 1])
    plot_directional_accuracy(ax_da, y_true_final, y_pred_final, window=100)

    # --- Subplot 4: Cumulative P&L ---
    ax_pnl = fig.add_subplot(gs_main[1, 0])
    plot_cumulative_pnl(ax_pnl, y_true_final, y_pred_final)

    # --- Subplot 5: Feature Importance ---
    ax_fi = fig.add_subplot(gs_main[1, 1])
    if feature_names is not None and feature_importances is not None:
        plot_feature_importance(ax_fi, feature_names, feature_importances, top_n=20)
    else:
        ax_fi.text(0.5, 0.5, "Feature importance data not available",
                   transform=ax_fi.transAxes, ha="center", va="center",
                   fontsize=10, color=COLORS["grey"])
        ax_fi.set_title("TFT Variable Importance", fontsize=11,
                        color=COLORS["white"], pad=8)

    # --- Subplot 6: Prediction Distribution ---
    ax_dist = fig.add_subplot(gs_main[2, 0])
    plot_prediction_distribution(ax_dist, y_true_final, y_pred_final)

    # --- Subplot 7 (bottom-right): Metrics table ---
    ax_table = fig.add_subplot(gs_main[2, 1])
    ax_table.set_axis_off()
    _draw_metrics_table(ax_table, metrics)

    # ---- Save ----
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "evaluation_dashboard.png"
    pdf_path = output_dir / "evaluation_dashboard.pdf"

    fig.savefig(str(png_path), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)

    print(f"Dashboard saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")

    return png_path, pdf_path


def _draw_metrics_table(ax, metrics):
    """Render a styled metrics summary table in the given axes."""
    rows = [
        ("Metric", "Value", "Threshold", "Status"),
        ("Directional Acc (step 15)",
         f"{metrics['directional_accuracy_15']:.2%}",
         "> 52%",
         "PASS" if metrics['directional_accuracy_15'] > 0.52 else "FAIL"),
        ("Directional Acc (all steps)",
         f"{metrics['directional_accuracy_all_steps']:.2%}",
         "--",
         "--"),
        ("Win Rate",
         f"{metrics['win_rate']:.2%}",
         "> 51%",
         "PASS" if metrics['win_rate'] > 0.51 else "FAIL"),
        ("Sharpe Ratio",
         f"{metrics['sharpe_ratio']:.2f}",
         "> 0.50",
         "PASS" if metrics['sharpe_ratio'] > 0.5 else "FAIL"),
        ("Profit Factor",
         f"{metrics['profit_factor']:.2f}",
         "> 1.10",
         "PASS" if metrics['profit_factor'] > 1.1 else "FAIL"),
        ("Max Drawdown",
         f"{metrics['max_drawdown']:.4f}",
         "< 0.20",
         "PASS" if metrics['max_drawdown'] < 0.20 else "FAIL"),
        ("MSE",
         f"{metrics['mse']:.6f}",
         "--",
         "--"),
        ("RMSE",
         f"{metrics['rmse']:.6f}",
         "--",
         "--"),
        ("MAE",
         f"{metrics['mae']:.6f}",
         "--",
         "--"),
        ("Total Return",
         f"{metrics['total_return']:.4f}",
         "--",
         "--"),
    ]

    ax.set_title("Evaluation Metrics Summary", fontsize=11,
                 color=COLORS["white"], pad=8)

    n_rows = len(rows)
    row_height = 1.0 / (n_rows + 1)
    col_positions = [0.02, 0.45, 0.68, 0.88]  # x positions for 4 columns

    for r, row in enumerate(rows):
        y = 1.0 - (r + 1) * row_height
        is_header = (r == 0)

        for c, (text, xpos) in enumerate(zip(row, col_positions)):
            fontweight = "bold" if is_header else "normal"
            fontsize = 7.5 if is_header else 7

            # Color the status column
            if c == 3 and not is_header:
                if text == "PASS":
                    color = COLORS["green"]
                elif text == "FAIL":
                    color = COLORS["red"]
                else:
                    color = COLORS["grey"]
            elif is_header:
                color = COLORS["cyan"]
            else:
                color = COLORS["light_grey"]

            ax.text(
                xpos, y, text,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight=fontweight,
                color=color,
                va="center", ha="left",
                family="monospace",
            )

        # Separator line after header
        if is_header:
            ax.plot(
                [0.01, 0.99],
                [y - row_height * 0.4, y - row_height * 0.4],
                color=COLORS["grid"], linewidth=0.5,
                transform=ax.transAxes, clip_on=False,
            )


# ---------------------------------------------------------------------------
# Individual chart export functions
# ---------------------------------------------------------------------------
def save_individual_charts(data, output_dir, dpi=150):
    """
    Save each chart type as a standalone PNG for flexibility.

    Args:
        data: dict from generate_demo_data() or loaded results
        output_dir: Path to save figures
        dpi: Resolution
    """
    import matplotlib.pyplot as plt

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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_true_final = y_true[:, -1]
    y_pred_final = y_pred[:, -1]
    quantiles = data.get("quantiles")
    quantiles_final = quantiles[:, -1, :] if quantiles is not None else None

    saved = []

    # 1. Predicted vs Actual
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
    zoom_start = min(400, max(0, len(y_true_final) - 250))
    plot_predicted_vs_actual(ax1, ax2, y_true_final, y_pred_final,
                            quantiles_final=quantiles_final,
                            zoom_start=zoom_start, zoom_len=200)
    fig.tight_layout(pad=1.5)
    p = output_dir / "01_predicted_vs_actual.png"
    fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    print(f"  Saved: {p}")

    # 2. Directional Accuracy
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_directional_accuracy(ax, y_true_final, y_pred_final, window=100)
    fig.tight_layout(pad=1.5)
    p = output_dir / "02_directional_accuracy.png"
    fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    print(f"  Saved: {p}")

    # 3. Cumulative P&L
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_cumulative_pnl(ax, y_true_final, y_pred_final)
    fig.tight_layout(pad=1.5)
    p = output_dir / "03_cumulative_pnl.png"
    fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    print(f"  Saved: {p}")

    # 4. Feature Importance
    feature_names = data.get("feature_names")
    feature_importances = data.get("feature_importances")
    if feature_names is not None and feature_importances is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_feature_importance(ax, feature_names, feature_importances, top_n=20)
        fig.tight_layout(pad=1.5)
        p = output_dir / "04_feature_importance.png"
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
        print(f"  Saved: {p}")

    # 5. Prediction Distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_prediction_distribution(ax, y_true_final, y_pred_final)
    fig.tight_layout(pad=1.5)
    p = output_dir / "05_prediction_distribution.png"
    fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    print(f"  Saved: {p}")

    return saved


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_numpy_data(pred_path, actual_path, quantile_path=None,
                    importance_path=None):
    """
    Load evaluation data from saved numpy arrays.

    Expected shapes:
        y_pred:              (N, 15) or (N,)
        y_true:              (N, 15) or (N,)
        quantiles (optional): (N, 15, 7)
        importances (optional): dict with 'names' and 'scores' keys in .npz

    Returns:
        data dict compatible with plotting functions
    """
    y_pred = np.load(pred_path)
    y_true = np.load(actual_path)

    # Ensure 2D
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]

    # Align lengths
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    data = {
        "y_true": y_true,
        "y_pred": y_pred,
        "quantiles": None,
        "feature_names": None,
        "feature_importances": None,
    }

    if quantile_path and Path(quantile_path).exists():
        data["quantiles"] = np.load(quantile_path)[:min_len]

    if importance_path and Path(importance_path).exists():
        imp_data = np.load(importance_path, allow_pickle=True)
        if isinstance(imp_data, np.lib.npyio.NpzFile):
            data["feature_names"] = list(imp_data["names"])
            data["feature_importances"] = imp_data["scores"]
        else:
            # Single array, use default feature names
            data["feature_importances"] = imp_data
            data["feature_names"] = [f"feature_{i}" for i in range(len(imp_data))]

    return data


def load_checkpoint_data(checkpoint_path, config_path=None):
    """
    Load a trained model, run inference on the test set, and extract
    predictions, actuals, quantiles, and feature importances.

    Requires pytorch-forecasting and pytorch-lightning.

    Args:
        checkpoint_path: Path to .ckpt model file
        config_path: Path to config YAML (defaults to configs/tft_config.yaml)

    Returns:
        data dict compatible with plotting functions
    """
    from src.dataset import create_datasets, create_dataloaders
    from src.model import load_config, load_trained_model

    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "tft_config.yaml")

    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_trained_model(checkpoint_path)

    print("Creating test dataset...")
    training_dataset, validation_dataset, test_dataset = create_datasets(config)
    dataset_cfg = config.get("dataset", {})
    _, _, test_dataloader = create_dataloaders(
        training_dataset, validation_dataset, test_dataset,
        batch_size=dataset_cfg.get("batch_size", 64) * 2,
        num_workers=dataset_cfg.get("num_workers", 4),
    )

    print("Running predictions...")
    # Get raw output (includes quantiles if model was trained with QuantileLoss)
    raw_predictions = model.predict(
        test_dataloader, mode="raw", return_x=True
    )
    predictions_point = model.predict(
        test_dataloader, mode="prediction", return_x=False
    )

    y_pred = predictions_point.numpy()

    # Extract quantile predictions if available
    quantiles = None
    if isinstance(raw_predictions, tuple):
        raw_out = raw_predictions[0]
        if hasattr(raw_out, "prediction"):
            # pytorch-forecasting raw output has .prediction with shape (N, horizon, n_quantiles)
            quantiles = raw_out.prediction.numpy()
        elif isinstance(raw_out, dict) and "prediction" in raw_out:
            quantiles = raw_out["prediction"].numpy()

    # Collect actuals
    actuals = []
    for batch in test_dataloader:
        x, y = batch
        if isinstance(y, (tuple, list)):
            actuals.append(y[0].numpy())
        else:
            actuals.append(y.numpy())
    y_true = np.concatenate(actuals, axis=0)

    # Align
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    if quantiles is not None:
        quantiles = quantiles[:min_len]

    # Ensure 2D
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]

    # Feature importance from TFT interpretation
    feature_names = None
    feature_importances = None
    try:
        print("Extracting feature importances...")
        interpretation = model.interpret_output(
            raw_predictions[0] if isinstance(raw_predictions, tuple) else raw_predictions,
            reduction="mean",
        )
        # TFT returns dict with 'encoder_variables' and 'decoder_variables'
        if "encoder_variables" in interpretation:
            enc_imp = interpretation["encoder_variables"]
            # enc_imp is a dict of variable_name -> importance_score
            if isinstance(enc_imp, dict):
                feature_names = list(enc_imp.keys())
                feature_importances = np.array(list(enc_imp.values()))
            else:
                # It may be a tensor; get names from dataset
                dataset_vars = (
                    list(training_dataset.reals)
                    if hasattr(training_dataset, "reals")
                    else [f"feature_{i}" for i in range(len(enc_imp))]
                )
                feature_names = dataset_vars[:len(enc_imp)]
                feature_importances = (
                    enc_imp.numpy() if hasattr(enc_imp, "numpy") else np.array(enc_imp)
                )
    except Exception as e:
        print(f"  Warning: Could not extract feature importances: {e}")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "quantiles": quantiles,
        "feature_names": feature_names,
        "feature_importances": feature_importances,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality evaluation charts for TFT Crypto Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with synthetic data (layout testing)
  python scripts/visualize.py --demo

  # From saved numpy arrays
  python scripts/visualize.py --predictions results/y_pred.npy --actuals results/y_true.npy

  # From model checkpoint
  python scripts/visualize.py --checkpoint models/tft/best.ckpt

  # Customize output
  python scripts/visualize.py --demo --output-dir figures/ --dpi 200 --individual
        """,
    )

    # --- Data source (mutually exclusive) ---
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--demo",
        action="store_true",
        help="Generate with synthetic data for layout testing",
    )
    source.add_argument(
        "--checkpoint",
        type=str,
        help="Path to trained model checkpoint (.ckpt) for inference",
    )
    source.add_argument(
        "--predictions",
        type=str,
        help="Path to predictions .npy file (requires --actuals)",
    )

    # --- Additional data files ---
    parser.add_argument(
        "--actuals",
        type=str,
        help="Path to actuals .npy file (required with --predictions)",
    )
    parser.add_argument(
        "--quantiles",
        type=str,
        default=None,
        help="Path to quantile predictions .npy file (optional)",
    )
    parser.add_argument(
        "--importances",
        type=str,
        default=None,
        help="Path to feature importances .npz file (optional, keys: 'names', 'scores')",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (used with --checkpoint, defaults to configs/tft_config.yaml)",
    )

    # --- Output options ---
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "figures"),
        help="Directory to save output figures (default: figures/)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for PNG output (default: 150)",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Also save each chart as a standalone PNG",
    )
    parser.add_argument(
        "--demo-samples",
        type=int,
        default=2000,
        help="Number of samples for demo data (default: 2000)",
    )

    args = parser.parse_args()

    # Validate
    if args.predictions and not args.actuals:
        parser.error("--actuals is required when using --predictions")

    return args


def main():
    args = parse_args()

    print("=" * 60)
    print("TFT Crypto Predictor - Evaluation Visualizer")
    print("=" * 60)

    # --- Load or generate data ---
    if args.demo:
        print(f"\nGenerating synthetic demo data ({args.demo_samples} samples)...")
        data = generate_demo_data(n_samples=args.demo_samples)
        print(f"  y_true shape: {data['y_true'].shape}")
        print(f"  y_pred shape: {data['y_pred'].shape}")
        print(f"  quantiles shape: {data['quantiles'].shape}")
        print(f"  features: {len(data['feature_names'])}")

    elif args.checkpoint:
        print(f"\nLoading from checkpoint: {args.checkpoint}")
        data = load_checkpoint_data(args.checkpoint, args.config)
        print(f"  y_true shape: {data['y_true'].shape}")
        print(f"  y_pred shape: {data['y_pred'].shape}")
        if data["quantiles"] is not None:
            print(f"  quantiles shape: {data['quantiles'].shape}")
        if data["feature_names"] is not None:
            print(f"  features: {len(data['feature_names'])}")

    elif args.predictions:
        print(f"\nLoading from numpy arrays:")
        print(f"  predictions: {args.predictions}")
        print(f"  actuals:     {args.actuals}")
        data = load_numpy_data(
            pred_path=args.predictions,
            actual_path=args.actuals,
            quantile_path=args.quantiles,
            importance_path=args.importances,
        )
        print(f"  y_true shape: {data['y_true'].shape}")
        print(f"  y_pred shape: {data['y_pred'].shape}")

    # --- Compute and print metrics ---
    metrics = compute_metrics(data["y_true"], data["y_pred"])
    print(f"\nKey Metrics:")
    print(f"  Directional Accuracy (step 15): {metrics['directional_accuracy_15']:.2%}")
    print(f"  Win Rate:                       {metrics['win_rate']:.2%}")
    print(f"  Sharpe Ratio:                   {metrics['sharpe_ratio']:.2f}")
    print(f"  Profit Factor:                  {metrics['profit_factor']:.2f}")
    print(f"  Max Drawdown:                   {metrics['max_drawdown']:.4f}")
    print(f"  RMSE:                           {metrics['rmse']:.6f}")

    # --- Generate dashboard ---
    print(f"\nGenerating dashboard (dpi={args.dpi})...")
    png_path, pdf_path = create_summary_dashboard(
        data, output_dir=args.output_dir, dpi=args.dpi,
    )

    # --- Individual charts ---
    if args.individual:
        print("\nGenerating individual charts...")
        individual_dir = Path(args.output_dir) / "individual"
        saved = save_individual_charts(data, individual_dir, dpi=args.dpi)
        print(f"  {len(saved)} charts saved to {individual_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
