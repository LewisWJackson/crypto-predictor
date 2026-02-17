"""
Regime-gated trading strategy.

Adjusts position sizing, entry thresholds, and risk parameters based on
the detected market regime. Uses soft-blending via HMM posterior
probabilities to avoid whipsaw at regime transitions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


@dataclass
class RegimeConfig:
    """Strategy parameters for a single regime."""

    position_size: float    # Multiplier (0.0 = flat, 1.0 = full, 1.5 = leveraged)
    threshold_bps: float    # Minimum predicted return in basis points to enter
    stop_loss: float        # Stop loss as fraction (e.g. 0.003 = 0.3%)
    take_profit: float      # Take profit as fraction
    direction_bias: str     # "long_only", "short_only", or "both"


# Default regime configs matching the plan
DEFAULT_REGIME_CONFIGS: Dict[str, RegimeConfig] = {
    "accumulation": RegimeConfig(
        position_size=0.5,
        threshold_bps=2.0,
        stop_loss=0.003,
        take_profit=0.005,
        direction_bias="long_only",
    ),
    "markup": RegimeConfig(
        position_size=1.5,
        threshold_bps=0.0,
        stop_loss=0.008,
        take_profit=0.015,
        direction_bias="long_only",
    ),
    "distribution": RegimeConfig(
        position_size=0.25,
        threshold_bps=5.0,
        stop_loss=0.002,
        take_profit=0.003,
        direction_bias="both",
    ),
    "markdown": RegimeConfig(
        position_size=0.0,
        threshold_bps=0.0,
        stop_loss=0.0,
        take_profit=0.0,
        direction_bias="long_only",  # irrelevant — position_size is 0
    ),
}

REGIME_NAMES = ["accumulation", "markup", "distribution", "markdown"]

# Transition strategy configs — these override during transition states
DEFAULT_TRANSITION_CONFIGS: Dict[str, RegimeConfig] = {
    "accumulation_to_markup": RegimeConfig(
        position_size=2.0,       # Most aggressive — catching the breakout
        threshold_bps=0.0,       # Enter on any positive signal
        stop_loss=0.005,         # 0.5% — tight but room for volatility
        take_profit=0.02,        # 2.0% — let the breakout run
        direction_bias="long_only",
    ),
    "markup_to_distribution": RegimeConfig(
        position_size=0.5,       # Scale down — protect gains
        threshold_bps=3.0,       # Require conviction to enter new trades
        stop_loss=0.003,         # 0.3% — tighten stops
        take_profit=0.005,       # 0.5% — take profits quickly
        direction_bias="long_only",
    ),
    "distribution_to_markdown": RegimeConfig(
        position_size=0.0,       # Exit everything — breakdown imminent
        threshold_bps=0.0,
        stop_loss=0.0,
        take_profit=0.0,
        direction_bias="long_only",
    ),
    "markdown_to_accumulation": RegimeConfig(
        position_size=0.3,       # Small positions — bottoming but not confirmed
        threshold_bps=3.0,       # Require conviction
        stop_loss=0.004,         # 0.4%
        take_profit=0.008,       # 0.8%
        direction_bias="long_only",
    ),
}

# All config names (steady + transition)
ALL_CONFIG_NAMES = REGIME_NAMES + list(DEFAULT_TRANSITION_CONFIGS.keys())


def load_regime_configs(config_path: str | Path) -> Dict[str, RegimeConfig]:
    """Load regime + transition strategy configs from a YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    strategy = raw.get("strategy", {})
    configs = {}

    # Load steady-state configs
    for name in REGIME_NAMES:
        if name in strategy:
            s = strategy[name]
            configs[name] = RegimeConfig(
                position_size=s.get("position_size", 0.0),
                threshold_bps=s.get("threshold_bps", 0.0),
                stop_loss=s.get("stop_loss", 0.0),
                take_profit=s.get("take_profit", 0.0),
                direction_bias=s.get("direction_bias", "long_only"),
            )
        else:
            configs[name] = DEFAULT_REGIME_CONFIGS[name]

    # Load transition configs
    for name, default in DEFAULT_TRANSITION_CONFIGS.items():
        if name in strategy:
            s = strategy[name]
            configs[name] = RegimeConfig(
                position_size=s.get("position_size", default.position_size),
                threshold_bps=s.get("threshold_bps", default.threshold_bps),
                stop_loss=s.get("stop_loss", default.stop_loss),
                take_profit=s.get("take_profit", default.take_profit),
                direction_bias=s.get("direction_bias", default.direction_bias),
            )
        else:
            configs[name] = default

    return configs


def get_effective_config(
    state_name: str,
    configs: Dict[str, RegimeConfig] | None = None,
) -> RegimeConfig:
    """Get the strategy config for a given state (steady or transition).

    Args:
        state_name: Regime or transition name (e.g. "markup" or "accumulation_to_markup").
        configs: Full config dict. Uses defaults if None.

    Returns:
        The RegimeConfig for this state.
    """
    if configs is None:
        all_configs = {**DEFAULT_REGIME_CONFIGS, **DEFAULT_TRANSITION_CONFIGS}
    else:
        all_configs = configs

    return all_configs.get(state_name, DEFAULT_REGIME_CONFIGS.get("accumulation"))


def apply_regime_strategy(
    y_pred: np.ndarray,
    regime_probs: np.ndarray,
    configs: Dict[str, RegimeConfig] | None = None,
) -> Dict[str, np.ndarray]:
    """Apply regime-gated strategy using soft probability blending.

    Args:
        y_pred: Predicted returns, shape (N,) — final step predictions.
        regime_probs: Regime posterior probabilities, shape (N, 4).
        configs: Per-regime strategy configs. Uses defaults if None.

    Returns:
        Dict with:
            positions: (N,) effective position sizes (0 to ~1.5)
            thresholds: (N,) effective entry thresholds
            regime_position_sizes: (N, 4) per-regime position contributions
    """
    if configs is None:
        configs = DEFAULT_REGIME_CONFIGS

    n = len(y_pred)
    config_list = [configs[name] for name in REGIME_NAMES]

    # Build arrays from configs
    pos_sizes = np.array([c.position_size for c in config_list])   # (4,)
    thresholds = np.array([c.threshold_bps for c in config_list])  # (4,)

    # Soft-blend: effective parameter = sum(prob[i] * param[i])
    effective_position_size = regime_probs @ pos_sizes      # (N,)
    effective_threshold = regime_probs @ thresholds          # (N,)

    # Convert threshold from bps to return magnitude
    threshold_return = effective_threshold * 1e-4  # bps -> decimal

    # Apply: position = effective_size when pred > threshold, else 0
    signal = (y_pred > threshold_return).astype(float)
    positions = signal * effective_position_size

    return {
        "positions": positions,
        "effective_position_size": effective_position_size,
        "effective_threshold": effective_threshold,
        "threshold_return": threshold_return,
    }


def simulate_regime_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime_probs: np.ndarray,
    configs: Dict[str, RegimeConfig] | None = None,
) -> Dict:
    """Run a full backtest with the regime-gated strategy.

    Args:
        y_true: Actual returns, shape (N,) — final step.
        y_pred: Predicted returns, shape (N,) — final step.
        regime_probs: Regime posterior probs, shape (N, 4).
        configs: Per-regime strategy configs.

    Returns:
        Dict with strategy arrays and summary metrics.
    """
    result = apply_regime_strategy(y_pred, regime_probs, configs)
    positions = result["positions"]

    # Per-trade returns (position-weighted)
    strategy_returns = positions * y_true
    cumulative_pnl = np.cumsum(strategy_returns)

    # Metrics
    active_mask = positions > 0
    total_trades = int(np.sum(active_mask))
    wins = int(np.sum((strategy_returns > 0) & active_mask))
    losses = int(np.sum((strategy_returns < 0) & active_mask))

    win_rate = wins / total_trades if total_trades > 0 else 0.0

    # Sharpe ratio (annualized, 15-min bars -> ~35,040 bars/year)
    if total_trades > 1:
        trade_rets = strategy_returns[active_mask]
        mean_ret = np.mean(trade_rets)
        std_ret = np.std(trade_rets) + 1e-8
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

    # Buy & hold benchmark
    bh_cumulative = np.cumsum(y_true)

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
        "n_samples": len(y_true),
        "effective_position_size": result["effective_position_size"],
        "effective_threshold": result["effective_threshold"],
    }
