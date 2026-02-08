"""
Evaluation metrics and baseline models for crypto price prediction.

Provides:
- compute_metrics(): Full evaluation metric suite
- Baseline models: random, persistence, MA crossover
"""

import numpy as np
from typing import Dict


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics for multi-horizon predictions.

    Args:
        y_true: Actual log returns, shape (N, 15) — N samples, 15-step horizon
        y_pred: Predicted log returns, shape (N, 15)

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}

    # --- Regression Metrics (on the 15th step, i.e., final prediction) ---
    y_true_final = y_true[:, -1]
    y_pred_final = y_pred[:, -1]

    # MSE
    metrics["mse"] = float(np.mean((y_true_final - y_pred_final) ** 2))

    # RMSE
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))

    # MAE
    metrics["mae"] = float(np.mean(np.abs(y_true_final - y_pred_final)))

    # --- Directional Accuracy (most important for trading) ---
    # Across all 15 time steps
    direction_correct_all = (np.sign(y_true) == np.sign(y_pred)).astype(float)
    metrics["directional_accuracy_all_steps"] = float(np.mean(direction_correct_all))

    # For the final step only
    direction_correct_final = (
        np.sign(y_true_final) == np.sign(y_pred_final)
    ).astype(float)
    metrics["directional_accuracy_15"] = float(np.mean(direction_correct_final))

    # --- Simulated Trading Metrics ---
    # Strategy: go long if predicted return > 0, short if < 0
    # Position size = 1 (fixed)
    positions = np.sign(y_pred_final)  # +1 or -1
    strategy_returns = positions * y_true_final  # Per-trade return

    # Win rate
    wins = (strategy_returns > 0).sum()
    total_trades = len(strategy_returns)
    metrics["win_rate"] = float(wins / total_trades) if total_trades > 0 else 0.0

    # Sharpe ratio (annualized)
    # Each prediction covers 15 mins -> ~35,040 predictions/year
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns) + 1e-8
    metrics["sharpe_ratio"] = float(mean_return / std_return * np.sqrt(35040))

    # Cumulative return
    cumulative = np.cumsum(strategy_returns)
    metrics["total_return"] = float(np.sum(strategy_returns))

    # Maximum drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    metrics["max_drawdown"] = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Profit factor
    gross_profit = np.sum(strategy_returns[strategy_returns > 0])
    gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0])) + 1e-8
    metrics["profit_factor"] = float(gross_profit / gross_loss)

    return metrics


# ---------------------------------------------------------------------------
# Baseline Models
# ---------------------------------------------------------------------------

class RandomPredictor:
    """Baseline: random directional predictions.

    Predicts +1 or -1 uniformly at random for each step.
    Expected directional accuracy: ~50%, Sharpe: ~0.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def predict(self, n_samples: int, horizon: int = 15) -> np.ndarray:
        """Generate random predictions.

        Args:
            n_samples: Number of prediction samples
            horizon: Prediction horizon (default 15)

        Returns:
            Predictions of shape (n_samples, horizon) with values in {-1, +1}
        """
        return self.rng.choice([-1.0, 1.0], size=(n_samples, horizon)).astype(
            np.float32
        )


class PersistenceModel:
    """Baseline: predict that the next return equals the last observed return.

    If the last observed 1-bar return was +0.001, predict +0.001 for all
    future steps. This is the simplest "naive" forecast.
    """

    def predict(
        self, last_observed_returns: np.ndarray, horizon: int = 15
    ) -> np.ndarray:
        """Generate persistence predictions.

        Args:
            last_observed_returns: Last observed 1-bar log return for each
                sample, shape (n_samples,)
            horizon: Prediction horizon (default 15)

        Returns:
            Predictions of shape (n_samples, horizon), repeating the last
            observed return across all steps.
        """
        return np.tile(
            last_observed_returns[:, np.newaxis], (1, horizon)
        ).astype(np.float32)


class MACrossoverBaseline:
    """Baseline: Simple SMA(9)/SMA(21) crossover signal.

    Computes fast and slow SMAs on historical close prices.
    Signal = +1 when fast > slow (bullish), -1 when fast < slow (bearish).
    """

    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def predict(
        self, close_prices: np.ndarray, horizon: int = 15
    ) -> np.ndarray:
        """Generate MA crossover predictions.

        Args:
            close_prices: Historical close prices, shape (n_samples, lookback)
                where lookback >= slow_period. Each row is the price history
                leading up to that prediction point.
            horizon: Prediction horizon (default 15)

        Returns:
            Predictions of shape (n_samples, horizon) with values in {-1, +1}
        """
        n_samples = close_prices.shape[0]
        predictions = np.zeros((n_samples, horizon), dtype=np.float32)

        for i in range(n_samples):
            prices = close_prices[i]
            # Compute SMAs from the end of the lookback window
            fast_sma = np.mean(prices[-self.fast_period :])
            slow_sma = np.mean(prices[-self.slow_period :])

            # Signal: +1 if fast > slow (bullish), -1 otherwise
            signal = 1.0 if fast_sma > slow_sma else -1.0
            predictions[i, :] = signal

        return predictions


def compute_baseline_metrics(
    y_true: np.ndarray,
    last_returns: np.ndarray = None,
    close_prices: np.ndarray = None,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for all baseline models.

    Args:
        y_true: Actual log returns, shape (N, 15)
        last_returns: Last observed 1-bar return per sample, shape (N,).
            Required for persistence baseline.
        close_prices: Historical close prices, shape (N, lookback).
            Required for MA crossover baseline.

    Returns:
        Dict mapping baseline name -> metrics dict
    """
    n_samples, horizon = y_true.shape
    baselines = {}

    # Random predictor
    random_model = RandomPredictor(seed=42)
    random_preds = random_model.predict(n_samples, horizon)
    # Scale random predictions to have similar magnitude as actual returns
    scale = np.std(y_true) + 1e-8
    random_preds_scaled = random_preds * scale
    baselines["random"] = compute_metrics(y_true, random_preds_scaled)

    # Persistence model
    if last_returns is not None:
        persistence = PersistenceModel()
        persistence_preds = persistence.predict(last_returns, horizon)
        baselines["persistence"] = compute_metrics(y_true, persistence_preds)

    # MA crossover
    if close_prices is not None:
        ma_model = MACrossoverBaseline(fast_period=9, slow_period=21)
        ma_preds = ma_model.predict(close_prices, horizon)
        # Scale MA signals to return magnitude
        ma_preds_scaled = ma_preds * scale
        baselines["ma_crossover"] = compute_metrics(y_true, ma_preds_scaled)

    return baselines


def format_metrics_report(
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, Dict[str, float]] = None,
) -> str:
    """Format a human-readable evaluation report.

    Args:
        model_metrics: Metrics dict from compute_metrics()
        baseline_metrics: Optional dict of baseline name -> metrics

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EVALUATION REPORT")
    lines.append("=" * 70)

    lines.append("\n--- TFT Model ---")
    lines.append(f"  MSE:                         {model_metrics['mse']:.6f}")
    lines.append(f"  RMSE:                        {model_metrics['rmse']:.6f}")
    lines.append(f"  MAE:                         {model_metrics['mae']:.6f}")
    lines.append(
        f"  Directional Acc (all steps): {model_metrics['directional_accuracy_all_steps']:.4f}"
    )
    lines.append(
        f"  Directional Acc (step 15):   {model_metrics['directional_accuracy_15']:.4f}"
    )
    lines.append(f"  Win Rate:                    {model_metrics['win_rate']:.4f}")
    lines.append(
        f"  Sharpe Ratio (annualized):   {model_metrics['sharpe_ratio']:.4f}"
    )
    lines.append(f"  Total Return:                {model_metrics['total_return']:.6f}")
    lines.append(f"  Max Drawdown:                {model_metrics['max_drawdown']:.6f}")
    lines.append(f"  Profit Factor:               {model_metrics['profit_factor']:.4f}")

    if baseline_metrics:
        for name, bmetrics in baseline_metrics.items():
            lines.append(f"\n--- Baseline: {name} ---")
            lines.append(
                f"  Directional Acc (step 15):   {bmetrics['directional_accuracy_15']:.4f}"
            )
            lines.append(f"  Win Rate:                    {bmetrics['win_rate']:.4f}")
            lines.append(
                f"  Sharpe Ratio (annualized):   {bmetrics['sharpe_ratio']:.4f}"
            )
            lines.append(
                f"  Profit Factor:               {bmetrics['profit_factor']:.4f}"
            )

        # Summary comparison
        lines.append("\n--- Model vs Baselines (Directional Acc @ step 15) ---")
        model_da = model_metrics["directional_accuracy_15"]
        for name, bmetrics in baseline_metrics.items():
            baseline_da = bmetrics["directional_accuracy_15"]
            diff = model_da - baseline_da
            direction = "+" if diff > 0 else ""
            lines.append(
                f"  vs {name:15s}: {direction}{diff:.4f} "
                f"({model_da:.4f} vs {baseline_da:.4f})"
            )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
