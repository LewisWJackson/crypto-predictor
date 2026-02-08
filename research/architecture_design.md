# Architecture Design: Crypto Price Prediction with TFT

> Implementation-ready specification for the Crypto Predictor project
> Date: February 2026

---

## 1. Model Architecture: Temporal Fusion Transformer (TFT)

### Why TFT

The TFT is the best fit for our use case because it:
- Natively supports multi-horizon forecasting (no autoregressive error accumulation)
- Has built-in variable selection networks (automatically learns which features matter)
- Handles static, known-future, and observed-only inputs natively
- Provides interpretable attention weights for debugging
- Has a production-ready implementation in `pytorch-forecasting`

### Implementation: pytorch-forecasting + PyTorch Lightning

We use the `TemporalFusionTransformer` class from `pytorch-forecasting`. This handles the full TFT architecture internally (variable selection, gated residual networks, interpretable multi-head attention, quantile outputs).

### Hyperparameters

```python
# Model hyperparameters
MODEL_CONFIG = {
    "hidden_size": 64,                # Hidden state size for all internal networks
    "attention_head_size": 4,          # Number of attention heads
    "dropout": 0.15,                   # Dropout rate across all layers
    "hidden_continuous_size": 32,      # Size of hidden layer for continuous variable processing
    "learning_rate": 1e-3,             # Initial learning rate (will be scheduled)
    "gradient_clip_val": 1.0,          # Max gradient norm
    "output_size": 7,                  # Number of quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    "loss": "QuantileLoss",            # We override with custom loss (see Section 5)
    "log_interval": 10,
    "reduce_on_plateau_patience": 5,
}

# Data dimensions
DATA_CONFIG = {
    "encoder_length": 256,             # Lookback: 256 bars (~4.3 hours of 1-min data)
    "decoder_length": 15,              # Prediction horizon: 15 bars (15 minutes)
    "batch_size": 64,                  # Training batch size
}
```

### Input Categories (TFT-native)

The TFT framework categorizes inputs into three types. This is how we map our features:

| TFT Category | Description | Our Features |
|---|---|---|
| **time_varying_known_reals** | Features known at prediction time (can be computed from past data or are deterministic) | Temporal encodings (hour_sin, hour_cos, day_sin, day_cos, minute_sin, minute_cos), higher-timeframe indicators computed from past data |
| **time_varying_unknown_reals** | Features only known up to the present (not available in the future) | OHLCV, returns, all 1-min technical indicators, volume indicators |
| **static_categoricals** | Time-invariant categorical features | `asset_id` (e.g., "BTC"), `exchange_id` (e.g., "binance") |

### Target Variable

```python
TARGET = "log_return_15"  # Log return over next 15 bars: ln(close[t+15] / close[t])
```

We predict **log returns** (not raw prices) because:
- Returns are approximately stationary (prices are not)
- Removes scale dependency — model generalizes across price levels
- Log returns are additive (easier to aggregate multi-step predictions)
- The sign directly gives us direction (positive = up, negative = down)

---

## 2. Feature Set (43 Features Total)

### 2.1 Price-Derived Features (6 features)

| # | Feature Name | Computation | Category |
|---|---|---|---|
| 1 | `log_return_1` | `ln(close[t] / close[t-1])` | unknown_real |
| 2 | `log_return_5` | `ln(close[t] / close[t-5])` | unknown_real |
| 3 | `log_return_15` | `ln(close[t] / close[t-15])` | unknown_real |
| 4 | `log_return_60` | `ln(close[t] / close[t-60])` | unknown_real |
| 5 | `high_low_range` | `(high - low) / close` | unknown_real |
| 6 | `close_open_range` | `(close - open) / close` | unknown_real |

### 2.2 Trend Indicators (8 features)

| # | Feature Name | Computation | Category |
|---|---|---|---|
| 7 | `sma_20` | `SMA(close, 20) / close - 1` (normalized distance from SMA) | unknown_real |
| 8 | `sma_50` | `SMA(close, 50) / close - 1` | unknown_real |
| 9 | `sma_200` | `SMA(close, 200) / close - 1` | unknown_real |
| 10 | `ema_9` | `EMA(close, 9) / close - 1` | unknown_real |
| 11 | `ema_21` | `EMA(close, 21) / close - 1` | unknown_real |
| 12 | `macd_line` | `EMA(close, 12) - EMA(close, 26)`, normalized by close | unknown_real |
| 13 | `macd_signal` | `EMA(macd_line, 9)`, normalized by close | unknown_real |
| 14 | `macd_histogram` | `macd_line - macd_signal`, normalized by close | unknown_real |

### 2.3 Momentum Indicators (7 features)

| # | Feature Name | Computation | Category |
|---|---|---|---|
| 15 | `rsi_14` | RSI(close, 14) / 100 (scale to 0-1) | unknown_real |
| 16 | `stoch_k` | Stochastic %K(14, 3) / 100 | unknown_real |
| 17 | `stoch_d` | Stochastic %D(14, 3) / 100 | unknown_real |
| 18 | `williams_r` | Williams %R(14) / 100 | unknown_real |
| 19 | `cci_20` | CCI(20) / 200 (normalize around ±1) | unknown_real |
| 20 | `roc_10` | Rate of Change(close, 10) / 100 | unknown_real |
| 21 | `momentum_10` | `(close - close[t-10]) / close[t-10]` | unknown_real |

### 2.4 Volatility Indicators (7 features)

| # | Feature Name | Computation | Category |
|---|---|---|---|
| 22 | `bb_upper` | `(BB_upper(20, 2) - close) / close` | unknown_real |
| 23 | `bb_lower` | `(close - BB_lower(20, 2)) / close` | unknown_real |
| 24 | `bb_bandwidth` | `(BB_upper - BB_lower) / BB_middle` | unknown_real |
| 25 | `bb_pctb` | `(close - BB_lower) / (BB_upper - BB_lower)` | unknown_real |
| 26 | `atr_14` | `ATR(14) / close` (normalized) | unknown_real |
| 27 | `rolling_volatility_20` | `std(log_return_1, window=20)` | unknown_real |
| 28 | `rolling_volatility_60` | `std(log_return_1, window=60)` | unknown_real |

### 2.5 Volume Indicators (6 features)

| # | Feature Name | Computation | Category |
|---|---|---|---|
| 29 | `volume_sma_ratio` | `volume / SMA(volume, 20)` | unknown_real |
| 30 | `obv_normalized` | `OBV / SMA(abs(OBV), 20)` — direction + relative magnitude | unknown_real |
| 31 | `vwap_distance` | `(close - VWAP) / close` | unknown_real |
| 32 | `mfi_14` | Money Flow Index(14) / 100 | unknown_real |
| 33 | `volume_roc` | `(volume - volume[t-10]) / (volume[t-10] + 1e-8)` | unknown_real |
| 34 | `ad_line_normalized` | A/D Line / SMA(abs(A/D), 20) | unknown_real |

### 2.6 Rolling Statistics (3 features)

| # | Feature Name | Computation | Category |
|---|---|---|---|
| 35 | `rolling_skew_60` | `skew(log_return_1, window=60)` | unknown_real |
| 36 | `rolling_kurtosis_60` | `kurtosis(log_return_1, window=60)` | unknown_real |
| 37 | `autocorr_lag1` | `autocorrelation(log_return_1, lag=1, window=60)` | unknown_real |

### 2.7 Temporal Encodings (6 features)

| # | Feature Name | Computation | Category |
|---|---|---|---|
| 38 | `hour_sin` | `sin(2π × hour / 24)` | known_real |
| 39 | `hour_cos` | `cos(2π × hour / 24)` | known_real |
| 40 | `day_of_week_sin` | `sin(2π × day_of_week / 7)` | known_real |
| 41 | `day_of_week_cos` | `cos(2π × day_of_week / 7)` | known_real |
| 42 | `minute_of_hour_sin` | `sin(2π × minute / 60)` | known_real |
| 43 | `minute_of_hour_cos` | `cos(2π × minute / 60)` | known_real |

### Multi-Timeframe Integration

Instead of a separate hierarchical encoder, we include **higher-timeframe indicators as features within the 1-minute feature vector**. Each 1-minute bar carries the "current value" of the higher timeframe indicator (which updates less frequently). These are computed by resampling the raw 1-minute OHLCV data.

This is Approach C from the research — using TFT's native variable selection to handle multi-resolution inputs.

The higher-timeframe features are already captured above:
- `sma_200` (200-minute ≈ 3.3 hours) captures hourly trend
- `rolling_volatility_60` (60-minute) captures hourly volatility regime
- `log_return_60` captures hourly momentum
- The temporal encodings capture session-level patterns

This keeps the feature set manageable at 43 features while providing multi-scale context. Adding explicit 5min/15min/1hr resampled indicators is an optional enhancement for later iteration.

---

## 3. Data Pipeline Specification

### 3.1 Raw Data Requirements

```
Source: Binance via CCXT (already implemented in scripts/download_data.py)
Pair: BTC/USDT (primary), ETH/USDT (secondary)
Timeframe: 1-minute candles
Minimum history: 2 years (~1,052,000 candles)
Fields per candle: timestamp, open, high, low, close, volume
Storage format: Parquet files in data/raw/
```

### 3.2 Feature Computation Pipeline

```
data/raw/BTC_USDT_1m.parquet
    │
    ▼
[1] Load raw OHLCV data (pandas DataFrame)
    │
    ▼
[2] Compute all 43 features (vectorized pandas/numpy operations)
    │  - Technical indicators via `ta` library or manual computation
    │  - Normalize all indicators as specified (divide by close, /100, etc.)
    │
    ▼
[3] Drop rows with NaN from indicator warm-up period
    │  (first 200 rows will have NaN from SMA_200)
    │
    ▼
[4] Compute target variable: log_return_15 = ln(close[t+15] / close[t])
    │  (last 15 rows will have NaN target — drop them)
    │
    ▼
[5] Save processed data to data/processed/BTC_USDT_features.parquet
    │
    ▼
[6] Create TimeSeriesDataSet (pytorch-forecasting)
```

### 3.3 Normalization Strategy

**Per-feature z-score normalization using ONLY training set statistics:**

```python
# Computed on training data only
feature_means = train_df[feature_columns].mean()
feature_stds = train_df[feature_columns].std()

# Applied to all splits
def normalize(df, means, stds):
    return (df[feature_columns] - means) / (stds + 1e-8)
```

Many features are already "self-normalizing" by construction (RSI is 0-100, BB %B is 0-1, returns are naturally small). For these, z-score still applies but the effect is minor.

**Exception**: Temporal encodings (sin/cos) are already in [-1, 1] — do not normalize.

### 3.4 Train/Validation/Test Split

```
Total data: ~1,052,000 candles (2 years of 1-min data)
Drop first 200 (indicator warm-up) and last 15 (target computation)
Usable: ~1,051,785 candles

Split (chronological, no shuffling):
┌─────────────────────────────────────────────────────────────────┐
│  Training (70%)      │ Purge │  Val (15%)   │ Purge │Test (15%)│
│  ~736,250 candles    │  500  │ ~157,518     │  500  │~157,517  │
│  Jan 2024 – Sep 2025 │       │Oct-Nov 2025  │       │Dec25-Feb26│
└─────────────────────────────────────────────────────────────────┘

Purge gap: 500 bars (8.3 hours) between each split.
Purpose: Prevents lookahead bias from overlapping lookback windows.
```

**Important**: Normalization statistics (mean, std) are computed ONLY on the training split. Validation and test data are normalized using training statistics.

### 3.5 pytorch-forecasting TimeSeriesDataSet Configuration

```python
from pytorch_forecasting import TimeSeriesDataSet

# Add required columns to the DataFrame
df["time_idx"] = range(len(df))  # Monotonically increasing time index
df["group"] = "BTC_USDT"          # Series identifier (single series for now)

# Define the dataset
training_dataset = TimeSeriesDataSet(
    data=train_df,
    time_idx="time_idx",
    target="log_return_15",
    group_ids=["group"],

    # Lookback and horizon
    max_encoder_length=256,
    max_prediction_length=15,
    min_encoder_length=256,       # Fixed encoder length (no variable length)
    min_prediction_length=15,

    # Feature categorization
    time_varying_known_reals=[
        "hour_sin", "hour_cos",
        "day_of_week_sin", "day_of_week_cos",
        "minute_of_hour_sin", "minute_of_hour_cos",
    ],
    time_varying_unknown_reals=[
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
    ],
    static_categoricals=["group"],  # Asset identifier

    # Target normalization (handled by pytorch-forecasting internally)
    target_normalizer=None,  # We pre-normalize ourselves

    add_relative_time_idx=True,
    add_target_scales=False,
    add_encoder_length=True,
)

# Validation dataset inherits parameters from training
validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,
    data=val_df,
    predict=True,
    stop_randomization=True,
)
```

---

## 4. Training Configuration

### 4.1 Optimizer and Learning Rate Schedule

```python
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer

# TFT model instantiation
model = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.15,
    hidden_continuous_size=32,
    output_size=7,  # 7 quantiles
    loss=CombinedTradingLoss(),  # Custom loss (see Section 5)
    log_interval=10,
    reduce_on_plateau_patience=5,
    optimizer="adam",
)

# PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=100,
    gradient_clip_val=1.0,
    callbacks=[
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-5,
            mode="min",
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ],
    enable_progress_bar=True,
    accelerator="auto",  # Uses GPU if available, else CPU
    devices=1,
)
```

### 4.2 Learning Rate Schedule

`pytorch-forecasting`'s TFT has a built-in `ReduceLROnPlateau` scheduler controlled by `reduce_on_plateau_patience`. We use:

```python
# Built into pytorch-forecasting TFT:
# - Optimizer: Adam (or Ranger if specified)
# - Scheduler: ReduceLROnPlateau
# - reduce_on_plateau_patience: 5 (reduce LR after 5 epochs without improvement)
# - reduce_on_plateau_reduction: 10 (divide LR by 10)
# - Initial LR: 1e-3 → 1e-4 → 1e-5 as training plateaus
```

### 4.3 Training Loop

```python
# Dataloaders
train_dataloader = training_dataset.to_dataloader(
    train=True,
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
)

val_dataloader = validation_dataset.to_dataloader(
    train=False,
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
)

# Train
trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# Load best checkpoint
best_model = TemporalFusionTransformer.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path
)
```

### 4.4 Full Training Parameters Summary

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam | Standard, well-tested |
| Initial LR | 1e-3 | Good default for TFT |
| LR Schedule | ReduceLROnPlateau (patience=5, factor=0.1) | Adaptive to convergence |
| Batch size | 64 | Balance between noise and gradient stability |
| Max epochs | 100 | Early stopping will halt before this |
| Early stopping patience | 10 | Financial data is noisy — need patience |
| Gradient clipping | 1.0 (max norm) | Prevents exploding gradients |
| Dropout | 0.15 | Moderate — financial data needs regularization but signal is weak |
| Weight decay | 0.0 | Handled by dropout; can add 1e-4 if overfitting persists |
| Encoder length | 256 bars | ~4.3 hours lookback |
| Decoder length | 15 bars | 15-minute prediction horizon |
| Num workers | 4 | Parallel data loading |

---

## 5. Loss Function: Combined Directional + MSE Loss

### The Problem with Standard Losses

For trading, we care about:
1. **Direction** — is the predicted move up or down? (most important)
2. **Magnitude** — how big is the predicted move? (important for position sizing)
3. **Large moves** — getting big moves right matters more than small moves

### Custom Loss Implementation

Since `pytorch-forecasting` expects a loss that implements its `MultiHorizonMetric` interface, we create a custom loss class:

```python
import torch
import torch.nn as nn
from pytorch_forecasting.metrics import MultiHorizonMetric


class CombinedTradingLoss(MultiHorizonMetric):
    """
    Combined loss: MSE + directional penalty + return-weighted MSE.

    L = w_mse * MSE + w_dir * DirectionalPenalty + w_rw * ReturnWeightedMSE

    where:
        MSE = mean((y_true - y_pred)^2)
        DirectionalPenalty = mean(1 - I(sign(y_true) == sign(y_pred)))
        ReturnWeightedMSE = mean((y_true - y_pred)^2 * (1 + |y_true|))
    """

    def __init__(
        self,
        w_mse: float = 0.3,
        w_dir: float = 0.4,
        w_rw: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.w_mse = w_mse
        self.w_dir = w_dir
        self.w_rw = w_rw

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined trading loss.

        Args:
            y_pred: Predicted values, shape (batch_size, decoder_length)
            target: Actual values, shape (batch_size, decoder_length)

        Returns:
            Loss tensor, shape (batch_size, decoder_length)
        """
        # MSE component (per-element)
        mse = (target - y_pred) ** 2

        # Directional penalty (per-element)
        # 1.0 when direction is wrong, 0.0 when correct
        direction_correct = (torch.sign(target) * torch.sign(y_pred) > 0).float()
        direction_penalty = 1.0 - direction_correct

        # Return-weighted MSE (per-element)
        # Larger actual moves get more weight
        return_weight = 1.0 + torch.abs(target) * 100  # Scale since log returns are small
        return_weighted_mse = mse * return_weight

        # Combined loss (per-element — pytorch-forecasting handles reduction)
        loss = self.w_mse * mse + self.w_dir * direction_penalty + self.w_rw * return_weighted_mse

        return loss

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Convert network output to point prediction (take the mean/median)."""
        if y_pred.ndim == 3:
            # If output has quantile dimension, take the median (index 3 of 7 quantiles)
            return y_pred[..., 3]
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Convert to quantiles (pass through if already quantiles)."""
        return y_pred
```

### Alternative: Use Built-in QuantileLoss + Evaluate Directional Accuracy Separately

If the custom loss causes training instability, fall back to:

```python
from pytorch_forecasting.metrics import QuantileLoss

loss = QuantileLoss(quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
```

Then evaluate directional accuracy as a **metric** (not a loss). This is a safe fallback because:
- QuantileLoss is well-tested with TFT
- Quantile predictions inherently capture uncertainty
- The median quantile (0.5) gives us the point prediction for direction

### Recommendation

**Start with `QuantileLoss`** (proven, stable) and evaluate directional accuracy as a metric. Then try `CombinedTradingLoss` as an experiment. Compare the two on validation data.

---

## 6. Evaluation Metrics

### 6.1 Metric Definitions

```python
import numpy as np
from typing import Dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Actual log returns, shape (N, 15)  — N samples, 15-step horizon
        y_pred: Predicted log returns, shape (N, 15)

    Returns:
        Dictionary of metric name → value
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
    direction_correct_final = (np.sign(y_true_final) == np.sign(y_pred_final)).astype(float)
    metrics["directional_accuracy_15"] = float(np.mean(direction_correct_final))

    # --- Simulated Trading Metrics ---
    # Simple strategy: go long if predicted return > 0, short if < 0
    # Position size = 1 (fixed)
    positions = np.sign(y_pred_final)  # +1 or -1
    strategy_returns = positions * y_true_final  # Per-trade return

    # Win rate
    wins = (strategy_returns > 0).sum()
    total_trades = len(strategy_returns)
    metrics["win_rate"] = float(wins / total_trades)

    # Sharpe ratio (annualized, assuming 1-min bars, 525,600 mins/year)
    # Each prediction covers 15 mins → ~35,040 predictions/year
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns) + 1e-8
    metrics["sharpe_ratio"] = float(mean_return / std_return * np.sqrt(35040))

    # Cumulative return
    cumulative = np.cumsum(strategy_returns)
    metrics["total_return"] = float(np.sum(strategy_returns))

    # Maximum drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    metrics["max_drawdown"] = float(np.max(drawdowns))

    # Profit factor
    gross_profit = np.sum(strategy_returns[strategy_returns > 0])
    gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0])) + 1e-8
    metrics["profit_factor"] = float(gross_profit / gross_loss)

    return metrics
```

### 6.2 Target Performance Thresholds

These are the minimum thresholds the model must meet on the **test set** to be considered viable:

| Metric | Minimum Threshold | Good | Excellent |
|---|---|---|---|
| Directional Accuracy (15-step) | > 52% | > 55% | > 58% |
| Win Rate | > 51% | > 53% | > 56% |
| Sharpe Ratio (annualized) | > 0.5 | > 1.0 | > 2.0 |
| Max Drawdown | < 20% | < 10% | < 5% |
| Profit Factor | > 1.1 | > 1.3 | > 1.5 |

**Important**: These are measured on out-of-sample test data with the 500-bar purge gap. If the model can't beat 52% directional accuracy on unseen data, it's not capturing real signal.

### 6.3 Baseline Comparisons

Always compare against these baselines:

1. **Random Predictor**: 50% directional accuracy, Sharpe ≈ 0
2. **Persistence Model**: Predict that the next return equals the last observed return
3. **Moving Average Crossover**: Simple SMA(9)/SMA(21) crossover signal

The TFT model must significantly beat all three baselines.

---

## 7. Project File Structure

```
crypto-predictor/
│
├── configs/
│   └── tft_config.yaml          # All hyperparameters in one place
│
├── data/
│   ├── raw/                      # Raw OHLCV parquet files
│   │   ├── BTC_USDT_1m.parquet
│   │   └── ETH_USDT_1m.parquet
│   └── processed/                # Feature-engineered parquet files
│       ├── BTC_USDT_features.parquet
│       └── ETH_USDT_features.parquet
│
├── models/                       # Saved model checkpoints
│   └── tft/
│       ├── best_model.ckpt
│       └── training_logs/
│
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_evaluation.ipynb
│
├── research/                     # Research documents (already exists)
│   ├── data_sources.md
│   ├── ml_approaches.md
│   └── architecture_design.md   # This document
│
├── scripts/                      # Executable scripts
│   ├── download_data.py          # Already exists — downloads OHLCV via CCXT
│   ├── utils.py                  # Already exists — shared utilities
│   ├── compute_features.py       # NEW: Feature engineering pipeline
│   ├── train_tft.py              # NEW: TFT training script
│   ├── evaluate.py               # NEW: Model evaluation and metrics
│   └── predict.py                # NEW: Inference / prediction script
│
├── src/                          # Reusable modules
│   ├── __init__.py
│   ├── features.py               # Feature computation functions
│   ├── dataset.py                # TimeSeriesDataSet creation
│   ├── loss.py                   # CombinedTradingLoss + other losses
│   ├── metrics.py                # compute_metrics() and baselines
│   └── model.py                  # Model factory / config loading
│
├── requirements.txt              # Already exists — needs updating
├── README.md
└── .gitignore
```

### Updated requirements.txt

```
# Core data processing
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Data download
ccxt>=4.0.0
requests>=2.31.0

# Technical analysis
ta>=0.11.0

# Machine learning / preprocessing
scikit-learn>=1.3.0

# Deep learning
torch>=2.1.0
pytorch-lightning>=2.1.0
pytorch-forecasting>=1.0.0

# Hyperparameter optimization
optuna>=3.4.0

# Experiment tracking
tensorboard>=2.15.0

# Progress bars
tqdm>=4.65.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0
```

---

## 8. Implementation Order

The implementation agent should build in this order:

### Phase 1: Feature Engineering (`compute_features.py` + `src/features.py`)
1. Load raw parquet data
2. Implement all 43 features as documented in Section 2
3. Compute target variable (`log_return_15`)
4. Handle NaN rows (drop warm-up period)
5. Save processed parquet to `data/processed/`
6. **Verify**: Print feature statistics, check for NaN/inf values

### Phase 2: Dataset Creation (`src/dataset.py`)
1. Implement train/val/test split with 500-bar purge gaps
2. Compute normalization statistics from training set
3. Apply normalization to all splits
4. Create `TimeSeriesDataSet` objects as specified in Section 3.5
5. Create dataloaders
6. **Verify**: Print dataset shapes, sample a batch, check dimensions

### Phase 3: Loss and Metrics (`src/loss.py` + `src/metrics.py`)
1. Implement `CombinedTradingLoss` (Section 5)
2. Implement `compute_metrics()` (Section 6)
3. Implement baseline models (random, persistence, MA crossover)
4. **Verify**: Unit test loss with synthetic data

### Phase 4: Training Script (`scripts/train_tft.py`)
1. Load config from `configs/tft_config.yaml`
2. Create datasets and dataloaders
3. Instantiate TFT model
4. Configure trainer with callbacks
5. Run training
6. Save best model checkpoint
7. **Verify**: Training converges, val loss decreases

### Phase 5: Evaluation (`scripts/evaluate.py`)
1. Load best model checkpoint
2. Run predictions on test set
3. Compute all metrics from Section 6
4. Compare against baselines
5. Generate evaluation report
6. **Verify**: Metrics meet minimum thresholds from Section 6.2

---

## 9. Configuration File

All hyperparameters should be centralized in a YAML config:

```yaml
# configs/tft_config.yaml

# Data
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  pair: "BTC_USDT"
  timeframe: "1m"

# Feature engineering
features:
  # SMA periods
  sma_periods: [20, 50, 200]
  ema_periods: [9, 21]

  # MACD
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9

  # RSI
  rsi_period: 14

  # Stochastic
  stoch_k_period: 14
  stoch_d_period: 3

  # Bollinger Bands
  bb_period: 20
  bb_std: 2

  # ATR
  atr_period: 14

  # Rolling windows
  volatility_windows: [20, 60]
  stats_window: 60

  # Volume
  volume_sma_period: 20
  mfi_period: 14
  volume_roc_period: 10

# Dataset
dataset:
  encoder_length: 256
  decoder_length: 15
  train_fraction: 0.70
  val_fraction: 0.15
  test_fraction: 0.15
  purge_gap: 500
  batch_size: 64
  num_workers: 4

# Model
model:
  hidden_size: 64
  attention_head_size: 4
  dropout: 0.15
  hidden_continuous_size: 32
  output_size: 7
  learning_rate: 0.001

# Training
training:
  max_epochs: 100
  gradient_clip_val: 1.0
  early_stopping_patience: 10
  early_stopping_min_delta: 0.00001
  reduce_lr_patience: 5
  checkpoint_top_k: 3

# Loss
loss:
  type: "quantile"  # "quantile" or "combined_trading"
  # Combined trading loss weights (used when type == "combined_trading")
  w_mse: 0.3
  w_dir: 0.4
  w_rw: 0.3
  # Quantile loss quantiles (used when type == "quantile")
  quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

# Evaluation
evaluation:
  min_directional_accuracy: 0.52
  min_sharpe_ratio: 0.5
  max_drawdown: 0.20
```

---

## 10. Key Implementation Notes

### Things to Watch Out For

1. **NaN/Inf values**: Technical indicators produce NaN during their warm-up period. Always drop these rows AFTER computing ALL features (the longest warm-up is SMA_200 = 200 bars).

2. **Feature leakage**: The target is `log_return_15` = `ln(close[t+15] / close[t])`. Make sure NO feature uses data from `t+1` onward. All indicators must be computed using data up to and including time `t`.

3. **Time index alignment**: `pytorch-forecasting` requires a monotonically increasing `time_idx` column. After dropping NaN rows, reset the time index to be contiguous.

4. **Normalization boundary**: Compute mean/std from training data ONLY. Apply those same statistics to validation and test data. Never fit normalizer on the full dataset.

5. **Quantile output shape**: When using `QuantileLoss` with `output_size=7`, the model output shape is `(batch, decoder_length, 7)`. The point prediction is the median (index 3).

6. **GPU memory**: With `encoder_length=256`, `batch_size=64`, and `hidden_size=64`, memory usage should be modest (~2-4 GB). If OOM occurs, reduce batch size to 32.

7. **Data type**: Use `float32` throughout. `float64` wastes memory with no benefit for neural network training.

### Performance Expectations

- **Training time**: ~30-60 minutes on a single GPU (RTX 3060+) for 100 epochs with early stopping
- **Training time on CPU**: ~4-8 hours (feasible but slow)
- **Inference time**: < 50ms per prediction batch (fast enough for real-time)
- **Model size**: ~500K-2M parameters (much larger than original 200K, but still small by modern standards)
