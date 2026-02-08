# ML Approaches for Financial Time Series Prediction

> Research report for the Crypto Price Prediction AI project
> Date: February 2026

## Table of Contents

1. [Architecture Comparison](#1-architecture-comparison)
2. [Feature Engineering](#2-feature-engineering)
3. [Multi-Timeframe Analysis](#3-multi-timeframe-analysis)
4. [Training Strategies](#4-training-strategies)
5. [Loss Functions](#5-loss-functions)
6. [Regularization](#6-regularization)
7. [Ensemble Methods](#7-ensemble-methods)
8. [Key Improvements Over the Original Approach](#8-key-improvements-over-the-original-approach)

---

## 1. Architecture Comparison

### Overview of Architectures

| Architecture | Strengths | Weaknesses | Best For |
|---|---|---|---|
| **Temporal Fusion Transformer (TFT)** | Interpretable, multi-horizon, variable selection | Complex, slower training | Multi-step forecasting with mixed inputs |
| **PatchTST** | Efficient patching, long context, channel-independent | Less interpretable | Long-term univariate/multivariate forecasting |
| **iTransformer** | Inverted attention on variables, ICLR 2024 | Newer, less battle-tested | Short-term multivariate forecasting |
| **TimesNet** | 2D variation modeling, top short-term benchmark | Complex architecture | Short-term forecasting |
| **LSTM** | Proven, stable, well-understood, fast inference | Vanishing gradients, sequential training | Sequential patterns, shorter lookbacks |
| **GRU** | Lighter than LSTM, competitive performance | Slightly less expressive than LSTM | Resource-constrained scenarios |
| **TCN (Temporal Convolutional Network)** | Parallelizable, stable gradients, faster training | Fixed receptive field | Real-time inference, long sequences |
| **Hybrid LSTM-Transformer** | Combines local + global patterns | More hyperparameters | Complex temporal dependencies |

### Detailed Analysis

#### Temporal Fusion Transformer (TFT) — **Recommended Primary Architecture**

The TFT ([Lim et al., 2019](https://arxiv.org/abs/1912.09363)) is the strongest candidate for our use case because it provides:

- **Variable Selection Networks**: Automatically learns which input features matter most at each time step. This is critical when feeding dozens of technical indicators — the model learns to ignore irrelevant ones.
- **Multi-horizon forecasting**: Natively predicts multiple future time steps simultaneously rather than one-step-at-a-time autoregressive generation (a major weakness of the original approach).
- **Interpretable attention**: Shows which past time steps the model focuses on, enabling debugging and trust-building.
- **Static covariate handling**: Can incorporate time-invariant features (e.g., which crypto asset, which exchange).
- **Gated residual connections**: Help with gradient flow and training stability.

Research shows TFT demonstrates the best overall performance in multivariate settings with exogenous variables, significantly outperforming baselines across nearly all forecast horizons ([Hewamalage et al., 2023](https://www.sciencedirect.com/science/article/pii/S2405844024161737)).

**Benchmark results**: RMSE of 41.87 with 69.1% directional accuracy in crypto forecasting tasks.

#### PatchTST

PatchTST ([Nie et al., 2023](https://arxiv.org/abs/2211.14730)) treats time series as "sentences" of "word patches":

- Segments time series into patches (e.g., 16 consecutive time steps per patch)
- Reduces attention complexity from O(L^2) to O((L/P)^2) where P is patch size
- Channel-independent design prevents overfitting on cross-variable correlations
- Achieves 21% MSE reduction vs prior Transformer approaches
- **Best for**: When you have very long lookback windows (>500 steps)

#### LSTM and GRU

Still competitive, especially in hybrid configurations:

- LSTM achieves RMSE of 43.25 — close to Transformers but with much lower computational cost
- The hybrid LSTM+XGBoost model consistently outperforms individual models by capturing temporal dependencies and leveraging gradient boosting's non-linearity handling
- GRU outperforms TCN in multi-factor time series prediction in some benchmarks

#### TCN (Temporal Convolutional Networks)

- **Training speed**: Builds stable models faster than LSTM (fully parallelizable)
- **Long-range dependencies**: Dilated causal convolutions can cover very long histories
- **Performance**: RMSE of 15.26 vs 23.53 for CNN-LSTM in gold price prediction
- **Limitation**: Less effective than GRU/LSTM for multi-factor prediction in some studies
- **Best for**: Real-time inference where latency matters

#### Hybrid Architectures — **Emerging Best Practice**

The consensus in 2025-2026 research is that **hybrid architectures combining complementary strengths consistently outperform single architectures**:

- **LSTM-Transformer**: LSTM captures short-term sequential patterns; Transformer captures long-range dependencies via attention
- **TCN-LSTM**: TCN for efficient parallel feature extraction; LSTM for sequential refinement
- **TFT-GNN**: TFT for temporal modeling; Graph Neural Networks for cross-asset relationships

### Architecture Recommendation

**Primary**: Temporal Fusion Transformer (TFT) — best balance of performance, interpretability, and multi-horizon capability.

**Secondary/Ensemble member**: TCN or LSTM as a faster, complementary model in an ensemble.

**For experimentation**: PatchTST if we want to push long-context performance, iTransformer for multivariate short-term forecasting.

---

## 2. Feature Engineering

### Beyond Raw OHLCV: Feature Categories

The original approach used **only raw 1-minute OHLCV candle data**. Research overwhelmingly shows that enriching the feature set dramatically improves predictions.

### 2.1 Technical Indicators (Computed from OHLCV)

**Trend Indicators:**
- Moving Averages: SMA(7, 14, 21, 50, 100, 200), EMA(9, 21, 55)
- MACD (12, 26, 9) — signal line, histogram, divergence
- ADX (Average Directional Index) — trend strength
- Ichimoku Cloud components (Tenkan, Kijun, Senkou spans)

**Momentum Indicators:**
- RSI (14-period) — overbought/oversold
- Stochastic Oscillator (%K, %D)
- Williams %R
- CCI (Commodity Channel Index)
- Rate of Change (ROC)
- Momentum (price change over N periods)

**Volatility Indicators:**
- Bollinger Bands (20, 2σ) — upper, lower, bandwidth, %B
- ATR (Average True Range) — 14-period
- Keltner Channels
- Historical volatility (rolling std of returns)
- Garman-Klass volatility estimator (uses OHLC, more efficient than close-to-close)

**Volume Indicators:**
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- Volume Rate of Change
- Accumulation/Distribution Line
- Money Flow Index (MFI)
- Chaikin Money Flow (CMF)

### 2.2 Market Microstructure Features

- **Bid-ask spread** (if available from exchange API)
- **Order book imbalance**: Ratio of bid vs ask volume at top N levels
- **Trade flow imbalance**: Buy vs sell aggressor volume
- **Volume profile**: Distribution of volume at price levels
- **Tick-based features**: Number of trades per candle, average trade size

### 2.3 On-Chain Data (Crypto-Specific)

Research shows on-chain data significantly improves Bitcoin price direction prediction ([ScienceDirect, 2025](https://www.sciencedirect.com/science/article/pii/S266682702500057X)):

- **Active addresses**: Number of unique sending/receiving addresses
- **Transaction count and volume**: Network activity proxy
- **Hash rate**: Mining activity and network security
- **Exchange inflows/outflows**: Selling pressure vs accumulation signals
- **Whale wallet movements**: Large holder behavior
- **Funding rates**: Perpetual futures market sentiment
- **Open interest**: Derivatives market positioning
- **MVRV ratio**: Market Value to Realized Value — historically strong signal

### 2.4 Temporal/Cyclical Features

Incorporating cyclical temporal features leads to consistent and significant improvements ([Springer, 2025](https://link.springer.com/chapter/10.1007/978-981-95-3358-9_25)):

- **Hour of day** (sin/cos encoded): Captures intraday patterns (Asian, European, US sessions)
- **Day of week** (sin/cos encoded): Weekend vs weekday dynamics
- **Month of year** (sin/cos encoded): Seasonal patterns
- **Minutes since last significant move**: Captures volatility clustering
- **Time since last major support/resistance touch**

### 2.5 Cross-Asset Features

- **BTC dominance**: Bitcoin's share of total crypto market cap
- **ETH/BTC ratio**: Risk-on/risk-off within crypto
- **DXY (US Dollar Index)**: Inverse correlation with crypto
- **S&P 500 / NASDAQ futures**: Risk appetite proxy
- **Gold price**: Safe haven comparison
- **VIX (Volatility Index)**: Fear gauge
- **US Treasury yields**: Macro liquidity conditions

### 2.6 Derived Statistical Features

- **Returns at multiple horizons**: 1, 5, 15, 60, 240 minute returns
- **Rolling statistics**: Mean, std, skew, kurtosis over windows (10, 30, 60, 120 bars)
- **Autocorrelation**: Lagged correlation coefficients
- **Hurst exponent** (rolling): Trending vs mean-reverting regime detection
- **Realized volatility**: Calculated from high-frequency returns

### Feature Engineering Recommendation

Start with **~50-80 features** across these categories. Use the TFT's built-in variable selection to automatically determine feature importance, then prune features with near-zero attention weights after initial training.

---

## 3. Multi-Timeframe Analysis

### Why Multiple Timeframes?

The original approach used only 1-minute candles. Multi-timeframe analysis is a fundamental technique in quantitative trading that significantly improves signal quality by providing a "top-down" view that filters out noise and false signals.

### Recommended Timeframe Hierarchy

| Timeframe | Purpose | Features |
|---|---|---|
| **1 minute** | Execution timing, microstructure | Raw OHLCV, volume, tick count |
| **5 minutes** | Short-term momentum | RSI, MACD, Bollinger Bands |
| **15 minutes** | Intraday trend | Moving averages, trend strength |
| **1 hour** | Intraday bias/direction | Support/resistance levels, session VWAP |
| **4 hours** | Swing direction | Ichimoku, ADX, key S/R zones |
| **Daily** | Overall trend context | 50/200 MA, daily ATR, volume profile |

### Implementation Approaches

#### Approach A: Feature Concatenation (Simplest)

Compute technical indicators at each timeframe and concatenate into a single feature vector:

```
features = [
    1min_ohlcv,          # shape: (lookback, 5)
    5min_indicators,      # shape: (lookback/5, N_indicators)
    15min_indicators,     # shape: (lookback/15, N_indicators)
    1hr_indicators,       # shape: (lookback/60, N_indicators)
]
```

**Pros**: Simple, works with any architecture
**Cons**: Alignment challenges, feature vector can be very wide

#### Approach B: Hierarchical Encoding (Recommended)

Use separate encoders for each timeframe, then fuse:

```
1min_encoder → 1min_embedding
5min_encoder → 5min_embedding
15min_encoder → 15min_embedding
1hr_encoder  → 1hr_embedding

Cross-attention fusion → final_embedding → prediction
```

**Pros**: Each encoder specializes, cross-attention learns inter-timeframe relationships
**Cons**: More complex architecture, more parameters

#### Approach C: TFT with Multi-Timeframe Static/Known Inputs

The TFT can handle this naturally:
- **Time-varying known inputs**: Higher timeframe indicators (they're known at prediction time since they use past data)
- **Time-varying observed inputs**: 1-minute OHLCV and derived features
- **Static inputs**: Asset identifier, exchange identifier

This is the recommended approach as it leverages TFT's built-in variable selection to determine which timeframe features are most relevant.

#### Approach D: Multi-Scale Patching (PatchTST-style)

Use different patch sizes corresponding to different timeframes:
- Patch size 1: 1-minute resolution
- Patch size 5: 5-minute resolution
- Patch size 15: 15-minute resolution
- Patch size 60: 1-hour resolution

Feed all patch scales into the Transformer with positional encodings that encode both position and scale.

### Multi-Timeframe Recommendation

Start with **Approach C (TFT with multi-timeframe inputs)** for simplicity and interpretability. Consider **Approach B (Hierarchical Encoding)** as a more advanced alternative if performance plateaus.

---

## 4. Training Strategies

### Critical Issues with Naive Train/Test Splits

The original approach likely used a simple sequential split (train on first X%, test on last Y%). This is **insufficient** for financial data due to:

1. **Lookahead bias**: Features computed on future data leaking into training
2. **Non-stationarity**: Market regimes change over time
3. **Temporal autocorrelation**: Adjacent samples are highly correlated, inflating metrics
4. **Overfitting to specific market conditions**: Model memorizes regime-specific patterns

### 4.1 Walk-Forward Validation (Expanding Window)

The most realistic simulation of live deployment:

```
Train: [----Jan-Jun----]  Test: [Jul]
Train: [----Jan-Jul----]  Test: [Aug]
Train: [----Jan-Aug----]  Test: [Sep]
...
```

**Pros**: Most realistic, tests on every period, model always trained on all available past data
**Cons**: Expensive (retrain for each fold), single path through history (high variance)

### 4.2 Sliding Window Walk-Forward

Fixed training window size that slides forward:

```
Train: [Jan-Jun]  Test: [Jul]
Train: [Feb-Jul]  Test: [Aug]
Train: [Mar-Aug]  Test: [Sep]
...
```

**Pros**: Adapts to non-stationarity (older data gets dropped), faster training per fold
**Cons**: Loses oldest data, need to choose window size carefully

### 4.3 Combinatorial Purged Cross-Validation (CPCV)

Developed by Marcos López de Prado ([SSRN, 2018](https://reasonabledeviations.com/notes/adv_fin_ml/)):

- Constructs multiple train-test splits from historical data
- **Purging**: Removes training observations whose labels overlap with test labels (prevents lookahead bias)
- **Embargo**: Adds a gap period between train and test sets to prevent information leakage from autocorrelation

**Why it's superior**: Produces a distribution of performance estimates (not a single number), enabling statistical significance testing and much lower probability of backtest overfitting (lower PBO, superior Deflated Sharpe Ratio).

### 4.4 Recommended Training Protocol

```
1. Primary validation: Walk-forward with expanding window
   - Retrain every week on all data up to that point
   - Test on the next week
   - Minimum 52 test periods (1 year of weekly tests)

2. Confirmation: CPCV for statistical significance
   - 10-fold CPCV with 5% purge window and 1% embargo
   - Compute Probability of Backtest Overfitting (PBO)
   - Reject model if PBO > 0.50

3. Train/validation/test split:
   - Training: 70% (oldest data)
   - Validation: 15% (hyperparameter tuning)
   - Test: 15% (final evaluation, touched ONCE)
   - Purge gap: 1000 bars (buffer between sets)

4. Final deployment training:
   - Retrain on 100% of data with best hyperparameters
   - Implement online learning with periodic retraining
```

### 4.5 Additional Training Best Practices

- **Shuffle within training folds only**: Never shuffle across train/test boundaries
- **Normalize per-window**: Compute normalization statistics on training data only, apply to test
- **Sample weighting**: Weight recent samples higher (exponential decay) to emphasize current regime
- **Curriculum learning**: Start training on "easy" market conditions (trending), gradually introduce harder samples (choppy, volatile)

---

## 5. Loss Functions

### Problems with Standard MSE/MAE

The original approach almost certainly used Mean Squared Error (MSE). For trading applications, this is suboptimal because:

1. **MSE treats all errors equally** — a 1% error where direction is correct is penalized the same as 1% where direction is wrong, but the trading consequences are vastly different
2. **MSE optimizes for price accuracy, not profitability** — a model can have low MSE but terrible trading performance
3. **MSE doesn't account for the asymmetry** of trading gains vs losses

### 5.1 Directional Loss Function — **Recommended Primary Loss**

Penalizes predictions that get the direction wrong more heavily:

```python
def directional_loss(y_true, y_pred, alpha=2.0):
    """
    Penalizes wrong-direction predictions by factor alpha.
    """
    mse = (y_true - y_pred) ** 2
    direction_correct = tf.sign(y_true) == tf.sign(y_pred)
    penalty = tf.where(direction_correct, 1.0, alpha)
    return tf.reduce_mean(mse * penalty)
```

Research shows a directional loss function repeatedly yields trading decisions that outperform those based on squared, absolute, percentage, or asymmetric losses, as measured by Sharpe ratio and profits per trade.

### 5.2 Asymmetric Loss Function

Penalizes overestimation and underestimation differently (useful when missing a downward move is costlier than missing an upward move):

```python
def asymmetric_loss(y_true, y_pred, alpha_over=1.0, alpha_under=2.0):
    """
    Different penalties for over vs under prediction.
    """
    error = y_true - y_pred
    loss = tf.where(
        error > 0,
        alpha_under * error ** 2,  # underpredicted (missed upside)
        alpha_over * error ** 2     # overpredicted (missed downside)
    )
    return tf.reduce_mean(loss)
```

### 5.3 Return-Weighted Loss

Weights the MSE by the magnitude of the actual return — larger moves matter more:

```python
def return_weighted_loss(y_true, y_pred):
    """
    Weights errors by the magnitude of the actual return.
    """
    returns = tf.abs(y_true)
    mse = (y_true - y_pred) ** 2
    weighted = mse * (1.0 + returns)
    return tf.reduce_mean(weighted)
```

Research shows this achieved **61.73% annual return with Sharpe ratio of 1.18** — substantially better than MSE-trained models.

### 5.4 Sharpe Ratio Loss (Differentiable)

Directly optimizes for risk-adjusted returns:

```python
def sharpe_loss(y_true, y_pred, risk_free_rate=0.0):
    """
    Differentiable Sharpe ratio as loss (negative for minimization).
    y_pred: predicted returns used as position sizing signal
    y_true: actual returns
    """
    portfolio_returns = y_pred * y_true  # position * actual return
    mean_return = tf.reduce_mean(portfolio_returns) - risk_free_rate
    std_return = tf.math.reduce_std(portfolio_returns) + 1e-8
    sharpe = mean_return / std_return
    return -sharpe  # negative because we minimize
```

Research shows Sharpe ratio loss more than doubles the Sharpe and triples the D-ratio vs standard MSE.

### 5.5 Combined Loss — **Recommended**

```python
def combined_trading_loss(y_true, y_pred):
    """
    Weighted combination of MSE (accuracy) + directional penalty + return weighting.
    """
    mse = tf.reduce_mean((y_true - y_pred) ** 2)

    # Directional component
    direction_match = tf.cast(
        tf.sign(y_true) == tf.sign(y_pred), tf.float32
    )
    direction_penalty = tf.reduce_mean(1.0 - direction_match)

    # Return-weighted component
    returns = tf.abs(y_true)
    weighted_mse = tf.reduce_mean((y_true - y_pred) ** 2 * (1.0 + returns))

    return 0.3 * mse + 0.4 * direction_penalty + 0.3 * weighted_mse
```

### Loss Function Recommendation

Use the **Combined Trading Loss** as the primary loss function. Compare against pure directional loss and Sharpe loss during validation. The best loss function may vary by market regime — consider using different losses for trending vs ranging markets.

---

## 6. Regularization

### Why Financial Data Is Especially Prone to Overfitting

- **Low signal-to-noise ratio**: Most price movement is noise
- **Non-stationarity**: Patterns that work in one period may not work in another
- **Regime changes**: Bull markets, bear markets, ranging markets have different dynamics
- **Limited effective sample size**: Despite having many bars, the number of independent "market events" is much smaller

### 6.1 Dropout Strategies

**Standard Dropout** (rate 0.1-0.3 for financial data):
- Apply to dense layers and between recurrent layers
- Lower rates than typical deep learning (financial signal is weak)

**Monte Carlo (MC) Dropout** — **Recommended for uncertainty estimation**:
- Keep dropout active during inference
- Run 30-100 forward passes per prediction
- Compute mean (prediction) and variance (uncertainty)
- **Trading application**: Only trade when prediction uncertainty is below threshold
- This is especially valuable for financial predictions where "I don't know" is a valid and profitable output

**Adaptive Dropout**:
- Recent research shows adaptive dropout with Hellinger divergence penalty achieves Test R^2 of 0.97 vs 0.63 for no-dropout LSTM ([SSRN, 2025](https://papers.ssrn.com/sol3/Delivery.cfm/e55c39f5-95ac-4c3c-bc1e-c7e97c7e61e9-MECA.pdf?abstractid=5363522&mirid=1))

### 6.2 L1 and L2 Regularization

- **L2 (Weight Decay)**: Coefficient 0.01-0.001, prevents weights from growing too large
- **L1 (Lasso)**: Encourages sparsity, effectively performs feature selection in the weight space
- Apply to all dense layers, with potentially different coefficients per layer

### 6.3 Early Stopping

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,          # Financial data is noisy; need more patience
    restore_best_weights=True,
    min_delta=1e-5
)
```

**Critical**: Monitor validation loss on a **temporally separate** validation set, not a random split.

### 6.4 Data Augmentation for Financial Time Series

Since financial data is limited and noisy, augmentation helps ([arXiv, 2020](https://ar5iv.labs.arxiv.org/html/2010.15111)):

- **Jittering/Noise injection**: Add small Gaussian noise (σ = 0.01-0.05 of std) to training inputs. Makes model robust to noise without changing patterns.
- **Window slicing**: Generate sub-sequences of varying lengths from full sequences. Increases effective dataset size.
- **Magnitude warping**: Slightly scale the values up/down (×0.95 to ×1.05). Teaches invariance to absolute price levels.
- **Time warping**: Slightly stretch/compress time axis. Teaches invariance to speed of price movement.
- **Performance improvement**: Up to 4x performance improvement on small datasets.

### 6.5 Regime-Aware Regularization

- **Regime detection**: Use HMM or rolling statistics to classify market regimes (trending up, trending down, ranging, volatile)
- **Separate models per regime**: Train specialized sub-models for each regime
- **Regime-weighted loss**: Weight training samples by similarity to current regime

### 6.6 Additional Techniques

- **Gradient clipping** (max norm 1.0): Prevents exploding gradients in RNNs
- **Learning rate scheduling**: Cosine annealing with warm restarts
- **Batch normalization** / **Layer normalization**: Stabilizes training
- **Label smoothing**: Softens classification targets (for direction prediction)
- **Mixup augmentation**: Interpolate between training samples

### Regularization Recommendation

Apply the following stack:
1. Adaptive dropout (0.1-0.2) on all dense/recurrent layers
2. L2 weight decay (0.001)
3. Early stopping (patience 15) on walk-forward validation loss
4. MC Dropout at inference for uncertainty-gated trading
5. Jittering + magnitude warping data augmentation
6. Gradient clipping (max norm 1.0)

---

## 7. Ensemble Methods

### Why Ensembles for Financial Prediction

Ensemble methods are particularly powerful for financial data because:
- **Reduces model risk**: No single model architecture dominates all market regimes
- **Lowers variance**: Averaging predictions smooths noise
- **Empirical evidence**: Stacking achieves 81.80% accuracy, 81.49% F1-score, and 88.43% AUC-ROC, outperforming individual models significantly ([Springer, 2025](https://link.springer.com/article/10.1007/s44163-025-00519-y))

### 7.1 Stacking Ensemble — **Recommended**

```
Layer 1 (Base Models):
  - TFT → prediction_1
  - LSTM → prediction_2
  - TCN → prediction_3
  - XGBoost (on handcrafted features) → prediction_4

Layer 2 (Meta-Learner):
  - XGBoost or Ridge Regression
  - Input: [prediction_1, prediction_2, prediction_3, prediction_4]
  - Output: final_prediction
```

**Why XGBoost as meta-learner**: It handles non-linearity, has built-in regularization, prevents overfitting, and learns which base model to trust in different conditions.

**Critical rule**: Train meta-learner on **out-of-fold predictions only** to prevent information leakage.

### 7.2 Model Diversity Strategy

For maximum ensemble benefit, base models should be **diverse** — different architectures, different features, different timeframes:

| Base Model | Architecture | Primary Features | Timeframe Focus |
|---|---|---|---|
| Model A | TFT | All features + multi-timeframe | 1min + 15min + 1hr |
| Model B | LSTM | Technical indicators + volume | 5min |
| Model C | TCN | Raw OHLCV + microstructure | 1min |
| Model D | XGBoost | Handcrafted statistical features | 15min |
| Model E | LightGBM | On-chain + cross-asset features | 1hr |

### 7.3 Dynamic Ensemble Weighting

Rather than fixed weights, adjust ensemble weights based on recent performance:

```python
def dynamic_weights(recent_errors, lookback=100, temperature=1.0):
    """
    Weight models inversely proportional to recent error.
    """
    recent_mse = [np.mean(e[-lookback:]**2) for e in recent_errors]
    inv_errors = [1.0 / (mse + 1e-8) for mse in recent_mse]
    weights = softmax(np.array(inv_errors) / temperature)
    return weights
```

This automatically reduces weight on models that are performing poorly in the current regime.

### 7.4 Uncertainty-Gated Ensemble

Combine MC Dropout uncertainty with ensemble:

1. Each base model produces prediction + uncertainty (via MC Dropout)
2. Weight each model's prediction inversely by its uncertainty
3. Only generate a trading signal when **total ensemble uncertainty** is below threshold

This is powerful because it says "I don't know" during unpredictable periods — avoiding losses from random noise.

### Ensemble Recommendation

Implement a **3-5 model stacking ensemble** with:
- TFT as primary (highest expected weight)
- LSTM and TCN as complementary architectures
- XGBoost on engineered features for model diversity
- Dynamic weighting based on rolling performance
- Uncertainty gating to skip low-confidence predictions

---

## 8. Key Improvements Over the Original Approach

### Summary of the Original Approach (from video)

| Aspect | Original | Problem |
|---|---|---|
| Architecture | Simple dense NN (~200K params) | Cannot capture temporal dependencies |
| Features | Raw 1-min OHLCV only | Missing massive amounts of predictive signal |
| Lookback | 150 bars (2.5 hours) | Too short for multi-timeframe patterns |
| Prediction | Single next candle close | Autoregressive error accumulation |
| Validation | Unknown (likely simple split) | Likely lookahead bias |
| Loss function | Likely MSE | Not optimized for trading profitability |
| Regularization | Unknown | Likely overfitting to noise |
| Data | ~51,773 minutes (~36 days) | Far too little data |

### Recommended Improvements (Priority Order)

#### 1. **Architecture: Dense NN → TFT** (Impact: Critical)

The original uses a dense network that treats all 150 bars × 5 features as a flat vector — destroying all temporal structure. TFT explicitly models temporal dependencies with attention, handles multiple input types, and provides interpretability.

**Expected improvement**: 20-40% reduction in RMSE, significant improvement in directional accuracy.

#### 2. **Multi-Step Prediction Instead of Autoregressive** (Impact: Critical)

The original predicts one candle, then feeds that prediction back as input for the next — causing **error accumulation** where small errors compound exponentially.

**Fix**: Use TFT's native multi-horizon forecasting to predict all N future steps simultaneously. Alternatively, predict probability distributions rather than point estimates.

#### 3. **Feature Engineering: 5 features → 50-80 features** (Impact: High)

Adding technical indicators, volume analysis, temporal features, and cross-asset data provides dramatically more signal for the model to learn from.

#### 4. **Data Quantity: 36 days → 2+ years** (Impact: High)

36 days is wildly insufficient. Need minimum 2 years of 1-minute data (~1M candles) to capture diverse market conditions (bull, bear, ranging, volatile, quiet).

#### 5. **Multi-Timeframe: 1min only → 1min through Daily** (Impact: High)

Adding 5min, 15min, 1hr, 4hr, and daily context provides the model with multi-scale pattern recognition — critical for understanding whether a 1-minute move is noise or the start of a larger trend.

#### 6. **Training Validation: Unknown → Walk-Forward + CPCV** (Impact: High)

Proper temporal validation prevents overfitting to specific market conditions and gives reliable performance estimates. Without this, any reported accuracy is meaningless.

#### 7. **Loss Function: MSE → Combined Directional/Return-Weighted** (Impact: Medium-High)

Directly optimizing for trading-relevant metrics (direction, magnitude-weighted accuracy) instead of raw price error aligns the model's optimization with actual profitability.

#### 8. **Ensemble: Single Model → 3-5 Model Stack** (Impact: Medium)

Combining diverse architectures reduces model risk and improves robustness across market regimes.

#### 9. **Regularization Stack** (Impact: Medium)

Adding dropout, weight decay, data augmentation, and uncertainty estimation prevents overfitting to noise and enables the model to express "I don't know" — critical for avoiding losses.

#### 10. **Uncertainty Quantification: None → MC Dropout** (Impact: Medium)

The original makes predictions with false confidence in all market conditions. Adding uncertainty estimation enables selective trading — only acting when the model is confident.

### Projected Architecture

```
Input Pipeline:
  └─ 1-min OHLCV (raw)
  └─ 5-min technical indicators (RSI, MACD, BB, etc.)
  └─ 15-min trend indicators (MAs, ADX, Ichimoku)
  └─ 1-hr support/resistance + session VWAP
  └─ On-chain features (exchange flows, funding rates)
  └─ Cross-asset features (DXY, S&P futures, ETH/BTC)
  └─ Temporal encodings (hour, day-of-week, sin/cos)
       │
       ▼
  Feature Normalization (per-window, rolling stats)
       │
       ▼
  ┌─────────────────────────────────────┐
  │  Ensemble Base Models               │
  │  ├─ TFT (primary, multi-horizon)    │
  │  ├─ LSTM (sequential patterns)      │
  │  ├─ TCN (parallel, fast inference)  │
  │  └─ XGBoost (engineered features)   │
  └─────────────────────────────────────┘
       │
       ▼
  Meta-Learner (XGBoost / Ridge)
       │
       ▼
  MC Dropout Uncertainty Estimation
       │
       ▼
  Uncertainty Gate (trade only if confidence > threshold)
       │
       ▼
  Multi-Step Price Prediction (next 5-15 candles)
```

---

## Key References

1. [Lim et al. - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
2. [Nie et al. - PatchTST: A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730)
3. [López de Prado - Advances in Financial Machine Learning](https://reasonabledeviations.com/notes/adv_fin_ml/) (Purged CV, CPCV)
4. [MDPI - From LSTM to GPT-2: Deep Learning Architectures for Crypto Forecasting](https://www.mdpi.com/2073-8994/18/1/32)
5. [Springer - Machine Learning Approaches to Crypto Trading Optimization](https://link.springer.com/article/10.1007/s44163-025-00519-y)
6. [ScienceDirect - Bitcoin Price Direction Using On-Chain Data](https://www.sciencedirect.com/science/article/pii/S266682702500057X)
7. [Dessain - Custom Loss Functions for Asset Return Prediction](https://www.researchgate.net/publication/377071528)
8. [SSRN - Dropout Regularization for Financial Time Series LSTMs](https://papers.ssrn.com/sol3/Delivery.cfm/e55c39f5-95ac-4c3c-bc1e-c7e97c7e61e9-MECA.pdf?abstractid=5363522)
9. [arXiv - Data Augmentation for Financial Time Series Classification](https://ar5iv.labs.arxiv.org/html/2010.15111)
10. [Springer - Multi-Timeframe Confidence-Threshold Framework](https://www.mdpi.com/1999-4893/18/12/758)
11. [MDPI - TFT-Based Trading with On-Chain and Technical Indicators](https://www.mdpi.com/2079-8954/13/6/474)
12. [CFA Institute - Ensemble Learning in Investment](https://rpc.cfainstitute.org/research/foundation/2025/chapter-4-ensemble-learning-investment)

---

## Technology Stack Recommendation

| Component | Recommended | Alternative |
|---|---|---|
| **Framework** | PyTorch + PyTorch Lightning | TensorFlow/Keras |
| **TFT Implementation** | [pytorch-forecasting](https://pytorch-forecasting.readthedocs.io/) | Darts, NeuralForecast |
| **Feature Engineering** | pandas-ta, TA-Lib | Custom implementations |
| **Gradient Boosting** | XGBoost, LightGBM | CatBoost |
| **Experiment Tracking** | Weights & Biases (wandb) | MLflow, TensorBoard |
| **Data Storage** | Parquet files (columnar, fast) | TimescaleDB, InfluxDB |
| **Hyperparameter Tuning** | Optuna | Ray Tune |
| **On-Chain Data** | Glassnode API, CryptoQuant | Santiment |

### Why PyTorch over TensorFlow

The original used TensorFlow/Keras. We recommend switching to **PyTorch** because:
1. `pytorch-forecasting` provides a production-ready TFT implementation
2. PyTorch Lightning reduces boilerplate and standardizes training loops
3. Better debugging experience (eager execution by default)
4. Dominant framework in research (easier to implement latest papers)
5. Optuna integrates seamlessly for hyperparameter optimization
