#!/usr/bin/env python3
"""
Feature Engineering & Preprocessing Pipeline for Crypto Predictor

Loads raw OHLCV data, computes technical indicators and derived features,
normalizes appropriately, creates lookback windows, and splits into
train/val/test sets with NO data leakage.

Usage:
    python preprocess.py                          # Use defaults
    python preprocess.py --input data/raw/btc_usdt_1m.parquet
    python preprocess.py --lookback 128 --gap 60
    python preprocess.py --train-ratio 0.7 --val-ratio 0.15
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Add parent to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    PROCESSED_DIR,
    fill_gaps,
    get_logger,
    load_raw_data,
    save_arrays,
    validate_ohlcv,
)

log = get_logger("preprocess")


# =========================================================================
# 1. Feature Engineering
# =========================================================================

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns and percentage returns from close prices."""
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["pct_return"] = df["close"].pct_change()
    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a comprehensive set of technical indicators using the `ta` library."""
    import ta

    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- Trend Indicators ---

    # EMA (Exponential Moving Averages)
    df["ema_9"] = ta.trend.ema_indicator(close, window=9)
    df["ema_21"] = ta.trend.ema_indicator(close, window=21)
    df["ema_50"] = ta.trend.ema_indicator(close, window=50)

    # MACD (12, 26, 9)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()

    # --- Momentum Indicators ---

    # RSI (14)
    df["rsi_14"] = ta.momentum.rsi(close, window=14)

    # Stochastic RSI
    stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    df["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
    df["stoch_rsi_d"] = stoch_rsi.stochrsi_d()

    # --- Volatility Indicators ---

    # Bollinger Bands (20, 2)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ATR (14)
    df["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)

    # --- Volume Indicators ---

    # Volume SMA (20)
    df["volume_sma_20"] = ta.trend.sma_indicator(volume, window=20)
    # Volume ratio: current volume relative to 20-period average
    df["volume_ratio"] = volume / df["volume_sma_20"].replace(0, np.nan)

    return df


def compute_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute support/resistance levels from local minima/maxima.

    Uses rolling window to find local highs (resistance) and lows (support),
    then computes distance from current price to nearest levels.
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Rolling max/min as proxy for resistance/support
    df["resistance"] = high.rolling(window=window, center=False).max()
    df["support"] = low.rolling(window=window, center=False).min()

    # Distance from price to support/resistance (normalized by price)
    df["dist_to_resistance"] = (df["resistance"] - close) / close
    df["dist_to_support"] = (close - df["support"]) / close

    # Price position within support-resistance range
    sr_range = df["resistance"] - df["support"]
    df["sr_position"] = (close - df["support"]) / sr_range.replace(0, np.nan)

    return df


def compute_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional derived features useful for the model."""
    df = df.copy()

    # Candle body and wick features (normalized by close)
    df["body_size"] = (df["close"] - df["open"]).abs() / df["close"]
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]

    # Price momentum over various horizons
    for period in [5, 15, 60]:
        df[f"momentum_{period}"] = df["close"].pct_change(periods=period)

    # Volatility (rolling std of log returns)
    if "log_return" in df.columns:
        df["volatility_20"] = df["log_return"].rolling(window=20).std()
        df["volatility_60"] = df["log_return"].rolling(window=60).std()

    return df


# =========================================================================
# 2. Feature Selection & Normalization
# =========================================================================

# Features bounded in [0, 1] or similar fixed ranges -> MinMaxScaler
BOUNDED_FEATURES = [
    "rsi_14", "stoch_rsi_k", "stoch_rsi_d",
    "bb_pct", "sr_position",
]

# Features with unbounded distributions -> StandardScaler
UNBOUNDED_FEATURES = [
    "log_return", "pct_return",
    "ema_9", "ema_21", "ema_50",
    "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "atr_14",
    "volume_sma_20", "volume_ratio",
    "resistance", "support",
    "dist_to_resistance", "dist_to_support",
    "body_size", "upper_wick", "lower_wick",
    "momentum_5", "momentum_15", "momentum_60",
    "volatility_20", "volatility_60",
    # Raw OHLCV (normalized)
    "open", "high", "low", "close", "volume",
]

ALL_FEATURES = BOUNDED_FEATURES + UNBOUNDED_FEATURES


def select_and_order_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the feature columns we need, in a consistent order."""
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        log.warning(f"Missing features (will be excluded): {missing}")
    return df[available].copy()


def normalize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Normalize features using scalers fit ONLY on training data.

    Returns normalized numpy arrays and scaler metadata for inverse transforms.

    - Bounded features: MinMaxScaler (preserves [0,1] range semantics)
    - Unbounded features: StandardScaler (zero mean, unit variance)
    """
    bounded_cols = [c for c in BOUNDED_FEATURES if c in train_df.columns]
    unbounded_cols = [c for c in UNBOUNDED_FEATURES if c in train_df.columns]

    train_out = train_df.values.copy().astype(np.float32)
    val_out = val_df.values.copy().astype(np.float32)
    test_out = test_df.values.copy().astype(np.float32)

    col_list = list(train_df.columns)
    bounded_idx = [col_list.index(c) for c in bounded_cols]
    unbounded_idx = [col_list.index(c) for c in unbounded_cols]

    scaler_info = {"columns": col_list}

    # MinMax for bounded features
    if bounded_idx:
        mm_scaler = MinMaxScaler()
        train_out[:, bounded_idx] = mm_scaler.fit_transform(train_out[:, bounded_idx])
        val_out[:, bounded_idx] = mm_scaler.transform(val_out[:, bounded_idx])
        test_out[:, bounded_idx] = mm_scaler.transform(test_out[:, bounded_idx])
        scaler_info["minmax_min"] = mm_scaler.data_min_
        scaler_info["minmax_scale"] = mm_scaler.scale_
        scaler_info["minmax_idx"] = np.array(bounded_idx)

    # Standard for unbounded features
    if unbounded_idx:
        ss_scaler = StandardScaler()
        train_out[:, unbounded_idx] = ss_scaler.fit_transform(train_out[:, unbounded_idx])
        val_out[:, unbounded_idx] = ss_scaler.transform(val_out[:, unbounded_idx])
        test_out[:, unbounded_idx] = ss_scaler.transform(test_out[:, unbounded_idx])
        scaler_info["standard_mean"] = ss_scaler.mean_
        scaler_info["standard_scale"] = ss_scaler.scale_
        scaler_info["standard_idx"] = np.array(unbounded_idx)

    return train_out, val_out, test_out, scaler_info


# =========================================================================
# 3. Windowing
# =========================================================================

def create_windows(
    data: np.ndarray,
    lookback: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding lookback windows for sequence model input.

    For each position i, creates:
        X[i] = data[i : i + lookback]   (lookback bars of features)
        y[i] = the target at position i + lookback

    Target: next-bar log return (extracted from the feature matrix).
    The log_return column is expected to be the first feature (index 0).

    Returns:
        X: shape (num_samples, lookback, num_features)
        y: shape (num_samples,)
    """
    n_samples = len(data) - lookback
    if n_samples <= 0:
        raise ValueError(
            f"Data length ({len(data)}) must be > lookback ({lookback})"
        )

    num_features = data.shape[1]
    X = np.empty((n_samples, lookback, num_features), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.float32)

    # Target column: log_return is index 0 in ALL_FEATURES
    # (after feature selection, it's still the first in the output)
    target_idx = 0  # log_return

    for i in range(n_samples):
        X[i] = data[i : i + lookback]
        y[i] = data[i + lookback, target_idx]

    return X, y


# =========================================================================
# 4. Time-Series Train/Val/Test Split
# =========================================================================

def time_series_split(
    data: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    gap: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data chronologically with gaps to prevent leakage.

    The gap (in bars) between splits prevents the model from learning
    from data that overlaps with the lookback window of the next split.

    Timeline: [---train---|gap|---val---|gap|---test---]

    Args:
        data: 2D array of shape (n_bars, n_features), ordered chronologically
        train_ratio: proportion for training (default 0.70)
        val_ratio: proportion for validation (default 0.15)
        gap: number of bars to skip between splits (default 60 = 1 hour)

    Returns:
        train, val, test arrays
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + gap + int(n * val_ratio)

    train = data[:train_end]
    val = data[train_end + gap : val_end]
    test = data[val_end + gap :]

    return train, val, test


# =========================================================================
# 5. Main Pipeline
# =========================================================================

def run_pipeline(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    lookback: int = 256,
    gap: int = 60,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> None:
    """Execute the full preprocessing pipeline end-to-end."""

    out_dir = output_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load raw data ---
    log.info("Loading raw data...")
    df = load_raw_data(input_path)
    log.info(f"Loaded {len(df):,} candles")

    # --- Step 2: Validate & clean ---
    log.info("Validating data quality...")
    issues = validate_ohlcv(df)
    if issues:
        for issue in issues:
            log.warning(f"  {issue}")
    else:
        log.info("  All quality checks passed")

    # Sort and deduplicate
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Fill gaps
    initial_len = len(df)
    df = fill_gaps(df)
    filled = len(df) - initial_len
    if filled > 0:
        log.info(f"  Filled {filled:,} missing candles")

    log.info(f"Clean data: {len(df):,} candles")

    # --- Step 3: Feature engineering ---
    log.info("Computing features...")
    df = compute_returns(df)
    df = compute_technical_indicators(df)
    df = compute_support_resistance(df, window=20)
    df = compute_extra_features(df)

    # Drop initial NaN rows from indicators (warmup period)
    warmup = 60  # Conservative: longest indicator is EMA(50) + some lag
    df = df.iloc[warmup:].reset_index(drop=True)
    log.info(f"After indicator warmup: {len(df):,} candles")

    # --- Step 4: Select and reorder features ---
    feature_df = select_and_order_features(df)

    # Replace any remaining NaN/inf with 0
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feature_names = list(feature_df.columns)
    log.info(f"Feature matrix: {feature_df.shape[0]:,} rows x {feature_df.shape[1]} features")
    log.info(f"Features: {feature_names}")

    # --- Step 5: Time-series split (BEFORE normalization) ---
    log.info(f"Splitting data (train={train_ratio}, val={val_ratio}, "
             f"test={1-train_ratio-val_ratio:.2f}, gap={gap} bars)...")

    train_df_split, val_df_split, test_df_split = time_series_split(
        feature_df.values, train_ratio, val_ratio, gap
    )

    # Convert back to DataFrames for normalization
    train_feat = pd.DataFrame(train_df_split, columns=feature_names)
    val_feat = pd.DataFrame(val_df_split, columns=feature_names)
    test_feat = pd.DataFrame(test_df_split, columns=feature_names)

    log.info(f"  Train: {len(train_feat):,} | Val: {len(val_feat):,} | Test: {len(test_feat):,}")

    # --- Step 6: Normalize (fit on train only) ---
    log.info("Normalizing features (scalers fit on train set only)...")
    train_norm, val_norm, test_norm, scaler_info = normalize_features(
        train_feat, val_feat, test_feat
    )

    # --- Step 7: Create lookback windows ---
    log.info(f"Creating lookback windows (length={lookback})...")

    X_train, y_train = create_windows(train_norm, lookback)
    X_val, y_val = create_windows(val_norm, lookback)
    X_test, y_test = create_windows(test_norm, lookback)

    log.info(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    log.info(f"  X_val:   {X_val.shape}   | y_val:   {y_val.shape}")
    log.info(f"  X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

    # --- Step 8: Save outputs ---
    log.info("Saving processed data...")

    # Main training data
    save_arrays(
        out_dir / "train.npz",
        X=X_train, y=y_train,
    )
    save_arrays(
        out_dir / "val.npz",
        X=X_val, y=y_val,
    )
    save_arrays(
        out_dir / "test.npz",
        X=X_test, y=y_test,
    )

    # Scaler info for inverse transforms during inference
    scaler_arrays = {k: v for k, v in scaler_info.items() if isinstance(v, np.ndarray)}
    # Save column names separately as they're strings
    save_arrays(out_dir / "scalers.npz", **scaler_arrays)

    # Save feature names and metadata as a small npz
    save_arrays(
        out_dir / "metadata.npz",
        feature_names=np.array(feature_names),
        lookback=np.array([lookback]),
        gap=np.array([gap]),
        train_size=np.array([len(X_train)]),
        val_size=np.array([len(X_val)]),
        test_size=np.array([len(X_test)]),
        split_ratios=np.array([train_ratio, val_ratio, 1 - train_ratio - val_ratio]),
    )

    # Report file sizes
    total_mb = 0
    for f in out_dir.glob("*.npz"):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_mb += size_mb
        log.info(f"  {f.name}: {size_mb:.1f} MB")
    log.info(f"  Total: {total_mb:.1f} MB")

    log.info("Preprocessing complete!")


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess OHLCV data for crypto prediction model"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to raw data file (parquet or csv). Default: data/raw/btc_usdt_1m.parquet"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for processed files. Default: data/processed/"
    )
    parser.add_argument(
        "--lookback", type=int, default=256,
        help="Lookback window size in bars (default: 256)"
    )
    parser.add_argument(
        "--gap", type=int, default=60,
        help="Gap between train/val and val/test splits in bars (default: 60)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Training set proportion (default: 0.70)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Validation set proportion (default: 0.15)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input) if args.input else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        lookback=args.lookback,
        gap=args.gap,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
