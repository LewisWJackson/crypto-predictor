"""
Feature engineering for crypto price prediction.

Computes all 43 features as specified in the architecture design:
- Price-derived (6): log returns at 1/5/15/60, high_low_range, close_open_range
- Trend indicators (8): SMA 20/50/200, EMA 9/21, MACD line/signal/histogram
- Momentum (7): RSI, Stoch K/D, Williams %R, CCI, ROC, momentum
- Volatility (7): BB upper/lower/bandwidth/pctb, ATR, rolling vol 20/60
- Volume (6): volume SMA ratio, OBV normalized, VWAP distance, MFI, volume ROC, A/D line
- Rolling stats (3): skew, kurtosis, autocorrelation
- Temporal encodings (6): sin/cos for hour, day, minute

Target: log_return_15 = ln(close[t+15] / close[t])
"""

import numpy as np
import pandas as pd
import ta


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 6 price-derived features."""
    close = df["close"]

    df["log_return_1"] = np.log(close / close.shift(1))
    df["log_return_5"] = np.log(close / close.shift(5))
    df["log_return_15"] = np.log(close / close.shift(15))
    df["log_return_60"] = np.log(close / close.shift(60))
    df["high_low_range"] = (df["high"] - df["low"]) / close
    df["close_open_range"] = (close - df["open"]) / close

    return df


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 8 trend indicator features, all normalized by close."""
    close = df["close"]

    # SMA normalized as distance from close
    df["sma_20"] = ta.trend.sma_indicator(close, window=20) / close - 1
    df["sma_50"] = ta.trend.sma_indicator(close, window=50) / close - 1
    df["sma_200"] = ta.trend.sma_indicator(close, window=200) / close - 1

    # EMA normalized as distance from close
    df["ema_9"] = ta.trend.ema_indicator(close, window=9) / close - 1
    df["ema_21"] = ta.trend.ema_indicator(close, window=21) / close - 1

    # MACD — compute raw then normalize by close
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd_line"] = macd.macd() / close
    df["macd_signal"] = macd.macd_signal() / close
    df["macd_histogram"] = macd.macd_diff() / close

    return df


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 7 momentum features, all scaled to ~0-1."""
    close = df["close"]

    # RSI scaled to 0-1
    df["rsi_14"] = ta.momentum.rsi(close, window=14) / 100.0

    # Stochastic oscillator scaled to 0-1
    stoch = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=close,
        window=14, smooth_window=3,
    )
    df["stoch_k"] = stoch.stoch() / 100.0
    df["stoch_d"] = stoch.stoch_signal() / 100.0

    # Williams %R scaled to 0-1 (raw is -100 to 0, divide by 100 gives -1 to 0)
    df["williams_r"] = ta.momentum.williams_r(
        high=df["high"], low=df["low"], close=close, lbp=14,
    ) / 100.0

    # CCI normalized around +/-1
    df["cci_20"] = ta.trend.cci(
        high=df["high"], low=df["low"], close=close, window=20,
    ) / 200.0

    # Rate of Change
    df["roc_10"] = ta.momentum.roc(close, window=10) / 100.0

    # Momentum (manual)
    df["momentum_10"] = (close - close.shift(10)) / close.shift(10)

    return df


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 7 volatility features."""
    close = df["close"]

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_middle = bb.bollinger_mavg()

    df["bb_upper"] = (bb_upper - close) / close
    df["bb_lower"] = (close - bb_lower) / close
    df["bb_bandwidth"] = (bb_upper - bb_lower) / bb_middle
    df["bb_pctb"] = (close - bb_lower) / (bb_upper - bb_lower)

    # ATR normalized by close
    df["atr_14"] = ta.volatility.average_true_range(
        high=df["high"], low=df["low"], close=close, window=14,
    ) / close

    # Rolling volatility of log returns
    log_ret_1 = np.log(close / close.shift(1))
    df["rolling_volatility_20"] = log_ret_1.rolling(window=20).std()
    df["rolling_volatility_60"] = log_ret_1.rolling(window=60).std()

    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 6 volume features."""
    close = df["close"]
    volume = df["volume"]

    # Volume SMA ratio
    vol_sma = volume.rolling(window=20).mean()
    df["volume_sma_ratio"] = volume / (vol_sma + 1e-8)

    # OBV normalized
    obv = ta.volume.on_balance_volume(close, volume)
    obv_sma_abs = obv.abs().rolling(window=20).mean()
    df["obv_normalized"] = obv / (obv_sma_abs + 1e-8)

    # VWAP distance (cumulative VWAP using rolling window approach)
    typical_price = (df["high"] + df["low"] + close) / 3.0
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    vwap = cum_tp_vol / (cum_vol + 1e-8)
    df["vwap_distance"] = (close - vwap) / close

    # Money Flow Index
    df["mfi_14"] = ta.volume.money_flow_index(
        high=df["high"], low=df["low"], close=close, volume=volume, window=14,
    ) / 100.0

    # Volume Rate of Change
    df["volume_roc"] = (volume - volume.shift(10)) / (volume.shift(10) + 1e-8)

    # Accumulation/Distribution line normalized
    ad = ta.volume.acc_dist_index(
        high=df["high"], low=df["low"], close=close, volume=volume,
    )
    ad_sma_abs = ad.abs().rolling(window=20).mean()
    df["ad_line_normalized"] = ad / (ad_sma_abs + 1e-8)

    return df


def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 3 rolling statistical features on log returns."""
    close = df["close"]
    log_ret_1 = np.log(close / close.shift(1))

    df["rolling_skew_60"] = log_ret_1.rolling(window=60).skew()
    df["rolling_kurtosis_60"] = log_ret_1.rolling(window=60).kurt()

    # Autocorrelation lag=1 with window=60
    df["autocorr_lag1"] = log_ret_1.rolling(window=60).apply(
        lambda x: x.autocorr(lag=1), raw=False,
    )

    return df


def compute_temporal_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 6 temporal encoding features (sin/cos cycles)."""
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    hour = ts.dt.hour + ts.dt.minute / 60.0
    day_of_week = ts.dt.dayofweek
    minute = ts.dt.minute

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    df["minute_of_hour_sin"] = np.sin(2 * np.pi * minute / 60.0)
    df["minute_of_hour_cos"] = np.cos(2 * np.pi * minute / 60.0)

    return df


def compute_target(df: pd.DataFrame, horizons: list = None) -> pd.DataFrame:
    """Compute target variable(s): log return over next N bars.

    target = ln(close[t+N] / close[t])

    Note: This uses FUTURE data (shift(-N)) which is correct for the target
    but must NEVER be used as an input feature.

    Args:
        df: DataFrame with 'close' column.
        horizons: List of forward horizons to compute targets for.
            Defaults to [15] for backward compatibility.
            E.g., [5, 15] computes target_log_return_5 and target_log_return_15.
    """
    if horizons is None:
        horizons = [15]
    for h in horizons:
        df[f"target_log_return_{h}"] = np.log(df["close"].shift(-h) / df["close"])
    return df


# Feature name lists for dataset creation
PRICE_FEATURES = [
    "log_return_1", "log_return_5", "log_return_15", "log_return_60",
    "high_low_range", "close_open_range",
]

TREND_FEATURES = [
    "sma_20", "sma_50", "sma_200", "ema_9", "ema_21",
    "macd_line", "macd_signal", "macd_histogram",
]

MOMENTUM_FEATURES = [
    "rsi_14", "stoch_k", "stoch_d", "williams_r",
    "cci_20", "roc_10", "momentum_10",
]

VOLATILITY_FEATURES = [
    "bb_upper", "bb_lower", "bb_bandwidth", "bb_pctb",
    "atr_14", "rolling_volatility_20", "rolling_volatility_60",
]

VOLUME_FEATURES = [
    "volume_sma_ratio", "obv_normalized", "vwap_distance",
    "mfi_14", "volume_roc", "ad_line_normalized",
]

ROLLING_STATS_FEATURES = [
    "rolling_skew_60", "rolling_kurtosis_60", "autocorr_lag1",
]

TEMPORAL_FEATURES = [
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "minute_of_hour_sin", "minute_of_hour_cos",
]

# All time-varying unknown reals (37 features)
TIME_VARYING_UNKNOWN_REALS = (
    PRICE_FEATURES + TREND_FEATURES + MOMENTUM_FEATURES
    + VOLATILITY_FEATURES + VOLUME_FEATURES + ROLLING_STATS_FEATURES
)

# All time-varying known reals (6 features)
TIME_VARYING_KNOWN_REALS = TEMPORAL_FEATURES

# All 43 feature names
ALL_FEATURES = TIME_VARYING_UNKNOWN_REALS + TIME_VARYING_KNOWN_REALS

TARGET_COLUMN = "target_log_return_15"

# All target horizons to compute (parquet will include all of these)
TARGET_HORIZONS = [5, 15]


def compute_all_features(df: pd.DataFrame, drop_na: bool = True, target_horizons: list = None) -> pd.DataFrame:
    """Compute all 43 features and target variable.

    Args:
        df: Raw OHLCV DataFrame with columns:
            timestamp, open, high, low, close, volume
        drop_na: If True, drop NaN warm-up rows (first ~200) and
                 NaN target rows (last 15).

    Returns:
        DataFrame with all features and target computed. Uses float32.
    """
    if target_horizons is None:
        target_horizons = TARGET_HORIZONS

    df = df.copy()

    # Compute all feature groups
    df = compute_price_features(df)
    df = compute_trend_features(df)
    df = compute_momentum_features(df)
    df = compute_volatility_features(df)
    df = compute_volume_features(df)
    df = compute_rolling_stats(df)
    df = compute_temporal_encodings(df)

    # Compute target(s) (uses future data — only for labels)
    df = compute_target(df, horizons=target_horizons)

    # Build list of target columns to check
    target_columns = [f"target_log_return_{h}" for h in target_horizons]

    if drop_na:
        # Drop rows where any feature or target is NaN
        cols_to_check = ALL_FEATURES + target_columns
        df = df.dropna(subset=cols_to_check).reset_index(drop=True)

    # Convert feature columns to float32
    float_cols = ALL_FEATURES + target_columns
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    return df
