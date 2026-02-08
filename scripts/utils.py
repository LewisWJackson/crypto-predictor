#!/usr/bin/env python3
"""
Shared utility functions for the crypto-predictor data pipeline.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_PARQUET = RAW_DIR / "btc_usdt_1m.parquet"
RAW_CSV = RAW_DIR / "btc_usdt_1m.csv"

CANDLE_MS = 60_000  # 1 minute in milliseconds


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load raw OHLCV data from parquet (preferred) or CSV fallback."""
    if path is not None:
        p = Path(path)
        if p.suffix == ".csv":
            return pd.read_csv(p)
        return pd.read_parquet(p)

    if RAW_PARQUET.exists():
        return pd.read_parquet(RAW_PARQUET)
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)

    raise FileNotFoundError(
        f"No raw data found. Expected {RAW_PARQUET} or {RAW_CSV}. "
        "Run download_data.py first."
    )


def save_arrays(path: Path, **arrays: np.ndarray) -> None:
    """Save numpy arrays to a compressed .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load numpy arrays from a .npz file."""
    data = np.load(str(path))
    return dict(data)


# ---------------------------------------------------------------------------
# Data validation helpers
# ---------------------------------------------------------------------------

def validate_ohlcv(df: pd.DataFrame) -> list[str]:
    """Return a list of data quality issues found in OHLCV data."""
    issues = []

    if df.empty:
        return ["DataFrame is empty"]

    # Duplicates
    dupes = df["timestamp"].duplicated().sum()
    if dupes > 0:
        issues.append(f"{dupes} duplicate timestamps")

    # Gaps
    ts_sorted = df["timestamp"].sort_values()
    diffs = ts_sorted.diff().dropna()
    gaps = diffs[diffs > CANDLE_MS * 1.5]
    if len(gaps) > 0:
        max_gap = int(gaps.max() / CANDLE_MS)
        issues.append(f"{len(gaps)} gaps (largest: {max_gap} minutes)")

    # OHLC consistency
    bad_ohlc = (
        (df["low"] > df["open"]) | (df["low"] > df["close"])
        | (df["high"] < df["open"]) | (df["high"] < df["close"])
    ).sum()
    if bad_ohlc > 0:
        issues.append(f"{bad_ohlc} invalid OHLC relationships")

    # Negative volume
    neg_vol = (df["volume"] < 0).sum()
    if neg_vol > 0:
        issues.append(f"{neg_vol} negative volume entries")

    # Sort order
    if not ts_sorted.is_monotonic_increasing:
        issues.append("Data not sorted chronologically")

    return issues


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing 1-minute candles in OHLCV data.

    Creates entries for missing timestamps where close=open=previous close,
    high=low=close, volume=0.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    start_ts = df["timestamp"].iloc[0]
    end_ts = df["timestamp"].iloc[-1]
    full_range = np.arange(start_ts, end_ts + CANDLE_MS, CANDLE_MS)

    full_df = pd.DataFrame({"timestamp": full_range})
    merged = full_df.merge(df, on="timestamp", how="left")

    # Forward-fill price columns using previous close
    merged["close"] = merged["close"].ffill()
    merged["open"] = merged["open"].fillna(merged["close"])
    merged["high"] = merged["high"].fillna(merged["close"])
    merged["low"] = merged["low"].fillna(merged["close"])
    merged["volume"] = merged["volume"].fillna(0.0)

    return merged.reset_index(drop=True)
