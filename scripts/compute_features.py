"""
CLI script for computing features from raw OHLCV data.

Usage:
    python scripts/compute_features.py [--pair BTC_USDT] [--raw-dir data/raw] [--out-dir data/processed]

Loads raw parquet, computes all 43 features + target, saves processed parquet.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.features import compute_all_features, ALL_FEATURES, TARGET_COLUMN


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1-minute OHLCV data to a coarser timeframe.

    Args:
        df: DataFrame with timestamp (ms), open, high, low, close, volume.
        rule: Pandas resample rule, e.g. '15min', '5min'.

    Returns:
        Resampled DataFrame with the same columns.
    """
    df = df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("dt")

    resampled = df.resample(rule).agg({
        "timestamp": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["timestamp"]).reset_index(drop=True)

    resampled["timestamp"] = resampled["timestamp"].astype(np.int64)
    return resampled


def main():
    parser = argparse.ArgumentParser(description="Compute features from raw OHLCV data")
    parser.add_argument("--pair", default="btc_usdt", help="Trading pair (default: btc_usdt)")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory")
    parser.add_argument("--resample", default=None, help="Resample timeframe (e.g. 15min, 5min)")
    parser.add_argument("--target-horizons", default=None, help="Comma-separated target horizons (e.g. 1,5,15)")
    args = parser.parse_args()

    raw_dir = project_root / args.raw_dir
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find raw parquet file
    raw_file = raw_dir / f"{args.pair}_1m.parquet"
    if not raw_file.exists():
        print(f"Error: Raw file not found: {raw_file}")
        sys.exit(1)

    print(f"Loading raw data from {raw_file}...")
    df = pd.read_parquet(raw_file)
    print(f"  Raw data shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Resample if requested
    if args.resample:
        before = len(df)
        df = resample_ohlcv(df, args.resample)
        print(f"\n  Resampled to {args.resample}: {before:,} → {len(df):,} rows")

    # Parse target horizons
    target_horizons = None
    if args.target_horizons:
        target_horizons = [int(h) for h in args.target_horizons.split(",")]

    # Compute all features
    print("\nComputing all 43 features + target...")
    df_features = compute_all_features(df, drop_na=True, target_horizons=target_horizons)
    print(f"  Processed shape: {df_features.shape}")
    print(f"  Rows dropped (NaN warm-up + target): {len(df) - len(df_features)}")

    # Verify feature count
    available_features = [f for f in ALL_FEATURES if f in df_features.columns]
    missing_features = [f for f in ALL_FEATURES if f not in df_features.columns]

    print(f"\n  Features computed: {len(available_features)} / {len(ALL_FEATURES)}")
    if missing_features:
        print(f"  WARNING: Missing features: {missing_features}")

    # Check for NaN/Inf in final data
    feature_cols = available_features + [TARGET_COLUMN]
    nan_counts = df_features[feature_cols].isna().sum()
    inf_counts = np.isinf(df_features[feature_cols].select_dtypes(include=[np.number])).sum()

    if nan_counts.sum() > 0:
        print(f"\n  WARNING: NaN values found:")
        for col in nan_counts[nan_counts > 0].index:
            print(f"    {col}: {nan_counts[col]} NaN")

    if inf_counts.sum() > 0:
        print(f"\n  WARNING: Inf values found:")
        for col in inf_counts[inf_counts > 0].index:
            print(f"    {col}: {inf_counts[col]} Inf")

    # Print feature statistics
    print("\n--- Feature Statistics ---")
    stats = df_features[feature_cols].describe().T[["mean", "std", "min", "max"]]
    print(stats.to_string())

    # Save processed parquet
    suffix = f"_{args.resample}" if args.resample else ""
    out_file = out_dir / f"{args.pair}{suffix}_features.parquet"
    df_features.to_parquet(out_file, index=False)
    print(f"\nSaved processed features to {out_file}")
    print(f"  Final shape: {df_features.shape}")
    print(f"  File size: {out_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
