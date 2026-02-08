#!/usr/bin/env python3
"""
Bitcoin Historical Data Downloader

Downloads 1-minute BTC/USDT OHLCV data from Binance using two methods:
1. data.binance.vision bulk ZIP files (fastest for historical backfill)
2. ccxt API (for incremental updates and flexibility)

Usage:
    python download_data.py                    # Download last 6 months via API
    python download_data.py --method bulk      # Download via bulk ZIP files
    python download_data.py --method api       # Download via ccxt API
    python download_data.py --start 2024-01-01 # Custom start date
    python download_data.py --start 2023-01-01 --end 2024-12-31  # Custom range
    python download_data.py --candles 100000   # Download exactly N candles
"""

import argparse
import io
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOL = "BTC/USDT"
BINANCE_SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
CANDLE_MS = 60_000  # 1 minute in milliseconds

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PARQUET_PATH = DATA_DIR / "btc_usdt_1m.parquet"
CSV_PATH = DATA_DIR / "btc_usdt_1m.csv"

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Binance Vision bulk download base URL
BINANCE_VISION_BASE = "https://data.binance.vision/data/spot/monthly/klines"

# Binance Vision CSV columns (from their documentation)
BINANCE_VISION_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


# ---------------------------------------------------------------------------
# Method 1: Bulk download from data.binance.vision
# ---------------------------------------------------------------------------

def download_bulk(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Download historical data from data.binance.vision ZIP files.

    This is the fastest method for large historical downloads.
    Files are organized by month: BTCUSDT-1m-YYYY-MM.zip
    """
    print(f"[Bulk] Downloading {BINANCE_SYMBOL} 1m data from data.binance.vision")
    print(f"[Bulk] Range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")

    all_dfs = []
    current = start_date.replace(day=1)

    while current <= end_date:
        year_month = current.strftime("%Y-%m")
        filename = f"{BINANCE_SYMBOL}-1m-{year_month}.zip"
        url = f"{BINANCE_VISION_BASE}/{BINANCE_SYMBOL}/1m/{filename}"

        print(f"  Downloading {filename}...", end=" ", flush=True)

        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 404:
                print("NOT FOUND (skipping)")
                current = _next_month(current)
                continue
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f, header=None, names=BINANCE_VISION_COLUMNS)

            # Keep only the columns we need and rename
            df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
            df.rename(columns={"open_time": "timestamp"}, inplace=True)

            all_dfs.append(df)
            print(f"OK ({len(df):,} candles)")

        except requests.RequestException as e:
            print(f"ERROR: {e}")

        current = _next_month(current)

    if not all_dfs:
        print("[Bulk] No data downloaded!")
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    result = pd.concat(all_dfs, ignore_index=True)

    # Filter to exact date range
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    result = result[(result["timestamp"] >= start_ts) & (result["timestamp"] <= end_ts)]

    result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"[Bulk] Total: {len(result):,} candles downloaded")
    return result


def _next_month(dt: datetime) -> datetime:
    """Return the first day of the next month."""
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1, day=1)
    return dt.replace(month=dt.month + 1, day=1)


# ---------------------------------------------------------------------------
# Method 2: ccxt API download
# ---------------------------------------------------------------------------

def download_api(start_date: datetime, end_date: datetime, max_candles: int | None = None) -> pd.DataFrame:
    """Download historical data using the ccxt library via Binance REST API.

    Fetches 1000 candles per request with automatic pagination.
    No API key required for market data.
    """
    try:
        import ccxt
    except ImportError:
        print("Error: ccxt not installed. Run: pip install ccxt")
        sys.exit(1)

    exchange = ccxt.binance({"enableRateLimit": True})

    since = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    limit = 1000

    print(f"[API] Downloading {SYMBOL} 1m data via ccxt/Binance API")
    print(f"[API] Range: {start_date.isoformat()} to {end_date.isoformat()}")
    if max_candles:
        print(f"[API] Target: {max_candles:,} candles")

    all_candles = []
    request_count = 0

    while since < end_ts:
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=limit)
        except ccxt.NetworkError as e:
            print(f"\n  Network error: {e}. Retrying in 5s...")
            time.sleep(5)
            continue
        except ccxt.ExchangeError as e:
            print(f"\n  Exchange error: {e}. Retrying in 10s...")
            time.sleep(10)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        request_count += 1
        since = candles[-1][0] + CANDLE_MS

        # Progress reporting
        collected = len(all_candles)
        latest_dt = datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc)
        if request_count % 10 == 0:
            print(f"  {collected:,} candles | up to {latest_dt.strftime('%Y-%m-%d %H:%M')} UTC | {request_count} requests")

        # Stop conditions
        if max_candles and collected >= max_candles:
            all_candles = all_candles[:max_candles]
            break
        if len(candles) < limit:
            break

        # Respect rate limits (ccxt handles this, but add small buffer)
        time.sleep(0.1)

    if not all_candles:
        print("[API] No data downloaded!")
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.DataFrame(all_candles, columns=OHLCV_COLUMNS)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"[API] Total: {len(df):,} candles downloaded in {request_count} requests")
    return df


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def validate_data(df: pd.DataFrame) -> dict:
    """Run quality checks on the downloaded OHLCV data."""
    issues = {}

    if df.empty:
        return {"empty": True}

    # Check duplicates
    dupes = df["timestamp"].duplicated().sum()
    if dupes > 0:
        issues["duplicate_timestamps"] = int(dupes)

    # Check gaps (missing candles)
    timestamps = df["timestamp"].sort_values()
    diffs = timestamps.diff().dropna()
    expected_diff = CANDLE_MS
    gaps = diffs[diffs > expected_diff * 1.5]  # Allow small tolerance
    if len(gaps) > 0:
        issues["missing_candle_gaps"] = len(gaps)
        max_gap_minutes = int(gaps.max() / CANDLE_MS)
        issues["largest_gap_minutes"] = max_gap_minutes

    # Check OHLC validity: low <= open/close <= high
    invalid_ohlc = (
        (df["low"] > df["open"]) | (df["low"] > df["close"]) |
        (df["high"] < df["open"]) | (df["high"] < df["close"])
    ).sum()
    if invalid_ohlc > 0:
        issues["invalid_ohlc_relationships"] = int(invalid_ohlc)

    # Check for negative volume
    neg_vol = (df["volume"] < 0).sum()
    if neg_vol > 0:
        issues["negative_volume"] = int(neg_vol)

    # Check for extreme price spikes (>10% in 1 candle)
    pct_change = df["close"].pct_change().abs()
    spikes = (pct_change > 0.10).sum()
    if spikes > 0:
        issues["extreme_spikes_gt_10pct"] = int(spikes)

    # Check sort order
    if not timestamps.is_monotonic_increasing:
        issues["not_sorted"] = True

    return issues


def print_summary(df: pd.DataFrame):
    """Print a summary of the downloaded data."""
    if df.empty:
        print("\nNo data to summarize.")
        return

    ts_col = df["timestamp"]
    start_dt = datetime.fromtimestamp(ts_col.min() / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(ts_col.max() / 1000, tz=timezone.utc)

    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"  Symbol:       {SYMBOL}")
    print(f"  Timeframe:    {TIMEFRAME}")
    print(f"  Candles:      {len(df):,}")
    print(f"  Date range:   {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Duration:     {(end_dt - start_dt).days} days")
    print(f"  Price range:  ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
    print(f"  Avg volume:   {df['volume'].mean():,.2f} BTC/candle")

    # Validation
    issues = validate_data(df)
    if issues:
        print(f"\n  QUALITY ISSUES:")
        for k, v in issues.items():
            print(f"    - {k}: {v}")
    else:
        print(f"\n  QUALITY: All checks passed!")


# ---------------------------------------------------------------------------
# Save data
# ---------------------------------------------------------------------------

def save_data(df: pd.DataFrame, save_csv: bool = True):
    """Save DataFrame to Parquet (primary) and optionally CSV (backup)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save Parquet
    df.to_parquet(PARQUET_PATH, index=False, engine="pyarrow")
    parquet_size = PARQUET_PATH.stat().st_size / (1024 * 1024)
    print(f"\n  Saved: {PARQUET_PATH} ({parquet_size:.1f} MB)")

    # Save CSV backup
    if save_csv:
        df.to_csv(CSV_PATH, index=False)
        csv_size = CSV_PATH.stat().st_size / (1024 * 1024)
        print(f"  Saved: {CSV_PATH} ({csv_size:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Bitcoin 1-minute OHLCV data from Binance"
    )
    parser.add_argument(
        "--method", choices=["bulk", "api"], default="api",
        help="Download method: 'bulk' for data.binance.vision ZIPs (fastest), "
             "'api' for ccxt REST API (default: api)"
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date in YYYY-MM-DD format (default: 6 months ago)"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date in YYYY-MM-DD format (default: now)"
    )
    parser.add_argument(
        "--candles", type=int, default=None,
        help="Target number of candles to download (API method only)"
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Skip CSV backup (only save Parquet)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory (default: ../data/raw/)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set output directory
    global DATA_DIR, PARQUET_PATH, CSV_PATH
    if args.output_dir:
        DATA_DIR = Path(args.output_dir)
        PARQUET_PATH = DATA_DIR / "btc_usdt_1m.parquet"
        CSV_PATH = DATA_DIR / "btc_usdt_1m.csv"

    # Parse dates
    end_date = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(timezone.utc)
    )

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif args.candles:
        # Calculate start date from candle count
        start_date = end_date - timedelta(minutes=args.candles + 1000)  # Buffer
    else:
        # Default: 6 months
        start_date = end_date - timedelta(days=180)

    print(f"Bitcoin 1-Minute Data Downloader")
    print(f"================================")
    print(f"Method: {args.method}")
    print(f"Range:  {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print()

    # Download
    if args.method == "bulk":
        df = download_bulk(start_date, end_date)
    else:
        df = download_api(start_date, end_date, max_candles=args.candles)

    if df.empty:
        print("No data downloaded. Exiting.")
        sys.exit(1)

    # Summary and validation
    print_summary(df)

    # Save
    save_data(df, save_csv=not args.no_csv)

    print(f"\nDone! {len(df):,} candles ready for training.")


if __name__ == "__main__":
    main()
