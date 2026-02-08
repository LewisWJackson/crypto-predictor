#!/usr/bin/env python3
"""
Alternative Data Fetcher for Crypto Prediction.

Fetches unconventional data sources that most crypto traders ignore.
The hypothesis: alpha comes from signals nobody else is looking at.

Each data source is a plugin — easy to add new ones.
All data is resampled to 1-minute intervals and merged with OHLCV data.

Usage:
    python scripts/fetch_alternative_data.py --all
    python scripts/fetch_alternative_data.py --sources moon,fear_greed,gold
    python scripts/fetch_alternative_data.py --list
"""

import argparse
import math
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ALT_DATA_DIR = DATA_DIR / "alternative"
ALT_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Base class for alternative data sources
# =====================================================================

class AltDataSource(ABC):
    """Base class for alternative data plugins."""

    name: str = ""
    description: str = ""
    frequency: str = ""  # "1min", "1h", "1d", etc.
    features: list[str] = []
    requires_api_key: bool = False
    api_key_env: str = ""

    @abstractmethod
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch data for the given date range.

        Returns DataFrame with 'timestamp' column (unix ms) and feature columns.
        """
        pass

    def get_cache_path(self) -> Path:
        return ALT_DATA_DIR / f"{self.name}.parquet"

    def load_cached(self) -> pd.DataFrame | None:
        path = self.get_cache_path()
        if path.exists():
            return pd.read_parquet(path)
        return None

    def save_cache(self, df: pd.DataFrame):
        df.to_parquet(self.get_cache_path(), index=False)


# =====================================================================
# Source 1: Moon Phases
# =====================================================================

class MoonPhaseSource(AltDataSource):
    """Lunar cycle data — surprisingly studied in finance literature.

    Papers: "Lunar Cycle Effects in Stock Returns" (Dichev & Janes, 2003)
    found statistically significant returns around new moons.

    No API needed — computed mathematically.
    """

    name = "moon_phases"
    description = "Lunar cycle phase, illumination, distance"
    frequency = "1h"
    features = ["moon_phase", "moon_illumination", "moon_age_days",
                "moon_is_new", "moon_is_full", "moon_is_waxing"]

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Compute moon phase for each hour in range."""
        rows = []
        current = start.replace(minute=0, second=0, microsecond=0)

        while current <= end:
            phase = self._moon_phase(current)
            illumination = self._illumination(phase)
            age = phase * 29.53059  # days into cycle

            rows.append({
                "timestamp": int(current.timestamp() * 1000),
                "moon_phase": phase,  # 0-1 (0=new, 0.5=full)
                "moon_illumination": illumination,  # 0-1
                "moon_age_days": age,
                "moon_is_new": 1.0 if phase < 0.05 or phase > 0.95 else 0.0,
                "moon_is_full": 1.0 if 0.45 < phase < 0.55 else 0.0,
                "moon_is_waxing": 1.0 if phase < 0.5 else 0.0,
            })
            current += timedelta(hours=1)

        print(f"  [{self.name}] Computed {len(rows)} hourly moon phases")
        return pd.DataFrame(rows)

    @staticmethod
    def _moon_phase(dt: datetime) -> float:
        """Calculate moon phase (0-1) using Conway's algorithm."""
        year = dt.year
        month = dt.month
        day = dt.day + dt.hour / 24.0

        if month <= 2:
            year -= 1
            month += 12

        a = math.floor(year / 100)
        b = 2 - a + math.floor(a / 4)
        jd = (math.floor(365.25 * (year + 4716)) +
              math.floor(30.6001 * (month + 1)) + day + b - 1524.5)

        # Days since known new moon (Jan 6, 2000 18:14 UTC)
        days_since = jd - 2451550.1
        # Synodic month = 29.53059 days
        phase = (days_since % 29.53059) / 29.53059
        return phase

    @staticmethod
    def _illumination(phase: float) -> float:
        """Convert phase to illumination percentage."""
        return (1 - math.cos(2 * math.pi * phase)) / 2


# =====================================================================
# Source 2: Fear & Greed Index
# =====================================================================

class FearGreedSource(AltDataSource):
    """Crypto Fear & Greed Index from alternative.me.

    Measures market sentiment from 0 (extreme fear) to 100 (extreme greed).
    Uses: volatility, momentum, social media, surveys, dominance, trends.
    Free API, no key needed.
    """

    name = "fear_greed"
    description = "Crypto Fear & Greed Index (0-100)"
    frequency = "1d"
    features = ["fear_greed_value", "fear_greed_normalized",
                "fg_extreme_fear", "fg_fear", "fg_neutral",
                "fg_greed", "fg_extreme_greed"]

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        import requests

        days = (end - start).days + 1
        url = f"https://api.alternative.me/fng/?limit={days}&format=json"

        print(f"  [{self.name}] Fetching {days} days of Fear & Greed data...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        rows = []
        for entry in data:
            ts = int(entry["timestamp"]) * 1000  # seconds to ms
            value = int(entry["value"])
            rows.append({
                "timestamp": ts,
                "fear_greed_value": value,
                "fear_greed_normalized": value / 100.0,
                "fg_extreme_fear": 1.0 if value <= 20 else 0.0,
                "fg_fear": 1.0 if 20 < value <= 40 else 0.0,
                "fg_neutral": 1.0 if 40 < value <= 60 else 0.0,
                "fg_greed": 1.0 if 60 < value <= 80 else 0.0,
                "fg_extreme_greed": 1.0 if value > 80 else 0.0,
            })

        print(f"  [{self.name}] Got {len(rows)} daily readings")
        return pd.DataFrame(rows)


# =====================================================================
# Source 3: Gold / DXY / Oil / S&P 500 (via Yahoo Finance)
# =====================================================================

class MacroAssetsSource(AltDataSource):
    """Traditional macro assets — gold, dollar index, oil, S&P 500.

    BTC often trades inversely to the dollar and correlates with gold.
    Uses yfinance (free, no API key).
    """

    name = "macro_assets"
    description = "Gold, DXY, Oil, S&P500, VIX, Corn, Copper"
    frequency = "1d"
    features = [
        "gold_close", "gold_return", "gold_sma20_dist",
        "dxy_close", "dxy_return", "dxy_sma20_dist",
        "oil_close", "oil_return",
        "sp500_close", "sp500_return", "sp500_sma20_dist",
        "vix_close", "vix_level",
        "corn_close", "corn_return",
        "copper_close", "copper_return",
    ]

    TICKERS = {
        "gold": "GC=F",
        "dxy": "DX-Y.NYB",
        "oil": "CL=F",
        "sp500": "^GSPC",
        "vix": "^VIX",
        "corn": "ZC=F",
        "copper": "HG=F",
    }

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            print(f"  [{self.name}] yfinance not installed, pip install yfinance")
            return pd.DataFrame()

        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        all_data = {}
        for asset_name, ticker in self.TICKERS.items():
            print(f"  [{self.name}] Fetching {asset_name} ({ticker})...")
            try:
                df = yf.download(ticker, start=start_str, end=end_str,
                                 progress=False, auto_adjust=True)
                if not df.empty:
                    all_data[asset_name] = df["Close"]
            except Exception as e:
                print(f"  [{self.name}] WARNING: Failed to fetch {asset_name}: {e}")

        if not all_data:
            return pd.DataFrame()

        combined = pd.DataFrame(all_data)
        combined.index = pd.to_datetime(combined.index)

        rows = []
        for dt, row in combined.iterrows():
            ts = int(dt.timestamp() * 1000)
            entry = {"timestamp": ts}

            for asset in self.TICKERS:
                if asset in row and not pd.isna(row[asset]):
                    entry[f"{asset}_close"] = row[asset]

            rows.append(entry)

        result = pd.DataFrame(rows)

        # Compute returns and indicators
        for asset in self.TICKERS:
            col = f"{asset}_close"
            if col in result.columns:
                result[f"{asset}_return"] = result[col].pct_change()
                sma20 = result[col].rolling(20).mean()
                result[f"{asset}_sma20_dist"] = (result[col] - sma20) / sma20

        # VIX levels
        if "vix_close" in result.columns:
            result["vix_level"] = pd.cut(
                result["vix_close"],
                bins=[0, 15, 20, 25, 35, 100],
                labels=[0, 1, 2, 3, 4],
            ).astype(float)

        print(f"  [{self.name}] Got {len(result)} daily readings for {len(all_data)} assets")
        return result


# =====================================================================
# Source 4: Google Trends
# =====================================================================

class GoogleTrendsSource(AltDataSource):
    """Google search interest for crypto-related terms.

    Retail interest spikes often precede/follow price moves.
    Uses pytrends (free, no API key, but rate-limited).
    """

    name = "google_trends"
    description = "Google search interest for crypto terms"
    frequency = "1d"
    features = ["gt_bitcoin", "gt_crypto_crash", "gt_buy_bitcoin",
                "gt_bitcoin_price", "gt_crypto"]

    KEYWORDS = ["bitcoin", "crypto crash", "buy bitcoin",
                "bitcoin price", "crypto"]

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        try:
            from pytrends.request import TrendReq
        except ImportError:
            print(f"  [{self.name}] pytrends not installed, pip install pytrends")
            return pd.DataFrame()

        pytrends = TrendReq(hl="en-US", tz=0)
        timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"

        all_data = {}
        for kw in self.KEYWORDS:
            print(f"  [{self.name}] Fetching trend: '{kw}'...")
            try:
                pytrends.build_payload([kw], timeframe=timeframe)
                df = pytrends.interest_over_time()
                if not df.empty and kw in df.columns:
                    col_name = f"gt_{kw.replace(' ', '_')}"
                    all_data[col_name] = df[kw]
                time.sleep(2)  # Rate limit
            except Exception as e:
                print(f"  [{self.name}] WARNING: Failed for '{kw}': {e}")

        if not all_data:
            return pd.DataFrame()

        combined = pd.DataFrame(all_data)
        combined.index = pd.to_datetime(combined.index)

        rows = []
        for dt, row in combined.iterrows():
            entry = {"timestamp": int(dt.timestamp() * 1000)}
            for col in combined.columns:
                entry[col] = row[col] / 100.0  # Normalize to 0-1
            rows.append(entry)

        print(f"  [{self.name}] Got {len(rows)} daily readings")
        return pd.DataFrame(rows)


# =====================================================================
# Source 5: Bitcoin On-Chain Data (via Blockchain.com)
# =====================================================================

class OnChainSource(AltDataSource):
    """Bitcoin on-chain metrics from blockchain.com.

    Free API, no key needed. These capture network-level signals
    that most price-only models miss entirely.
    """

    name = "onchain"
    description = "BTC hash rate, active addresses, transaction volume"
    frequency = "1d"
    features = ["hash_rate", "hash_rate_change", "active_addresses",
                "active_addr_change", "tx_volume_usd", "tx_volume_change",
                "mempool_size", "avg_block_size"]

    CHARTS = {
        "hash_rate": "hash-rate",
        "active_addresses": "n-unique-addresses",
        "tx_volume_usd": "estimated-transaction-volume-usd",
        "mempool_size": "mempool-size",
        "avg_block_size": "avg-block-size",
    }

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        import requests

        start_ts = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)

        all_series = {}
        for feature, chart_name in self.CHARTS.items():
            url = f"https://api.blockchain.info/charts/{chart_name}?timespan=all&format=json&start={start_ts//1000}&end={end_ts//1000}"
            print(f"  [{self.name}] Fetching {feature}...")
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                values = resp.json().get("values", [])
                series = {int(v["x"]) * 1000: v["y"] for v in values}
                all_series[feature] = series
                time.sleep(0.5)
            except Exception as e:
                print(f"  [{self.name}] WARNING: Failed {feature}: {e}")

        if not all_series:
            return pd.DataFrame()

        # Merge all series by timestamp
        all_timestamps = set()
        for series in all_series.values():
            all_timestamps.update(series.keys())

        rows = []
        for ts in sorted(all_timestamps):
            entry = {"timestamp": ts}
            for feature, series in all_series.items():
                entry[feature] = series.get(ts, np.nan)
            rows.append(entry)

        result = pd.DataFrame(rows)

        # Compute daily changes
        for feature in self.CHARTS:
            if feature in result.columns:
                result[f"{feature}_change"] = result[feature].pct_change()

        print(f"  [{self.name}] Got {len(result)} daily readings")
        return result


# =====================================================================
# Source 6: Crypto Exchange Flows (via CryptoQuant-style free data)
# =====================================================================

class ExchangeFlowSource(AltDataSource):
    """Exchange inflow/outflow signals.

    Large exchange inflows often precede sell-offs.
    Uses Binance API for their own flow metrics.
    """

    name = "exchange_flows"
    description = "Binance funding rates, open interest, long/short ratio"
    frequency = "1h"
    features = ["funding_rate", "funding_rate_abs", "funding_positive",
                "long_short_ratio", "ls_ratio_extreme"]

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        import requests

        rows = []
        current = start
        batch_size = 500

        print(f"  [{self.name}] Fetching Binance funding rates...")
        # Funding rate (every 8h but we fetch all available)
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {
            "symbol": "BTCUSDT",
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": 1000,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for entry in data:
                rate = float(entry["fundingRate"])
                rows.append({
                    "timestamp": int(entry["fundingTime"]),
                    "funding_rate": rate,
                    "funding_rate_abs": abs(rate),
                    "funding_positive": 1.0 if rate > 0 else 0.0,
                })
        except Exception as e:
            print(f"  [{self.name}] WARNING: Funding rate fetch failed: {e}")

        # Long/Short ratio
        print(f"  [{self.name}] Fetching long/short ratio...")
        url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        params = {
            "symbol": "BTCUSDT",
            "period": "1h",
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": 500,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            ls_rows = []
            for entry in data:
                ratio = float(entry["longShortRatio"])
                ls_rows.append({
                    "timestamp": int(entry["timestamp"]),
                    "long_short_ratio": ratio,
                    "ls_ratio_extreme": 1.0 if ratio > 2.0 or ratio < 0.5 else 0.0,
                })

            if ls_rows:
                ls_df = pd.DataFrame(ls_rows)
                if rows:
                    main_df = pd.DataFrame(rows)
                    result = pd.merge_asof(
                        main_df.sort_values("timestamp"),
                        ls_df.sort_values("timestamp"),
                        on="timestamp",
                        direction="nearest",
                    )
                    print(f"  [{self.name}] Got {len(result)} readings")
                    return result
                else:
                    print(f"  [{self.name}] Got {len(ls_rows)} readings")
                    return pd.DataFrame(ls_rows)
        except Exception as e:
            print(f"  [{self.name}] WARNING: Long/short ratio fetch failed: {e}")

        if rows:
            print(f"  [{self.name}] Got {len(rows)} readings")
            return pd.DataFrame(rows)
        return pd.DataFrame()


# =====================================================================
# Source 7: Day/Time Patterns (computed, no API)
# =====================================================================

class TemporalPatternsSource(AltDataSource):
    """Advanced temporal features beyond simple hour/day encoding.

    Trading session overlaps, day-of-week effects, month-end rebalancing,
    options expiry dates, etc.
    """

    name = "temporal_patterns"
    description = "Trading sessions, options expiry, month-end effects"
    frequency = "1h"
    features = [
        "is_asian_session", "is_european_session", "is_us_session",
        "session_overlap_eu_us", "session_overlap_asia_eu",
        "is_weekend", "is_monday", "is_friday",
        "is_month_end_3d", "is_month_start_3d",
        "is_quarter_end", "days_to_month_end",
        "is_btc_options_expiry",  # Last Friday of month
    ]

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        rows = []
        current = start.replace(minute=0, second=0, microsecond=0)

        while current <= end:
            hour = current.hour
            weekday = current.weekday()
            day = current.day

            # Trading sessions (UTC)
            is_asian = 1.0 if 0 <= hour < 8 else 0.0
            is_european = 1.0 if 7 <= hour < 16 else 0.0
            is_us = 1.0 if 13 <= hour < 22 else 0.0

            # Month-end effects
            next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
            days_to_end = (next_month - current).days

            # BTC options expiry (last Friday of month)
            last_day = next_month - timedelta(days=1)
            offset = (last_day.weekday() - 4) % 7
            last_friday = last_day - timedelta(days=offset)
            is_expiry = 1.0 if current.date() == last_friday.date() else 0.0

            rows.append({
                "timestamp": int(current.timestamp() * 1000),
                "is_asian_session": is_asian,
                "is_european_session": is_european,
                "is_us_session": is_us,
                "session_overlap_eu_us": 1.0 if 13 <= hour < 16 else 0.0,
                "session_overlap_asia_eu": 1.0 if 7 <= hour < 8 else 0.0,
                "is_weekend": 1.0 if weekday >= 5 else 0.0,
                "is_monday": 1.0 if weekday == 0 else 0.0,
                "is_friday": 1.0 if weekday == 4 else 0.0,
                "is_month_end_3d": 1.0 if days_to_end <= 3 else 0.0,
                "is_month_start_3d": 1.0 if day <= 3 else 0.0,
                "is_quarter_end": 1.0 if current.month in (3, 6, 9, 12) and days_to_end <= 3 else 0.0,
                "days_to_month_end": float(days_to_end),
                "is_btc_options_expiry": is_expiry,
            })
            current += timedelta(hours=1)

        print(f"  [{self.name}] Computed {len(rows)} hourly temporal patterns")
        return pd.DataFrame(rows)


# =====================================================================
# Registry & CLI
# =====================================================================

SOURCES = {
    "moon": MoonPhaseSource(),
    "fear_greed": FearGreedSource(),
    "macro": MacroAssetsSource(),
    "google_trends": GoogleTrendsSource(),
    "onchain": OnChainSource(),
    "exchange_flows": ExchangeFlowSource(),
    "temporal": TemporalPatternsSource(),
}


def merge_alt_data_with_ohlcv(ohlcv_path: str, alt_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge alternative data sources with main OHLCV features.

    Uses forward-fill for lower-frequency data (daily → minute).
    """
    main_df = pd.read_parquet(ohlcv_path)

    for source_name, alt_df in alt_dfs.items():
        if alt_df.empty:
            continue

        # Merge on timestamp (nearest match, forward fill)
        feature_cols = [c for c in alt_df.columns if c != "timestamp"]
        alt_df = alt_df.sort_values("timestamp")
        main_df = main_df.sort_values("timestamp")

        main_df = pd.merge_asof(
            main_df, alt_df,
            on="timestamp",
            direction="backward",  # Use most recent available value
        )
        print(f"  Merged {source_name}: +{len(feature_cols)} features")

    # Forward-fill any NaNs from merge
    for col in main_df.columns:
        if main_df[col].isna().any():
            main_df[col] = main_df[col].ffill().bfill()

    return main_df


def main():
    parser = argparse.ArgumentParser(description="Fetch alternative data for crypto prediction")
    parser.add_argument("--sources", type=str, default=None,
                        help="Comma-separated source names (default: all)")
    parser.add_argument("--all", action="store_true", help="Fetch all sources")
    parser.add_argument("--list", action="store_true", help="List available sources")
    parser.add_argument("--start", type=str, default="2017-09-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD, default: now)")
    parser.add_argument("--merge", type=str, default=None,
                        help="Path to OHLCV parquet to merge with")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for merged data")
    args = parser.parse_args()

    if args.list:
        print("Available alternative data sources:")
        print("-" * 60)
        for key, source in SOURCES.items():
            api_note = " (API key required)" if source.requires_api_key else " (free)"
            print(f"  {key:20s} {source.description}{api_note}")
            print(f"  {'':20s} Features: {', '.join(source.features[:4])}...")
            print(f"  {'':20s} Frequency: {source.frequency}")
            print()
        return

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
           if args.end else datetime.now(timezone.utc))

    # Select sources
    if args.all or args.sources is None:
        selected = SOURCES
    else:
        names = [s.strip() for s in args.sources.split(",")]
        selected = {k: v for k, v in SOURCES.items() if k in names}
        missing = set(names) - set(selected.keys())
        if missing:
            print(f"WARNING: Unknown sources: {missing}")

    print(f"Fetching alternative data: {', '.join(selected.keys())}")
    print(f"Date range: {start.date()} to {end.date()}")
    print()

    # Fetch each source
    alt_dfs = {}
    for name, source in selected.items():
        print(f"[{name}] {source.description}")
        try:
            df = source.fetch(start, end)
            if not df.empty:
                source.save_cache(df)
                alt_dfs[name] = df
                print(f"  Saved to {source.get_cache_path()}")
            else:
                print(f"  No data returned")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    # Merge with OHLCV if requested
    if args.merge:
        print("Merging alternative data with OHLCV features...")
        merged = merge_alt_data_with_ohlcv(args.merge, alt_dfs)
        output_path = args.output or str(DATA_DIR / "processed" / "btc_usdt_features_enhanced.parquet")
        merged.to_parquet(output_path, index=False)
        print(f"Enhanced dataset saved: {output_path}")
        print(f"Total features: {len(merged.columns)}")
        print(f"Total rows: {len(merged):,}")

    # Summary
    total_features = sum(len(df.columns) - 1 for df in alt_dfs.values())
    print(f"\nSummary: {len(alt_dfs)} sources fetched, {total_features} alternative features available")


if __name__ == "__main__":
    main()
