# Bitcoin Historical Data Sources & Collection Methods

## 1. Free API Options for 1-Minute OHLCV Data

### Binance API (RECOMMENDED - Primary)
- **Endpoint**: `/api/v3/klines` for spot, also futures via `/fapi/v1/klines`
- **Limit per request**: 1000 candles (default 500)
- **Rate limit**: 6000 request weight/min (klines endpoint = weight 2, so ~3000 requests/min theoretical)
- **History depth**: Data available back to 2017 for BTC/USDT — easily 3+ million 1-minute candles
- **No API key required** for market data endpoints
- **Bulk download**: `data.binance.vision` provides pre-packaged monthly/daily ZIP files of klines data for all timeframes including 1m. Each ZIP has SHA256 checksum files.
  - URL pattern: `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip`
  - Updated: Daily data ~10:00 UTC next day; monthly data on 1st of following month
- **Verdict**: Best free source. Huge history, generous rate limits, bulk download option.

### CryptoDataDownload
- **URL**: https://www.cryptodatadownload.com/data/
- **Format**: Direct CSV downloads, also API (JSON/CSV/XLSX)
- **Coverage**: Zero-gap OHLCV data for BTCUSDT and 7 other major pairs, 2020-2025
- **Resolution**: 1-minute intervals available
- **Cost**: Free (they state "We will never ask you to pay for raw, historical data!")
- **Verdict**: Great backup/validation source. Pre-cleaned, verified gap-free data.

### CoinGecko API
- **Free tier**: 30 calls/min, but **1-minute data NOT available** — minimum is 5-minute for Pro plan
- **History**: Free tier limited to 365 days of daily data
- **Verdict**: Not suitable for 1-min data. Good for daily/weekly supplementary data only.

### CryptoCompare
- **Free tier**: 100K calls/month
- **1-minute data**: Available via `/data/v2/histominute` but limited to 7 days of history on free tier
- **Verdict**: Too restrictive for bulk historical. Only useful for recent data fills.

### Polygon.io
- **Free tier**: 5 API calls/min, limited to end-of-day data for crypto
- **Paid**: Starter plan ($29/mo) for minute-level data
- **Verdict**: Not viable for free 1-min crypto data.

### Yahoo Finance (yfinance)
- **Crypto support**: Limited. BTC-USD available but 1-minute data limited to last 7 days
- **Verdict**: Not suitable for historical 1-min data collection.

### Kraken
- **Endpoint**: `/0/public/OHLC`
- **Limit**: 720 candles per request
- **Also offers**: Downloadable OHLCVT CSV files for all pairs at 1-min intervals
- **Verdict**: Good alternative exchange source for cross-validation.

### Kaggle Datasets (Pre-downloaded)
Several free datasets exist with millions of 1-minute candles:
- **"Bitcoin BTC, 7 Exchanges, 1m Full Historical Data"** — multi-exchange 1-min data
- **"BITCOIN Historical Datasets 2018-2025 Binance API"** — Binance-sourced
- **"BTC and ETH 1-min Price History"** — BTC + ETH combo
- **Verdict**: Instant access to large datasets. Good for quick prototyping, but may be stale.

---

## 2. Python Libraries Comparison

### ccxt (RECOMMENDED)
- **Install**: `pip install ccxt`
- **Exchanges**: 100+ exchanges unified under one API
- **OHLCV**: `exchange.fetch_ohlcv('BTC/USDT', '1m', since=timestamp, limit=1000)`
- **Pagination**: Manual loop using `since` parameter, or built-in `params={"paginate": True}`
- **Rate limiting**: Built-in rate limiter (`exchange.rateLimit`)
- **Pros**: Unified API across exchanges, well-maintained, great docs, async support
- **Cons**: Must handle pagination manually for bulk downloads
- **Best for**: Flexible data collection from any exchange

### python-binance
- **Install**: `pip install python-binance`
- **Key method**: `client.get_historical_klines("BTCUSDT", "1m", "1 Jan 2020")`
- **Pros**: Handles pagination automatically, date string parsing, Binance-specific features
- **Cons**: Binance-only, requires API key for some features (not for klines though)
- **Best for**: Binance-specific bulk downloads

### binance-historical-data (PyPI)
- **Install**: `pip install binance-historical-data`
- **Purpose**: Downloads from `data.binance.vision` bulk files
- **Pros**: Fastest way to get bulk Binance data, handles ZIP extraction
- **Cons**: Only Binance data

### yfinance
- **Install**: `pip install yfinance`
- **Crypto**: `yf.download("BTC-USD", interval="1m")`
- **Limitation**: 1-minute data only for last 7 days
- **Best for**: Quick daily/hourly data, not for our use case

### pandas-datareader
- **Crypto support**: Minimal, mostly traditional finance
- **Verdict**: Not recommended for crypto

### Recommendation: **ccxt** as primary library
- Unified API means we can easily switch exchanges or add more pairs
- Good async support for fast parallel downloads
- Well-tested pagination patterns available
- If we want fastest Binance-specific download: use `data.binance.vision` bulk files via requests/wget

---

## 3. Data Volume Assessment

### Can we get 100K+ candles?

**Absolutely yes.** Here's what's available:

| Source | BTC/USDT 1-min candles available | Time span |
|--------|----------------------------------|-----------|
| Binance API | ~3,700,000+ | Aug 2017 - present |
| data.binance.vision | ~3,700,000+ | Aug 2017 - present |
| CryptoDataDownload | ~2,600,000+ | 2020 - present |
| Kaggle datasets | 1,000,000 - 4,000,000 | Varies |

Our target of 50,000-100,000 candles is easily achievable — that's only ~35-70 days of 1-minute data.

For a more robust model, we should aim for **500K-1M+ candles** (1-2 years) since it's freely available.

### Download time estimates (Binance API via ccxt):
- 100K candles: ~100 requests = ~2 minutes
- 500K candles: ~500 requests = ~10 minutes
- 1M candles: ~1000 requests = ~20 minutes
- Full history (~3.7M): Use data.binance.vision bulk download = ~5-10 minutes

---

## 4. Order Book / Level 2 Data

The video emphasized **order book gaps** as the underlying cause of price patterns. Historical order book data is much harder to obtain for free.

### Free Sources:
- **Bybit**: Provides free historical L2 order book datasets (daily updates, no registration needed)
  - Spot and Contract data available
  - Updated daily
- **Kaggle**: "High Frequency Crypto Limit Order Book Data" dataset
  - Limited time periods

### Paid Sources (for reference):
- **Tardis.dev**: Tick-level order book snapshots and L2 incremental updates
- **CoinAPI**: Daily archives with L2/L3 order-book data in CSV.gz format (microsecond precision)
- **Crypto Lake**: 20-level depth snapshots
- **Kaiko**: Professional L1/L2 feeds

### Realistic Assessment:
For our project, **we should start with OHLCV data** (readily available) and treat order book features as a future enhancement. If needed:
1. Collect real-time order book snapshots going forward (free via exchange WebSocket APIs)
2. Use Bybit's free historical L2 data as supplementary input
3. Approximate order book dynamics from volume and price action patterns

---

## 5. Data Quality Considerations

### Common Issues:

1. **Missing candles**: Periods with zero trading activity produce no candle. Low-liquidity periods (weekends for some pairs, though BTC trades 24/7) may have gaps.
   - **Fix**: Forward-fill or interpolate missing timestamps. CryptoDataDownload provides verified "zero-gap" data.

2. **Exchange differences**: Different exchanges build candles differently:
   - Some aggregate by trade timestamp, others by block time
   - Outlier handling varies
   - The same minute may show different OHLCV across Binance, Coinbase, Kraken
   - **Fix**: Stick to one exchange (Binance = most liquid = most reliable).

3. **Timestamp handling**:
   - All Binance timestamps are in UTC milliseconds
   - Ensure consistent timezone handling (always use UTC)
   - **Fix**: Store as UTC epoch ms, convert only for display.

4. **Volume discrepancies**: Base volume vs quote volume, wash trading concerns
   - **Fix**: Use Binance (regulated, relatively clean volume data).

5. **Symbol naming**: BTC/USDT vs BTCUSDT vs BTC-USDT varies by exchange
   - **Fix**: ccxt normalizes to `BTC/USDT` format.

6. **API changes**: Exchange APIs evolve, endpoints get deprecated
   - **Fix**: Use ccxt which abstracts away API versions.

### Data Validation Checklist:
- [ ] Verify no duplicate timestamps
- [ ] Check for gaps > 1 minute
- [ ] Ensure OHLC relationships: Low <= Open,Close <= High
- [ ] Verify volume is non-negative
- [ ] Check for obvious outliers (>10% price spikes in 1 min)
- [ ] Confirm data is sorted chronologically

---

## 6. Recommended Approach

### Primary: ccxt + Binance for API collection
```python
import ccxt
import pandas as pd
import time

exchange = ccxt.binance({'enableRateLimit': True})
symbol = 'BTC/USDT'
timeframe = '1m'
since = exchange.parse8601('2024-01-01T00:00:00Z')
limit = 1000

all_candles = []
while True:
    candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not candles:
        break
    all_candles.extend(candles)
    since = candles[-1][0] + 60000  # Next minute
    if len(candles) < limit:
        break
    time.sleep(exchange.rateLimit / 1000)

df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
df.to_parquet('btc_usdt_1m.parquet', index=False)
```

### Alternative: data.binance.vision for bulk download
For maximum speed with full history, download pre-packaged ZIP files:
```python
import requests, zipfile, io, pandas as pd

base_url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m"
# Download month by month: BTCUSDT-1m-2024-01.zip, etc.
```

### Recommended Pipeline:
1. **Initial load**: Use `data.binance.vision` bulk ZIPs for historical backfill (fastest)
2. **Incremental updates**: Use ccxt API to fetch recent data not yet in bulk files
3. **Validation**: Cross-check sample against CryptoDataDownload's zero-gap data
4. **Storage**: Save as Parquet (see Section 7)

---

## 7. Storage Format Recommendation

### Comparison for our use case (~1M-4M rows, 6 columns):

| Format | File Size (1M rows) | Read Speed | Write Speed | Human Readable | Compression |
|--------|---------------------|------------|-------------|----------------|-------------|
| CSV | ~50 MB | Slow (2-5s) | Moderate | Yes | None |
| Parquet | ~8-12 MB | Fast (0.2-0.5s) | Fast | No | Built-in |
| SQLite | ~40 MB | Moderate (1-2s) | Slow | Via tools | None |

### Recommendation: **Parquet** (primary) + **CSV** (backup)

**Why Parquet:**
- 5-10x smaller than CSV thanks to columnar compression
- 5-50x faster reads with pandas (`pd.read_parquet()`)
- Native datetime/type preservation (no parsing on load)
- Column-level access: can read just `close` column without loading everything
- Standard in ML/data science workflows
- PyArrow is already a pandas dependency

**Why keep a CSV backup:**
- Human-readable for debugging
- Easy to share, inspect in any text editor or spreadsheet
- Universal compatibility

**Storage plan:**
```
data/
  raw/
    btc_usdt_1m.parquet          # Primary storage (~12 MB for 1M candles)
    btc_usdt_1m.csv              # Human-readable backup
  processed/
    btc_usdt_1m_features.parquet # After feature engineering
```

**Code:**
```python
# Write
df.to_parquet('data/raw/btc_usdt_1m.parquet', index=False, engine='pyarrow')
df.to_csv('data/raw/btc_usdt_1m.csv', index=False)  # Backup

# Read (fast)
df = pd.read_parquet('data/raw/btc_usdt_1m.parquet')
```

---

## Summary Decision Matrix

| Aspect | Choice | Reason |
|--------|--------|--------|
| **Exchange** | Binance | Most liquid, longest history, best free API |
| **Bulk download** | data.binance.vision | Pre-packaged ZIPs, fastest for historical |
| **API library** | ccxt | Unified, well-maintained, exchange-agnostic |
| **Incremental updates** | ccxt + Binance API | Real-time updates via REST/WebSocket |
| **Storage** | Parquet + CSV backup | Fast, compact, type-safe |
| **Order book** | OHLCV first, L2 later | Free L2 data limited; start with price data |
| **Target volume** | 500K-1M candles (1-2 years) | Freely available, good for training |
| **Pair** | BTC/USDT | Most liquid, most data, our target |
