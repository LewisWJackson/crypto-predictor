#!/usr/bin/env python3
"""
Inference Script for TFT Crypto Predictor.

Loads a trained model, takes recent price data (from parquet file
or live fetch), computes features, runs prediction, and outputs
predicted direction, magnitude, and confidence.

Usage:
    python scripts/predict.py --checkpoint models/tft/best.ckpt
    python scripts/predict.py --checkpoint models/tft/best.ckpt --data data/raw/BTC_USDT_1m.parquet
    python scripts/predict.py --checkpoint models/tft/best.ckpt --live
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

from src.model import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TFT model inference for crypto price prediction"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "tft_config.yaml"),
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to parquet file with recent OHLCV data",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch live data from Binance via CCXT",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output prediction as JSON",
    )
    return parser.parse_args()


def fetch_live_data(pair: str = "BTC/USDT", timeframe: str = "15m", limit: int = 500):
    """Fetch recent OHLCV data from Binance via CCXT.

    Args:
        pair: Trading pair (default BTC/USDT)
        timeframe: Candle timeframe (default 1m)
        limit: Number of candles to fetch

    Returns:
        DataFrame with timestamp, open, high, low, close, volume columns
    """
    try:
        import ccxt
    except ImportError:
        print("Error: ccxt is required for live data. Install with: pip install ccxt")
        sys.exit(1)

    exchange = ccxt.binance({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(pair, timeframe, limit=limit)

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    return df


def compute_prediction_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute features for the prediction input.

    Applies the same feature engineering pipeline as during training.
    Requires at least encoder_length + warm-up rows of data.

    Args:
        df: Raw OHLCV DataFrame
        config: Full config dict

    Returns:
        DataFrame with all features computed
    """
    # Import feature computation from the features module
    from src.features import compute_all_features

    target_horizons = None
    dataset_cfg = config.get("dataset", {})
    target_col = dataset_cfg.get("target_col", "forward_return_15")
    # Extract horizon number from target_col (e.g. "forward_return_1" -> [1])
    if target_col:
        import re
        m = re.search(r"(\d+)$", target_col)
        if m:
            target_horizons = [int(m.group(1))]
    df = compute_all_features(df, drop_na=True, target_horizons=target_horizons)

    # Drop NaN rows from indicator warm-up
    df = df.dropna().reset_index(drop=True)

    return df


def prepare_prediction_dataframe(
    feature_df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Prepare a DataFrame formatted for pytorch-forecasting prediction.

    Args:
        feature_df: Feature-engineered DataFrame
        config: Full config dict

    Returns:
        DataFrame ready for TimeSeriesDataSet
    """
    dataset_cfg = config.get("dataset", {})
    encoder_length = dataset_cfg.get("encoder_length", 256)
    decoder_length = dataset_cfg.get("decoder_length", 15)

    # Take the last encoder_length + decoder_length rows
    total_needed = encoder_length + decoder_length
    if len(feature_df) < total_needed:
        print(
            f"Warning: Only {len(feature_df)} rows available, "
            f"need {total_needed}. Using all available data."
        )
        df = feature_df.copy()
    else:
        df = feature_df.iloc[-total_needed:].copy()

    df = df.reset_index(drop=True)
    df["time_idx"] = range(len(df))
    df["group"] = "BTC_USDT"

    # For prediction, we need a target column (can be zeros for future steps)
    target_col = config.get("dataset", {}).get("target_col", "forward_return_15")
    if target_col not in df.columns:
        df[target_col] = 0.0

    return df


def run_prediction(model, prediction_df, config):
    """Run model prediction and extract results.

    Args:
        model: Loaded TFT model
        prediction_df: Prepared prediction DataFrame
        config: Full config dict

    Returns:
        Dict with prediction results
    """
    from src.dataset import build_timeseries_dataset

    dataset_cfg = config.get("dataset", {})

    # Create a TimeSeriesDataSet for prediction
    dataset = build_timeseries_dataset(
        prediction_df,
        encoder_length=dataset_cfg.get("encoder_length", 256),
        decoder_length=dataset_cfg.get("decoder_length", 15),
    )

    dataloader = dataset.to_dataloader(
        train=False, batch_size=1, num_workers=0
    )

    # Run prediction
    raw_output = model.predict(
        dataloader, mode="raw", return_x=False
    )

    # Extract point prediction and quantiles
    # raw_output is an Output namedtuple with .prediction (batch, horizon, n_quantiles)
    if hasattr(raw_output, "prediction"):
        pred_tensor = raw_output.prediction
    elif isinstance(raw_output, dict):
        pred_tensor = raw_output["prediction"]
    else:
        pred_tensor = raw_output

    pred_tensor = pred_tensor.cpu()

    if pred_tensor.ndim == 3:
        # Quantile output: (batch, horizon, n_quantiles)
        quantiles = pred_tensor[0].numpy()  # (horizon, n_quantiles)
        # Median is index 3 (of 7 quantiles: 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)
        point_pred = quantiles[:, 3]
        lower_bound = quantiles[:, 1]  # 10th percentile
        upper_bound = quantiles[:, 5]  # 90th percentile
    else:
        point_pred = pred_tensor[0].numpy()
        quantiles = None
        lower_bound = None
        upper_bound = None

    # Final step prediction (15 minutes ahead)
    final_pred = float(point_pred[-1])
    direction = "UP" if final_pred > 0 else "DOWN"
    magnitude_bps = abs(final_pred) * 10000  # Convert to basis points

    # Confidence from quantile spread (narrower spread = more confident)
    if lower_bound is not None and upper_bound is not None:
        spread = float(upper_bound[-1] - lower_bound[-1])
        # Normalize confidence: smaller spread = higher confidence
        # Use empirical scaling (typical spread for crypto is 0.001-0.01)
        confidence = max(0.0, min(1.0, 1.0 - spread * 100))
    else:
        confidence = None

    results = {
        "direction": direction,
        "predicted_return": final_pred,
        "magnitude_bps": magnitude_bps,
        "confidence": confidence,
        "horizon_minutes": 15,
        "predictions_all_steps": point_pred.tolist(),
    }

    if quantiles is not None:
        results["quantiles"] = {
            "p02": float(quantiles[-1, 0]),
            "p10": float(quantiles[-1, 1]),
            "p25": float(quantiles[-1, 2]),
            "p50": float(quantiles[-1, 3]),
            "p75": float(quantiles[-1, 4]),
            "p90": float(quantiles[-1, 5]),
            "p98": float(quantiles[-1, 6]),
        }

    return results


def print_prediction(results: dict, as_json: bool = False):
    """Print prediction results in a human-readable or JSON format."""
    if as_json:
        print(json.dumps(results, indent=2))
        return

    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"  Direction:        {results['direction']}")
    print(f"  Predicted Return: {results['predicted_return']:.6f}")
    print(f"  Magnitude:        {results['magnitude_bps']:.2f} bps")
    if results.get("confidence") is not None:
        print(f"  Confidence:       {results['confidence']:.2%}")
    print(f"  Horizon:          {results['horizon_minutes']} minutes")

    if "quantiles" in results:
        q = results["quantiles"]
        print(f"\n  Quantile Predictions (step 15):")
        print(f"    2%:  {q['p02']:.6f}")
        print(f"    10%: {q['p10']:.6f}")
        print(f"    25%: {q['p25']:.6f}")
        print(f"    50%: {q['p50']:.6f} (median)")
        print(f"    75%: {q['p75']:.6f}")
        print(f"    90%: {q['p90']:.6f}")
        print(f"    98%: {q['p98']:.6f}")

    print("=" * 50)


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # 1. Load config and model
    # -----------------------------------------------------------------------
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    print(f"Loading model from: {args.checkpoint}")
    model = TemporalFusionTransformer.load_from_checkpoint(args.checkpoint)
    model.eval()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # -----------------------------------------------------------------------
    # 2. Get input data
    # -----------------------------------------------------------------------
    encoder_length = config.get("dataset", {}).get("encoder_length", 256)
    needed_bars = encoder_length + 500 + 15  # encoder + warm-up + horizon

    if args.live:
        print(f"Fetching live data ({needed_bars} bars)...")
        pair = config.get("data", {}).get("pair", "BTC_USDT").replace("_", "/").upper()
        raw_df = fetch_live_data(pair=pair, limit=needed_bars)
        print(f"  Fetched {len(raw_df)} candles")
    elif args.data:
        print(f"Loading data from: {args.data}")
        data_path = Path(args.data)
        if data_path.suffix == ".csv":
            raw_df = pd.read_csv(data_path)
        else:
            raw_df = pd.read_parquet(data_path)
        # Take only the most recent bars needed
        raw_df = raw_df.tail(needed_bars).reset_index(drop=True)
        print(f"  Using last {len(raw_df)} candles")
    else:
        # Default: try to load from standard raw data location
        default_path = (
            PROJECT_ROOT
            / config.get("data", {}).get("raw_dir", "data/raw")
            / f"{config.get('data', {}).get('pair', 'BTC_USDT')}_1m.parquet"
        )
        if default_path.exists():
            print(f"Loading data from: {default_path}")
            raw_df = pd.read_parquet(default_path)
            raw_df = raw_df.tail(needed_bars).reset_index(drop=True)
            print(f"  Using last {len(raw_df)} candles")
        else:
            print(
                f"Error: No data source specified and default not found at {default_path}.\n"
                "Use --data <path> or --live to provide input data."
            )
            sys.exit(1)

    # -----------------------------------------------------------------------
    # 3. Compute features
    # -----------------------------------------------------------------------
    print("Computing features...")
    feature_df = compute_prediction_features(raw_df, config)
    print(f"  Feature matrix: {feature_df.shape}")

    # -----------------------------------------------------------------------
    # 4. Prepare prediction input
    # -----------------------------------------------------------------------
    print("Preparing prediction input...")
    prediction_df = prepare_prediction_dataframe(feature_df, config)
    print(f"  Prediction input: {prediction_df.shape}")

    # -----------------------------------------------------------------------
    # 5. Run prediction
    # -----------------------------------------------------------------------
    print("Running prediction...")
    with torch.no_grad():
        results = run_prediction(model, prediction_df, config)

    # -----------------------------------------------------------------------
    # 6. Output results
    # -----------------------------------------------------------------------
    print_prediction(results, as_json=args.json)


if __name__ == "__main__":
    main()
