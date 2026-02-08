#!/usr/bin/env python3
"""
FastAPI REST API for TFT Crypto Predictor.

Serves model predictions via HTTP endpoints. Loads a trained TFT
checkpoint on startup and exposes /health, /predict, and /metrics.

Usage:
    python scripts/serve_api.py --checkpoint models/tft/best.ckpt
    python scripts/serve_api.py --checkpoint models/tft/best.ckpt --port 8080
"""

import argparse
import sys
import traceback
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from src.model import load_config
from src.features import (
    TARGET_COLUMN,
    TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_REALS,
)
from scripts.predict import (
    compute_prediction_features,
    fetch_live_data,
    prepare_prediction_dataframe,
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app = FastAPI(title="TFT Crypto Predictor API", version="1.0.0")

_model: TemporalFusionTransformer | None = None
_config: dict | None = None
_checkpoint_path: str | None = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, config_path: str | None = None):
    """Load model and config into global state."""
    global _model, _config, _checkpoint_path

    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "tft_config.yaml")

    _config = load_config(config_path)
    _model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    _model.eval()
    _checkpoint_path = checkpoint_path

    param_count = sum(p.numel() for p in _model.parameters())
    print(f"Model loaded from {checkpoint_path} ({param_count:,} parameters)")


# ---------------------------------------------------------------------------
# Prediction logic (self-contained to avoid predict.py's broken import)
# ---------------------------------------------------------------------------

def _build_prediction_dataset(prediction_df, encoder_length, decoder_length):
    """Build a TimeSeriesDataSet from a prepared prediction DataFrame."""
    # Ensure target column exists
    target = TARGET_COLUMN
    if target not in prediction_df.columns:
        # Fallback for predict.py's convention
        if "log_return_15" in prediction_df.columns:
            target = "log_return_15"
        else:
            prediction_df[TARGET_COLUMN] = 0.0

    return TimeSeriesDataSet(
        data=prediction_df,
        time_idx="time_idx",
        target=target,
        group_ids=["group"],
        max_encoder_length=encoder_length,
        max_prediction_length=decoder_length,
        min_encoder_length=max(encoder_length // 2, encoder_length - 30),
        min_prediction_length=decoder_length,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        static_categoricals=["group"],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )


def _run_prediction(model, prediction_df, config):
    """Run model prediction and extract results."""
    dataset_cfg = config.get("dataset", {})
    encoder_length = dataset_cfg.get("encoder_length", 256)
    decoder_length = dataset_cfg.get("decoder_length", 15)

    dataset = _build_prediction_dataset(
        prediction_df, encoder_length, decoder_length
    )
    dataloader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    raw_output = model.predict(dataloader, mode="raw", return_x=False)

    # Extract point prediction and quantiles
    # raw_output may be a dict, an Output namedtuple, or a tensor
    if hasattr(raw_output, "prediction"):
        pred_tensor = raw_output.prediction
    elif isinstance(raw_output, dict):
        pred_tensor = raw_output["prediction"]
    else:
        pred_tensor = raw_output

    pred_tensor = pred_tensor.cpu()

    if pred_tensor.ndim == 3:
        quantiles = pred_tensor[0].numpy()
        point_pred = quantiles[:, 3]
        lower_bound = quantiles[:, 1]
        upper_bound = quantiles[:, 5]
    else:
        point_pred = pred_tensor[0].numpy()
        quantiles = None
        lower_bound = None
        upper_bound = None

    final_pred = float(point_pred[-1])
    direction = "UP" if final_pred > 0 else "DOWN"
    magnitude_bps = abs(final_pred) * 10000

    if lower_bound is not None and upper_bound is not None:
        spread = float(upper_bound[-1] - lower_bound[-1])
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "checkpoint": _checkpoint_path,
    }


@app.get("/predict")
def predict():
    """Run a live prediction using recent market data.

    Fetches live OHLCV data, computes features, and returns the model's
    directional prediction with confidence and quantile information.
    """
    if _model is None or _config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Fetch live data
        encoder_length = _config.get("dataset", {}).get("encoder_length", 256)
        needed_bars = encoder_length + 300 + 15  # extra margin for indicator warm-up
        pair = _config.get("data", {}).get("pair", "BTC_USDT").replace("_", "/").upper()

        raw_df = fetch_live_data(pair=pair, limit=needed_bars)

        # 2. Compute features
        feature_df = compute_prediction_features(raw_df, _config)

        # 3. Prepare prediction input
        prediction_df = prepare_prediction_dataframe(feature_df, _config)

        # 4. Run prediction
        with torch.no_grad():
            results = _run_prediction(_model, prediction_df, _config)

        return results

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )


@app.get("/metrics")
def metrics():
    """Return the latest evaluation metrics from training.

    Reads models/tft/training_metrics.txt and returns its contents
    as structured JSON.
    """
    metrics_path = PROJECT_ROOT / "models" / "tft" / "training_metrics.txt"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Metrics file not found: {metrics_path}",
        )

    try:
        text = metrics_path.read_text()
        # Parse key-value lines (e.g. "MAE: 0.00123")
        parsed = {}
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("="):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                # Try to convert numeric values
                try:
                    value = float(value)
                except ValueError:
                    pass
                parsed[key] = value

        return {"raw": text, "parsed": parsed}

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read metrics: {exc}",
        )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve TFT crypto predictions via REST API"
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
        default=None,
        help="Path to config YAML (defaults to configs/tft_config.yaml)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model(args.checkpoint, args.config)
    uvicorn.run(app, host=args.host, port=args.port)
