"""
Model factory for TFT crypto predictor.

Loads configuration from YAML, creates a TemporalFusionTransformer
from a pytorch-forecasting TimeSeriesDataSet, and configures the
appropriate loss function.
"""

from pathlib import Path
from typing import Any, Dict

import yaml
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from src.loss import CombinedTradingLoss, create_quantile_loss


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def create_loss(config: Dict[str, Any]):
    """Create a loss function based on configuration.

    Args:
        config: Full config dict (expects a 'loss' section).

    Returns:
        A pytorch-forecasting compatible loss instance.
    """
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "quantile")

    if loss_type == "combined_trading":
        return CombinedTradingLoss(
            w_mse=loss_cfg.get("w_mse", 0.3),
            w_dir=loss_cfg.get("w_dir", 0.4),
            w_rw=loss_cfg.get("w_rw", 0.3),
            dead_zone=loss_cfg.get("dead_zone", 0.0),
        )
    else:
        # Default: QuantileLoss (recommended for stable training)
        quantiles = loss_cfg.get(
            "quantiles", [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        )
        return create_quantile_loss(quantiles=quantiles)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(
    training_dataset: TimeSeriesDataSet,
    config: Dict[str, Any],
) -> TemporalFusionTransformer:
    """Create a TFT model from a dataset and configuration.

    Uses TemporalFusionTransformer.from_dataset() which automatically
    configures input dimensions based on the dataset.

    Args:
        training_dataset: The training TimeSeriesDataSet (used to infer
            input/output dimensions and feature types).
        config: Full config dict (expects 'model', 'loss', 'training' sections).

    Returns:
        Configured TemporalFusionTransformer instance.
    """
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    loss = create_loss(config)

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=model_cfg.get("learning_rate", 1e-3),
        hidden_size=model_cfg.get("hidden_size", 64),
        attention_head_size=model_cfg.get("attention_head_size", 4),
        dropout=model_cfg.get("dropout", 0.15),
        hidden_continuous_size=model_cfg.get("hidden_continuous_size", 32),
        output_size=model_cfg.get("output_size", 7),
        loss=loss,
        log_interval=10,
        reduce_on_plateau_patience=training_cfg.get("reduce_lr_patience", 5),
        optimizer="adam",
    )

    return model


def load_trained_model(
    checkpoint_path: str | Path,
) -> TemporalFusionTransformer:
    """Load a trained TFT model from a checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file.

    Returns:
        Loaded TemporalFusionTransformer model in eval mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    return model
