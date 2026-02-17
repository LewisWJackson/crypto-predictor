"""
Dataset creation for TFT model using pytorch-forecasting.

Handles:
- Train/val/test split with 500-bar purge gaps (70/15/15)
- Z-score normalization from training data ONLY
- TimeSeriesDataSet creation
- Dataloader creation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from pytorch_forecasting import TimeSeriesDataSet

from src.features import (
    ALL_FEATURES,
    TARGET_COLUMN,
    TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_REALS,
    TEMPORAL_FEATURES,
)


def split_data(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    purge_gap: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train/val/test with purge gaps.

    Args:
        df: Feature DataFrame (already processed, no NaN).
        train_frac: Fraction for training.
        val_frac: Fraction for validation.
        purge_gap: Number of bars to skip between splits.

    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)

    train_end = int(n * train_frac)
    val_start = train_end + purge_gap
    val_end = val_start + int(n * val_frac)
    test_start = val_end + purge_gap

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[val_start:val_end].copy()
    test_df = df.iloc[test_start:].copy()

    return train_df, val_df, test_df


def compute_normalization_stats(
    train_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series]:
    """Compute z-score normalization statistics from training data ONLY.

    Temporal encodings (sin/cos) are excluded — they're already in [-1, 1].

    Returns:
        (means, stds) Series indexed by feature name.
    """
    # Normalize all features EXCEPT temporal encodings
    features_to_normalize = [
        f for f in TIME_VARYING_UNKNOWN_REALS if f in train_df.columns
    ]

    means = train_df[features_to_normalize].mean()
    stds = train_df[features_to_normalize].std()

    return means, stds


def apply_normalization(
    df: pd.DataFrame,
    means: pd.Series,
    stds: pd.Series,
) -> pd.DataFrame:
    """Apply z-score normalization using pre-computed statistics.

    Temporal features are NOT normalized (already in [-1, 1]).
    """
    df = df.copy()
    features_to_normalize = means.index.tolist()

    df[features_to_normalize] = (
        (df[features_to_normalize] - means) / (stds + 1e-8)
    ).astype(np.float32)

    return df


def prepare_for_timeseries(
    df: pd.DataFrame,
    group_name: str = "BTC_USDT",
    target_col: str = "forward_return_15",
    source_target_col: str = None,
) -> pd.DataFrame:
    """Add required columns for pytorch-forecasting TimeSeriesDataSet.

    Adds time_idx (contiguous integer index) and group column.
    Copies the forward-looking target into a dedicated column that
    cannot collide with backward-looking input features.

    Args:
        df: DataFrame with features and target.
        group_name: Group identifier for the time series.
        target_col: Name of the target column for pytorch-forecasting.
        source_target_col: Source column to copy into target_col.
            Defaults to TARGET_COLUMN ("target_log_return_15").
    """
    if source_target_col is None:
        source_target_col = TARGET_COLUMN

    df = df.copy()
    df["time_idx"] = np.arange(len(df))
    df["group"] = group_name

    # Always copy forward-looking target into the dedicated target column
    if source_target_col in df.columns:
        df[target_col] = df[source_target_col]
    else:
        raise ValueError(
            f"Source target column '{source_target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def create_datasets(
    processed_path: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    purge_gap: int = 500,
    encoder_length: int = 256,
    decoder_length: int = 15,
    group_name: str = "BTC_USDT",
    target_col: str = "forward_return_15",
    start_date: str = None,
    regime_model_path: str = None,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, Dict]:
    """Create train/val/test TimeSeriesDataSets from processed parquet.

    Args:
        processed_path: Path to processed features parquet file.
        train_frac: Training data fraction.
        val_frac: Validation data fraction.
        purge_gap: Bars to skip between splits.
        encoder_length: Lookback window length.
        decoder_length: Prediction horizon length.
        group_name: Group identifier for the time series.
        target_col: Name of the target column in the dataset. Must not
            collide with input feature names.
        start_date: Optional start date filter (e.g. "2022-01-01"). Data
            before this date is discarded before splitting.
        regime_model_path: Optional path to a fitted RegimeDetector. When
            provided, adds 'regime_label' as a time-varying unknown
            categorical feature to the TimeSeriesDataSet.

    Returns:
        (training_dataset, validation_dataset, test_dataset, norm_stats)
        norm_stats contains {"means": Series, "stds": Series}
    """
    # Load processed features
    df = pd.read_parquet(processed_path)

    # Filter by start_date if provided
    if start_date:
        ts = pd.to_datetime(df["timestamp"], unit="ms")
        mask = ts >= pd.Timestamp(start_date)
        before = len(df)
        df = df[mask].reset_index(drop=True)
        print(f"  Filtered to data from {start_date}: {before:,} → {len(df):,} rows")

    # Map target_col name to the source column in parquet
    # e.g. "forward_return_15" -> "target_log_return_15"
    #      "forward_return_5"  -> "target_log_return_5"
    TARGET_COL_MAP = {
        "forward_return_1": "target_log_return_1",
        "forward_return_5": "target_log_return_5",
        "forward_return_15": "target_log_return_15",
    }
    source_target_col = TARGET_COL_MAP.get(target_col, TARGET_COLUMN)

    # Compute regime labels BEFORE splitting if regime model is provided
    # (uses raw features, must happen before normalization)
    use_regime = regime_model_path is not None
    if use_regime:
        from src.features import compute_regime_features
        df = compute_regime_features(df, regime_model_path)
        print(f"  Added regime_label feature (4 categories)")

    # Split chronologically
    train_df, val_df, test_df = split_data(df, train_frac, val_frac, purge_gap)

    # Compute normalization from training data only
    means, stds = compute_normalization_stats(train_df)

    # Apply normalization to all splits
    train_df = apply_normalization(train_df, means, stds)
    val_df = apply_normalization(val_df, means, stds)
    test_df = apply_normalization(test_df, means, stds)

    # Add time_idx and group columns (must be contiguous per split)
    train_df = prepare_for_timeseries(train_df, group_name, target_col=target_col, source_target_col=source_target_col)
    val_df = prepare_for_timeseries(val_df, group_name, target_col=target_col, source_target_col=source_target_col)
    test_df = prepare_for_timeseries(test_df, group_name, target_col=target_col, source_target_col=source_target_col)

    # Ensure regime_label is string type for pytorch-forecasting categoricals
    if use_regime:
        for split_df in [train_df, val_df, test_df]:
            split_df["regime_label"] = split_df["regime_label"].astype(str)

    # Build categorical feature lists
    time_varying_unknown_cats = ["regime_label"] if use_regime else []

    # Create training dataset
    training_dataset = TimeSeriesDataSet(
        data=train_df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["group"],
        max_encoder_length=encoder_length,
        max_prediction_length=decoder_length,
        min_encoder_length=encoder_length,
        min_prediction_length=decoder_length,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        time_varying_unknown_categoricals=time_varying_unknown_cats,
        static_categoricals=["group"],
        target_normalizer=None,  # We pre-normalize ourselves
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )

    # Create val/test datasets with same schema as training
    ts_kwargs = dict(
        time_idx="time_idx",
        target=target_col,
        group_ids=["group"],
        max_encoder_length=encoder_length,
        max_prediction_length=decoder_length,
        min_encoder_length=encoder_length,
        min_prediction_length=decoder_length,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        time_varying_unknown_categoricals=time_varying_unknown_cats,
        static_categoricals=["group"],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )

    validation_dataset = TimeSeriesDataSet(data=val_df, **ts_kwargs)
    test_dataset = TimeSeriesDataSet(data=test_df, **ts_kwargs)

    norm_stats = {"means": means, "stds": stds}

    return training_dataset, validation_dataset, test_dataset, norm_stats


def build_timeseries_dataset(
    df: pd.DataFrame,
    encoder_length: int = 256,
    decoder_length: int = 15,
    group_name: str = "BTC_USDT",
    target_col: str = "forward_return_15",
) -> TimeSeriesDataSet:
    """Create a single TimeSeriesDataSet for prediction from a prepared DataFrame.

    Args:
        df: DataFrame with feature columns and target.
        encoder_length: Lookback window length.
        decoder_length: Prediction horizon length.
        group_name: Group identifier for the time series.
        target_col: Target column name.

    Returns:
        TimeSeriesDataSet ready for prediction.
    """
    df = prepare_for_timeseries(df, group_name, target_col=target_col)

    return TimeSeriesDataSet(
        data=df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["group"],
        max_encoder_length=encoder_length,
        max_prediction_length=decoder_length,
        min_encoder_length=encoder_length,
        min_prediction_length=decoder_length,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        static_categoricals=["group"],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )


def create_dataloaders(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    test_dataset: TimeSeriesDataSet,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """Create dataloaders from TimeSeriesDataSet objects.

    Returns:
        (train_dataloader, val_dataloader, test_dataloader)
    """
    train_dataloader = training_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0,
    )

    val_dataloader = validation_dataset.to_dataloader(
        train=False,
        batch_size=batch_size * 2,
        num_workers=0,
    )

    test_dataloader = test_dataset.to_dataloader(
        train=False,
        batch_size=batch_size * 2,
        num_workers=0,
    )

    return train_dataloader, val_dataloader, test_dataloader
