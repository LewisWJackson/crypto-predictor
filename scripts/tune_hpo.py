#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for the TFT crypto predictor.

Searches over key model/training hyperparameters using short training
runs (3-5 epochs) to find the best configuration, then saves the
winning parameters to configs/best_hpo_params.yaml.

Usage:
    python scripts/tune_hpo.py
    python scripts/tune_hpo.py --n-trials 50
    python scripts/tune_hpo.py --config configs/tft_config.yaml --n-trials 20
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
import yaml
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

from src.dataset import create_datasets, create_dataloaders
from src.model import create_model, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna HPO for TFT crypto predictor"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "tft_config.yaml"),
        help="Path to base config YAML file",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of Optuna trials to run (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Max epochs per trial (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "best_hpo_params.yaml"),
        help="Path to save best hyperparameters",
    )
    return parser.parse_args()


def objective(trial: optuna.Trial, base_config: dict, max_epochs: int) -> float:
    """Optuna objective: train a TFT with sampled hyperparameters and return val_loss."""

    # --- Sample hyperparameters ---
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    attention_head_size = trial.suggest_categorical("attention_head_size", [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    encoder_length = trial.suggest_categorical("encoder_length", [128, 256, 512])

    # --- Build trial config by overriding base config ---
    config = {**base_config}
    config["model"] = {
        **base_config.get("model", {}),
        "hidden_size": hidden_size,
        "attention_head_size": attention_head_size,
        "dropout": dropout,
        "learning_rate": learning_rate,
    }
    config["dataset"] = {
        **base_config.get("dataset", {}),
        "encoder_length": encoder_length,
    }

    # --- Create datasets with the trial's encoder_length ---
    dataset_cfg = config.get("dataset", {})
    data_cfg = config.get("data", {})
    processed_dir = PROJECT_ROOT / data_cfg.get("processed_dir", "data/processed")
    pair = data_cfg.get("pair", "BTC_USDT")
    processed_path = str(processed_dir / f"{pair}_features.parquet")

    training_dataset, validation_dataset, _, _ = create_datasets(
        processed_path=processed_path,
        train_frac=dataset_cfg.get("train_fraction", 0.70),
        val_frac=dataset_cfg.get("val_fraction", 0.15),
        purge_gap=dataset_cfg.get("purge_gap", 500),
        encoder_length=encoder_length,
        decoder_length=dataset_cfg.get("decoder_length", 15),
        group_name=pair,
    )

    train_dataloader, val_dataloader, _ = create_dataloaders(
        training_dataset,
        validation_dataset,
        validation_dataset,  # placeholder for test (unused)
        batch_size=dataset_cfg.get("batch_size", 64),
        num_workers=dataset_cfg.get("num_workers", 0),
    )

    # --- Create model ---
    model = create_model(training_dataset, config)

    # --- Train for a few epochs ---
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=config.get("training", {}).get("gradient_clip_val", 1.0),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=False),
        ],
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="auto",
        devices=1,
        logger=False,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Return the best validation loss from this trial
    val_loss = trainer.callback_metrics.get("val_loss")
    if val_loss is None:
        return float("inf")
    return val_loss.item()


def main():
    args = parse_args()

    print(f"Loading base config from: {args.config}")
    base_config = load_config(args.config)

    print(f"Starting Optuna HPO with {args.n_trials} trials, {args.epochs} epochs each")
    print("Search space:")
    print("  hidden_size:        [32, 64, 128]")
    print("  attention_head_size: [2, 4, 8]")
    print("  dropout:            [0.05, 0.3]")
    print("  learning_rate:      [1e-4, 1e-2] (log)")
    print("  encoder_length:     [128, 256, 512]")

    study = optuna.create_study(
        direction="minimize",
        study_name="tft_crypto_hpo",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
    )

    study.optimize(
        lambda trial: objective(trial, base_config, args.epochs),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # --- Report results ---
    print("\n" + "=" * 60)
    print("HPO COMPLETE")
    print("=" * 60)
    print(f"  Best trial:      #{study.best_trial.number}")
    print(f"  Best val_loss:   {study.best_value:.6f}")
    print(f"  Best params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # --- Save best params to YAML ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_params = {
        "best_trial": study.best_trial.number,
        "best_val_loss": float(study.best_value),
        "hyperparameters": {
            "model": {
                "hidden_size": study.best_params["hidden_size"],
                "attention_head_size": study.best_params["attention_head_size"],
                "dropout": round(study.best_params["dropout"], 4),
                "learning_rate": float(study.best_params["learning_rate"]),
            },
            "dataset": {
                "encoder_length": study.best_params["encoder_length"],
            },
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False, sort_keys=False)

    print(f"\nBest hyperparameters saved to: {output_path}")


if __name__ == "__main__":
    main()
