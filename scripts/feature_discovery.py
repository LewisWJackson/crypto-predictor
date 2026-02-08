#!/usr/bin/env python3
"""
Automated Feature Discovery Pipeline.

Iteratively tests each alternative data feature to find which ones
actually improve prediction accuracy. This is the "think differently"
engine — it systematically evaluates unconventional signals.

Process:
1. Train a baseline model (standard features only)
2. For each alternative feature, train a model with that feature added
3. Compare directional accuracy, Sharpe ratio, and profit factor
4. Rank features by improvement over baseline
5. Build a "best features" model combining the top performers
6. Repeat with combinations of top features

Usage:
    python scripts/feature_discovery.py --max-trials 50
    python scripts/feature_discovery.py --quick  # 3 epochs per trial
"""

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "feature_discovery"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    config_path = PROJECT_ROOT / "configs" / "tft_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_baseline_features(processed_path: str) -> list[str]:
    """Get the list of standard technical indicator features."""
    df = pd.read_parquet(processed_path)
    # Exclude target, timestamp, and group columns
    exclude = {"timestamp", "log_return_15", "log_return_5", "time_idx", "group"}
    return [c for c in df.columns if c not in exclude]


def get_alternative_features(enhanced_path: str, baseline_features: list[str]) -> list[str]:
    """Get the list of alternative features (not in baseline)."""
    df = pd.read_parquet(enhanced_path)
    exclude = {"timestamp", "log_return_15", "log_return_5", "time_idx", "group"}
    all_features = [c for c in df.columns if c not in exclude]
    return [f for f in all_features if f not in baseline_features]


def train_and_evaluate(config: dict, features: list[str], data_path: str,
                       epochs: int = 5, trial_name: str = "trial") -> dict:
    """Train a model with specific features and return metrics."""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from src.dataset import create_datasets, create_dataloaders
    from src.metrics import compute_metrics
    from src.model import create_model

    data_cfg = config.get("data", {})
    dataset_cfg = config.get("dataset", {})
    pair = data_cfg.get("pair", "BTC_USDT")

    try:
        training_ds, val_ds, test_ds, norm_stats = create_datasets(
            processed_path=data_path,
            train_frac=dataset_cfg.get("train_fraction", 0.70),
            val_frac=dataset_cfg.get("val_fraction", 0.15),
            purge_gap=dataset_cfg.get("purge_gap", 500),
            encoder_length=dataset_cfg.get("encoder_length", 256),
            decoder_length=dataset_cfg.get("decoder_length", 15),
            group_name=pair,
        )

        train_dl, val_dl, test_dl = create_dataloaders(
            training_ds, val_ds, test_ds,
            batch_size=dataset_cfg.get("batch_size", 64),
            num_workers=dataset_cfg.get("num_workers", 0),
        )

        model = create_model(training_ds, config)

        ckpt_dir = RESULTS_DIR / "checkpoints" / trial_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        trainer = pl.Trainer(
            max_epochs=epochs,
            gradient_clip_val=1.0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=False),
                ModelCheckpoint(dirpath=str(ckpt_dir), monitor="val_loss",
                                filename="best", save_top_k=1, mode="min", verbose=False),
            ],
            enable_progress_bar=True,
            accelerator="auto",
            devices=1,
            default_root_dir=str(ckpt_dir),
        )

        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # Get best val_loss
        best_val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
        if hasattr(best_val_loss, "item"):
            best_val_loss = best_val_loss.item()

        # Evaluate on test set
        from pytorch_forecasting import TemporalFusionTransformer
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            best_model = TemporalFusionTransformer.load_from_checkpoint(best_path)
            predictions = best_model.predict(test_dl, mode="prediction", return_x=False)

            actuals = []
            for batch in test_dl:
                x, y = batch
                if isinstance(y, (tuple, list)):
                    actuals.append(y[0].numpy())
                else:
                    actuals.append(y.numpy())

            y_true = np.concatenate(actuals, axis=0)
            y_pred = predictions.numpy()

            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

            if y_true.ndim == 1:
                y_true = y_true[:, np.newaxis]
            if y_pred.ndim == 1:
                y_pred = y_pred[:, np.newaxis]

            metrics = compute_metrics(y_true, y_pred)
            metrics["val_loss"] = best_val_loss
            return metrics

        return {"val_loss": best_val_loss, "error": "no_checkpoint"}

    except Exception as e:
        return {"error": str(e), "val_loss": float("inf")}


def run_feature_discovery(enhanced_data_path: str, epochs: int = 5,
                          max_trials: int = 50):
    """Main feature discovery loop."""
    config = load_config()
    processed_path = str(PROJECT_ROOT / "data" / "processed" / "btc_usdt_features.parquet")

    baseline_features = get_baseline_features(processed_path)
    alt_features = get_alternative_features(enhanced_data_path, baseline_features)

    print(f"Baseline features: {len(baseline_features)}")
    print(f"Alternative features to test: {len(alt_features)}")
    print(f"Max trials: {max_trials}")
    print(f"Epochs per trial: {epochs}")
    print()

    # Phase 1: Baseline
    print("=" * 60)
    print("PHASE 1: Training baseline model")
    print("=" * 60)
    baseline_metrics = train_and_evaluate(
        config, baseline_features, processed_path,
        epochs=epochs, trial_name="baseline"
    )
    print(f"Baseline: val_loss={baseline_metrics.get('val_loss', '?')}, "
          f"DA={baseline_metrics.get('directional_accuracy_final_step', '?')}")

    # Phase 2: Test each alternative feature individually
    print()
    print("=" * 60)
    print("PHASE 2: Testing individual alternative features")
    print("=" * 60)

    results = []
    trial_count = 0

    for i, feature in enumerate(alt_features):
        if trial_count >= max_trials:
            print(f"Reached max trials ({max_trials}), stopping.")
            break

        trial_count += 1
        print(f"\n[{trial_count}/{min(len(alt_features), max_trials)}] "
              f"Testing feature: {feature}")

        trial_metrics = train_and_evaluate(
            config, baseline_features + [feature], enhanced_data_path,
            epochs=epochs, trial_name=f"feat_{feature}"
        )

        # Compute improvement over baseline
        improvement = {}
        for metric in ["directional_accuracy_final_step", "sharpe_ratio",
                        "profit_factor", "val_loss"]:
            baseline_val = baseline_metrics.get(metric, 0)
            trial_val = trial_metrics.get(metric, 0)
            if baseline_val and trial_val:
                if metric == "val_loss":
                    improvement[metric] = baseline_val - trial_val  # Lower is better
                else:
                    improvement[metric] = trial_val - baseline_val

        result = {
            "feature": feature,
            "metrics": trial_metrics,
            "improvement": improvement,
        }
        results.append(result)

        da = trial_metrics.get("directional_accuracy_final_step", "?")
        da_imp = improvement.get("directional_accuracy_final_step", 0)
        print(f"  DA={da}, improvement={da_imp:+.4f}" if isinstance(da_imp, float) else f"  DA={da}")

    # Phase 3: Rank and combine top features
    print()
    print("=" * 60)
    print("PHASE 3: Ranking features")
    print("=" * 60)

    # Sort by DA improvement
    valid_results = [r for r in results if "error" not in r["metrics"]]
    ranked = sorted(valid_results,
                    key=lambda r: r["improvement"].get("directional_accuracy_final_step", -999),
                    reverse=True)

    print(f"\nTop features by directional accuracy improvement:")
    print(f"{'Rank':<6} {'Feature':<35} {'DA Improvement':<16} {'Sharpe Imp':<12}")
    print("-" * 70)
    for i, r in enumerate(ranked[:20]):
        da_imp = r["improvement"].get("directional_accuracy_final_step", 0)
        sh_imp = r["improvement"].get("sharpe_ratio", 0)
        print(f"{i+1:<6} {r['feature']:<35} {da_imp:+.4f}{'':8} {sh_imp:+.4f}" if isinstance(da_imp, float) and isinstance(sh_imp, float) else f"{i+1:<6} {r['feature']:<35}")

    # Phase 4: Test combination of top N features
    top_n_values = [3, 5, 10]
    for n in top_n_values:
        if len(ranked) < n:
            continue

        top_features = [r["feature"] for r in ranked[:n]]
        print(f"\n[Combo] Testing top {n} features together: {top_features}")
        combo_metrics = train_and_evaluate(
            config, baseline_features + top_features, enhanced_data_path,
            epochs=epochs, trial_name=f"combo_top{n}"
        )
        combo_da = combo_metrics.get("directional_accuracy_final_step", "?")
        baseline_da = baseline_metrics.get("directional_accuracy_final_step", 0)
        print(f"  Top-{n} combo DA: {combo_da} (baseline: {baseline_da})")

    # Save full results
    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline": baseline_metrics,
        "feature_rankings": [
            {
                "rank": i + 1,
                "feature": r["feature"],
                "da_improvement": r["improvement"].get("directional_accuracy_final_step"),
                "sharpe_improvement": r["improvement"].get("sharpe_ratio"),
                "pf_improvement": r["improvement"].get("profit_factor"),
                "val_loss": r["metrics"].get("val_loss"),
            }
            for i, r in enumerate(ranked)
        ],
    }

    report_path = RESULTS_DIR / "discovery_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {report_path}")

    # Save best config
    if ranked:
        best_features = [r["feature"] for r in ranked if
                         r["improvement"].get("directional_accuracy_final_step", 0) > 0]
        if best_features:
            best_config = {
                "alternative_features": best_features,
                "total_features": len(baseline_features) + len(best_features),
                "baseline_features": len(baseline_features),
                "discovered_at": datetime.now().isoformat(),
            }
            config_path = PROJECT_ROOT / "configs" / "best_features.yaml"
            with open(config_path, "w") as f:
                yaml.dump(best_config, f)
            print(f"Best features config saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Automated feature discovery")
    parser.add_argument("--data", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed" / "btc_usdt_features_enhanced.parquet"),
                        help="Path to enhanced features parquet")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Epochs per trial (default: 5)")
    parser.add_argument("--max-trials", type=int, default=50,
                        help="Maximum number of feature trials (default: 50)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 epochs per trial")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 3

    run_feature_discovery(args.data, epochs=args.epochs, max_trials=args.max_trials)


if __name__ == "__main__":
    main()
