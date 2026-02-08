#!/usr/bin/env python3
"""
Generate experiment configs for systematic hyperparameter exploration.
Creates YAML configs for each experiment in configs/experiments/
"""

import yaml
import os
import itertools
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs" / "experiments"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

# Load base config
with open(PROJECT_ROOT / "configs" / "tft_config.yaml") as f:
    BASE_CONFIG = yaml.safe_load(f)


def save_config(config, name):
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return path


experiments = []

# =============================================================
# PHASE 1: Horizon Sweep (find optimal prediction window)
# 5 experiments × ~5 min = 25 min
# =============================================================
for horizon in [1, 3, 5, 10, 15]:
    cfg = deepcopy(BASE_CONFIG)
    cfg["dataset"]["decoder_length"] = horizon
    cfg["training"]["max_epochs"] = 20
    cfg["training"]["early_stopping_patience"] = 5
    cfg["experiment"] = {
        "name": f"horizon_{horizon}min",
        "phase": 1,
        "description": f"Prediction horizon: {horizon} minutes",
        "target": f"log_return_{horizon}",
    }
    name = f"p1_horizon_{horizon}min"
    save_config(cfg, name)
    experiments.append({"name": name, "phase": 1, "config": str(CONFIGS_DIR / f"{name}.yaml")})

# =============================================================
# PHASE 2: Loss Function Comparison (at each horizon)
# 10 experiments × ~5 min = 50 min
# =============================================================
for horizon in [1, 3, 5, 10, 15]:
    cfg = deepcopy(BASE_CONFIG)
    cfg["dataset"]["decoder_length"] = horizon
    cfg["training"]["max_epochs"] = 20
    cfg["training"]["early_stopping_patience"] = 5
    cfg["loss"]["type"] = "combined_trading"
    cfg["experiment"] = {
        "name": f"trading_loss_{horizon}min",
        "phase": 2,
        "description": f"Combined trading loss at {horizon}-min horizon",
        "target": f"log_return_{horizon}",
    }
    name = f"p2_trading_loss_{horizon}min"
    save_config(cfg, name)
    experiments.append({"name": name, "phase": 2, "config": str(CONFIGS_DIR / f"{name}.yaml")})

# =============================================================
# PHASE 3: Model Architecture Search
# 12 experiments × ~5 min = 60 min
# =============================================================
arch_combos = [
    (32, 2, 0.10),   # tiny
    (32, 4, 0.15),
    (64, 2, 0.10),
    (64, 4, 0.15),   # current baseline
    (64, 4, 0.25),   # more regularized
    (64, 8, 0.15),
    (128, 4, 0.10),  # bigger
    (128, 4, 0.20),
    (128, 8, 0.15),
    (128, 8, 0.25),
    (256, 4, 0.20),  # large
    (256, 8, 0.25),
]

for hidden, heads, dropout in arch_combos:
    cfg = deepcopy(BASE_CONFIG)
    cfg["model"]["hidden_size"] = hidden
    cfg["model"]["attention_head_size"] = heads
    cfg["model"]["dropout"] = dropout
    cfg["model"]["hidden_continuous_size"] = max(16, hidden // 2)
    cfg["training"]["max_epochs"] = 20
    cfg["training"]["early_stopping_patience"] = 5
    cfg["experiment"] = {
        "name": f"arch_h{hidden}_a{heads}_d{int(dropout*100)}",
        "phase": 3,
        "description": f"Architecture: hidden={hidden}, heads={heads}, dropout={dropout}",
    }
    name = f"p3_arch_h{hidden}_a{heads}_d{int(dropout*100)}"
    save_config(cfg, name)
    experiments.append({"name": name, "phase": 3, "config": str(CONFIGS_DIR / f"{name}.yaml")})

# =============================================================
# PHASE 4: Encoder Length (lookback window)
# 5 experiments × ~5 min = 25 min
# =============================================================
for encoder_len in [64, 128, 256, 512, 1024]:
    cfg = deepcopy(BASE_CONFIG)
    cfg["dataset"]["encoder_length"] = encoder_len
    cfg["training"]["max_epochs"] = 20
    cfg["training"]["early_stopping_patience"] = 5
    cfg["experiment"] = {
        "name": f"encoder_{encoder_len}",
        "phase": 4,
        "description": f"Encoder (lookback) length: {encoder_len} bars",
    }
    name = f"p4_encoder_{encoder_len}"
    save_config(cfg, name)
    experiments.append({"name": name, "phase": 4, "config": str(CONFIGS_DIR / f"{name}.yaml")})

# =============================================================
# PHASE 5: Learning Rate Sweep
# 5 experiments × ~5 min = 25 min
# =============================================================
for lr in [0.0001, 0.0003, 0.001, 0.003, 0.01]:
    cfg = deepcopy(BASE_CONFIG)
    cfg["model"]["learning_rate"] = lr
    cfg["training"]["max_epochs"] = 20
    cfg["training"]["early_stopping_patience"] = 5
    cfg["experiment"] = {
        "name": f"lr_{lr}",
        "phase": 5,
        "description": f"Learning rate: {lr}",
    }
    name = f"p5_lr_{lr}"
    save_config(cfg, name)
    experiments.append({"name": name, "phase": 5, "config": str(CONFIGS_DIR / f"{name}.yaml")})

# =============================================================
# PHASE 6: Batch Size
# 4 experiments × ~5 min = 20 min
# =============================================================
for batch_size in [32, 64, 128, 256]:
    cfg = deepcopy(BASE_CONFIG)
    cfg["dataset"]["batch_size"] = batch_size
    cfg["training"]["max_epochs"] = 20
    cfg["training"]["early_stopping_patience"] = 5
    cfg["experiment"] = {
        "name": f"batch_{batch_size}",
        "phase": 6,
        "description": f"Batch size: {batch_size}",
    }
    name = f"p6_batch_{batch_size}"
    save_config(cfg, name)
    experiments.append({"name": name, "phase": 6, "config": str(CONFIGS_DIR / f"{name}.yaml")})

# =============================================================
# Save experiment manifest
# =============================================================
manifest = {
    "total_experiments": len(experiments),
    "estimated_time_a100": f"{len(experiments) * 5} minutes (~{len(experiments) * 5 / 60:.1f} hours)",
    "estimated_cost_vast_ai": f"${len(experiments) * 5 / 60 * 0.40:.2f}",
    "phases": {
        1: "Horizon Sweep (1, 3, 5, 10, 15 min)",
        2: "Loss Function (trading loss at each horizon)",
        3: "Architecture Search (hidden size, heads, dropout)",
        4: "Encoder Length (lookback window)",
        5: "Learning Rate Sweep",
        6: "Batch Size",
    },
    "experiments": experiments,
}

manifest_path = CONFIGS_DIR / "manifest.yaml"
with open(manifest_path, "w") as f:
    yaml.dump(manifest, f, default_flow_style=False)

print(f"Generated {len(experiments)} experiment configs in {CONFIGS_DIR}/")
print(f"Estimated GPU time on A100: {len(experiments) * 5} min (~{len(experiments) * 5 / 60:.1f} hrs)")
print(f"Estimated Vast.ai cost: ${len(experiments) * 5 / 60 * 0.40:.2f}")
print(f"Manifest saved to: {manifest_path}")

for phase in sorted(set(e["phase"] for e in experiments)):
    count = sum(1 for e in experiments if e["phase"] == phase)
    print(f"  Phase {phase}: {count} experiments — {manifest['phases'][phase]}")
