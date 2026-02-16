#!/usr/bin/env bash
set -e

# Usage: bash run_single_experiment.sh <config_filename>
# e.g.:  bash run_single_experiment.sh p1_horizon_1min.yaml

CONFIG_NAME="$1"
CONFIG="configs/experiments/$CONFIG_NAME"
EXP_NAME=$(basename "$CONFIG_NAME" .yaml)

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR/$EXP_NAME"

echo "============================================"
echo "  Single Experiment: $EXP_NAME"
echo "  Config: $CONFIG"
echo "  $(date)"
echo "============================================"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Train
echo "Training..."
if python scripts/train_tft.py --config "$CONFIG" 2>&1 | tee "$RESULTS_DIR/$EXP_NAME/train.log"; then
    BEST_CKPT=$(ls -t models/tft/tft-*.ckpt 2>/dev/null | head -1)

    if [ -z "$BEST_CKPT" ]; then
        echo "WARNING: No checkpoint found"
        echo "$EXP_NAME,NO_CKPT" > "$RESULTS_DIR/$EXP_NAME/status.txt"
        exit 1
    fi

    echo "Best checkpoint: $BEST_CKPT"

    # Evaluate
    echo "Evaluating..."
    python scripts/evaluate.py --checkpoint "$BEST_CKPT" --config "$CONFIG" --save-report \
        2>&1 | tee "$RESULTS_DIR/$EXP_NAME/eval.log" || true

    # Backtest
    echo "Backtesting..."
    python scripts/backtest.py --checkpoint "$BEST_CKPT" --config "$CONFIG" \
        2>&1 | tee "$RESULTS_DIR/$EXP_NAME/backtest.log" || true

    cp models/tft/backtest_equity.png "$RESULTS_DIR/$EXP_NAME/equity.png" 2>/dev/null || true
    cp models/tft/tft-*.ckpt "$RESULTS_DIR/$EXP_NAME/" 2>/dev/null || true

    echo "$EXP_NAME,SUCCESS,$(date)" > "$RESULTS_DIR/$EXP_NAME/status.txt"
    echo "DONE: $EXP_NAME at $(date)"
else
    echo "$EXP_NAME,FAILED,$(date)" > "$RESULTS_DIR/$EXP_NAME/status.txt"
    echo "FAILED: $EXP_NAME"
fi

echo "COMPLETE" > "$RESULTS_DIR/COMPLETE"
