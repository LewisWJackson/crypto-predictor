#!/usr/bin/env bash
set -e

# Usage: bash run_phase_experiments.sh 1 2     (runs phase 1 and 2 experiments)
#        bash run_phase_experiments.sh 3        (runs phase 3 only)
#        bash run_phase_experiments.sh all      (runs all phases)

if [ $# -eq 0 ]; then
    echo "Usage: $0 <phase_numbers...>"
    echo "  e.g.: $0 1 2       (runs p1 and p2 experiments)"
    echo "  e.g.: $0 all       (runs all experiments)"
    exit 1
fi

RESULTS_DIR="results"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  Crypto-Predictor Phase Experiments"
echo "  Phases: $@"
echo "  $(date)"
echo "============================================"
echo ""

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: No GPU detected — training will be slow"
fi
echo ""

# Build config file list from requested phases
CONFIGS=""
if [ "$1" = "all" ]; then
    CONFIGS=$(ls configs/experiments/p*.yaml 2>/dev/null)
else
    for PHASE in "$@"; do
        PHASE_CONFIGS=$(ls configs/experiments/p${PHASE}_*.yaml 2>/dev/null || true)
        CONFIGS="$CONFIGS $PHASE_CONFIGS"
    done
fi

CONFIGS=$(echo "$CONFIGS" | tr ' ' '\n' | sort | grep -v '^$')
TOTAL=$(echo "$CONFIGS" | wc -l | tr -d ' ')

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No experiment configs found for phases: $@"
    exit 1
fi

echo "Experiments to run: $TOTAL"
echo "$CONFIGS" | while read f; do echo "  $(basename $f)"; done
echo ""

# CSV header
echo "experiment,phase,val_loss,da_step_final,da_all_steps,sharpe,profit_factor,win_rate,max_drawdown,cumulative_pnl,rmse" > "$SUMMARY_CSV"

# Run each experiment: train → evaluate → backtest
CURRENT=0
FAILED=0
SUCCEEDED=0

echo "$CONFIGS" | while read CONFIG; do
    CURRENT=$((CURRENT + 1))
    EXP_NAME=$(basename "$CONFIG" .yaml)
    EXP_DIR="$RESULTS_DIR/$EXP_NAME"
    mkdir -p "$EXP_DIR"

    echo "============================================"
    echo "  [$CURRENT/$TOTAL] $EXP_NAME"
    echo "  Config: $CONFIG"
    echo "  $(date)"
    echo "============================================"

    # Clean previous checkpoints
    rm -f models/tft/tft-*.ckpt

    # Train
    if python scripts/train_tft.py --config "$CONFIG" 2>&1 | tee "$EXP_DIR/train.log"; then
        BEST_CKPT=$(ls -t models/tft/tft-*.ckpt 2>/dev/null | head -1)

        if [ -z "$BEST_CKPT" ]; then
            echo "  WARNING: No checkpoint found"
            PHASE=$(echo "$EXP_NAME" | grep -oP 'p\K\d+' || echo '?')
            echo "$EXP_NAME,$PHASE,NO_CKPT,,,,,,,," >> "$SUMMARY_CSV"
            continue
        fi

        echo "  Best checkpoint: $BEST_CKPT"

        # Evaluate
        python scripts/evaluate.py --checkpoint "$BEST_CKPT" --save-report \
            2>&1 | tee "$EXP_DIR/eval.log" || true

        # Backtest
        python scripts/backtest.py --checkpoint "$BEST_CKPT" \
            2>&1 | tee "$EXP_DIR/backtest.log" || true

        # Copy artifacts
        cp models/tft/backtest_equity.png "$EXP_DIR/equity.png" 2>/dev/null || true

        # Extract metrics
        VAL_LOSS=$(grep -oP 'val_loss[=:]\s*\K[0-9.]+' "$EXP_DIR/train.log" | tail -1 || echo "")
        DA_FINAL=$(grep -iP 'directional.*step.*(15|final|last)[^0-9]*\K[0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        DA_ALL=$(grep -iP 'directional.*all[^0-9]*\K[0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        SHARPE=$(grep -iP 'sharpe[^0-9]*\K[-0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        PF=$(grep -iP 'profit.factor[^0-9]*\K[0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        WR=$(grep -iP 'win.rate[^0-9]*\K[0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        MDD=$(grep -iP 'max.draw[^0-9]*\K[-0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        PNL=$(grep -iP 'cumulative.*p.l[^0-9]*\K[-0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        RMSE=$(grep -iP 'rmse[^0-9]*\K[0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")

        PHASE=$(echo "$EXP_NAME" | grep -oP 'p\K\d+' || echo '?')
        echo "$EXP_NAME,$PHASE,$VAL_LOSS,$DA_FINAL,$DA_ALL,$SHARPE,$PF,$WR,$MDD,$PNL,$RMSE" >> "$SUMMARY_CSV"

        # Clean up checkpoints to save disk
        rm -f models/tft/tft-*.ckpt

        echo "  SUCCEEDED: $EXP_NAME"
    else
        echo "  FAILED: $EXP_NAME"
        PHASE=$(echo "$EXP_NAME" | grep -oP 'p\K\d+' || echo '?')
        echo "$EXP_NAME,$PHASE,FAILED,,,,,,,," >> "$SUMMARY_CSV"
    fi
    echo ""
done

echo ""
echo "============================================"
echo "  PHASE EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "============================================"
echo ""
echo "Results: $SUMMARY_CSV"
cat "$SUMMARY_CSV"
echo ""
echo "COMPLETE" > "$RESULTS_DIR/COMPLETE"
