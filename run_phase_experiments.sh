#!/usr/bin/env bash
# Do NOT use set -e — we want to continue on individual experiment failures

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

# Convert CONFIGS to array to avoid subshell issue with piped while-read
CONFIG_ARRAY=()
while IFS= read -r line; do
    CONFIG_ARRAY+=("$line")
done <<< "$CONFIGS"

# Run each experiment: train → evaluate → backtest
CURRENT=0
FAILED=0
SUCCEEDED=0

for CONFIG in "${CONFIG_ARRAY[@]}"; do
    CURRENT=$((CURRENT + 1))
    EXP_NAME=$(basename "$CONFIG" .yaml)
    EXP_DIR="$RESULTS_DIR/$EXP_NAME"
    mkdir -p "$EXP_DIR"

    # Extract phase number (portable, no -P flag needed)
    PHASE=$(echo "$EXP_NAME" | sed 's/^p\([0-9]*\).*/\1/')

    echo "============================================"
    echo "  [$CURRENT/$TOTAL] $EXP_NAME"
    echo "  Config: $CONFIG"
    echo "  $(date)"
    echo "============================================"

    # Clean previous checkpoints
    rm -f models/tft/tft-*.ckpt

    # Train (capture exit code, don't let tee mask it)
    python scripts/train_tft.py --config "$CONFIG" > "$EXP_DIR/train.log" 2>&1
    TRAIN_EXIT=$?

    # Show last few lines of training output
    tail -5 "$EXP_DIR/train.log"

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "  FAILED (exit $TRAIN_EXIT): $EXP_NAME"
        echo "$EXP_NAME,$PHASE,FAILED,,,,,,,," >> "$SUMMARY_CSV"
        FAILED=$((FAILED + 1))
        echo ""
        continue
    fi

    BEST_CKPT=$(ls -t models/tft/tft-*.ckpt 2>/dev/null | head -1)

    if [ -z "$BEST_CKPT" ]; then
        echo "  WARNING: No checkpoint found after training"
        echo "$EXP_NAME,$PHASE,NO_CKPT,,,,,,,," >> "$SUMMARY_CSV"
        FAILED=$((FAILED + 1))
        echo ""
        continue
    fi

    echo "  Best checkpoint: $BEST_CKPT"

    # Evaluate
    python scripts/evaluate.py --checkpoint "$BEST_CKPT" --save-report \
        > "$EXP_DIR/eval.log" 2>&1 || true

    # Backtest
    python scripts/backtest.py --checkpoint "$BEST_CKPT" \
        > "$EXP_DIR/backtest.log" 2>&1 || true

    # Copy artifacts
    cp models/tft/backtest_equity.png "$EXP_DIR/equity.png" 2>/dev/null || true

    # Extract metrics (use grep -oP if available, fallback to Python)
    extract_metric() {
        local pattern="$1"
        local file="$2"
        python3 -c "
import re, sys
text = open('$file').read()
m = re.search(r'$pattern', text, re.IGNORECASE)
print(m.group(1) if m else '')
" 2>/dev/null || echo ""
    }

    VAL_LOSS=$(extract_metric 'val_loss[=:]\s*([0-9.]+)' "$EXP_DIR/train.log")
    DA_FINAL=$(extract_metric 'directional.*(?:step.*(?:15|final|last)|accuracy)[^\d]*(\d+\.\d+)' "$EXP_DIR/eval.log")
    SHARPE=$(extract_metric 'sharpe[^\d]*([-\d.]+)' "$EXP_DIR/eval.log")
    PF=$(extract_metric 'profit.factor[^\d]*(\d+\.\d+)' "$EXP_DIR/eval.log")
    WR=$(extract_metric 'win.rate[^\d]*(\d+\.\d+)' "$EXP_DIR/eval.log")
    MDD=$(extract_metric 'max.draw[^\d]*([-\d.]+)' "$EXP_DIR/eval.log")
    PNL=$(extract_metric 'cumulative.*p.l[^\d]*([-\d.]+)' "$EXP_DIR/eval.log")
    RMSE=$(extract_metric 'rmse[^\d]*(\d+\.\d+)' "$EXP_DIR/eval.log")

    echo "$EXP_NAME,$PHASE,$VAL_LOSS,$DA_FINAL,,$SHARPE,$PF,$WR,$MDD,$PNL,$RMSE" >> "$SUMMARY_CSV"

    # Clean up checkpoints to save disk
    rm -f models/tft/tft-*.ckpt

    SUCCEEDED=$((SUCCEEDED + 1))
    echo "  SUCCEEDED: $EXP_NAME ($SUCCEEDED done, $FAILED failed)"
    echo ""
done

echo ""
echo "============================================"
echo "  PHASE EXPERIMENTS COMPLETE"
echo "  Succeeded: $SUCCEEDED  Failed: $FAILED  Total: $TOTAL"
echo "  $(date)"
echo "============================================"
echo ""
echo "Results: $SUMMARY_CSV"
cat "$SUMMARY_CSV"
echo ""
echo "COMPLETE" > "$RESULTS_DIR/COMPLETE"
