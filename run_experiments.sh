#!/usr/bin/env bash
set -e

RESULTS_DIR="results"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  Crypto-Predictor Full Experiment Suite"
echo "  $(date)"
echo "  Budget: \$5 Vast.ai — maximize learning"
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

# -----------------------------------------------
# Step 0: Download maximum historical data
# -----------------------------------------------
echo "============================================"
echo "  Step 0: Downloading ALL available BTC data"
echo "  (Sept 2017 → present, ~3.9M 1-min candles)"
echo "============================================"
python scripts/download_data.py --method bulk --start 2017-09-01
echo ""
echo "Recomputing features on full dataset..."
python scripts/compute_features.py
echo ""

# Generate experiment configs if not already done
if [ ! -d "configs/experiments" ] || [ -z "$(ls configs/experiments/*.yaml 2>/dev/null)" ]; then
    echo "Generating experiment configs..."
    python scripts/generate_experiments.py
fi

# Count experiments
TOTAL=$(ls configs/experiments/p*.yaml 2>/dev/null | wc -l | tr -d ' ')
echo "Total experiments to run: $TOTAL"
echo ""

# CSV header
echo "experiment,phase,val_loss,da_step_final,da_all_steps,sharpe,profit_factor,win_rate,max_drawdown,cumulative_pnl,rmse" > "$SUMMARY_CSV"

# -----------------------------------------------
# Run each experiment: train → evaluate → backtest
# -----------------------------------------------
CURRENT=0
FAILED=0

for CONFIG in configs/experiments/p*.yaml; do
    CURRENT=$((CURRENT + 1))
    EXP_NAME=$(basename "$CONFIG" .yaml)
    EXP_DIR="$RESULTS_DIR/$EXP_NAME"
    CKPT_DIR="models/tft/exp_$EXP_NAME"
    mkdir -p "$EXP_DIR"

    echo "============================================"
    echo "  [$CURRENT/$TOTAL] $EXP_NAME"
    echo "  Config: $CONFIG"
    echo "  $(date)"
    echo "============================================"

    # Clean previous checkpoints for this experiment
    rm -rf "$CKPT_DIR"
    mkdir -p "$CKPT_DIR"

    # Train (override checkpoint dir to keep experiments separate)
    if python scripts/train_tft.py --config "$CONFIG" 2>&1 | tee "$EXP_DIR/train.log"; then
        # Find best checkpoint (most recent)
        BEST_CKPT=$(ls -t models/tft/tft-*.ckpt 2>/dev/null | head -1)

        if [ -z "$BEST_CKPT" ]; then
            echo "  WARNING: No checkpoint found, skipping eval"
            echo "$EXP_NAME,$(grep -oP 'phase: \K\d+' "$CONFIG" || echo '?'),FAILED,,,,,,,," >> "$SUMMARY_CSV"
            FAILED=$((FAILED + 1))
            continue
        fi

        echo "  Best checkpoint: $BEST_CKPT"

        # Move checkpoint to experiment-specific dir
        cp "$BEST_CKPT" "$CKPT_DIR/"
        EXP_CKPT="$CKPT_DIR/$(basename "$BEST_CKPT")"

        # Evaluate
        python scripts/evaluate.py --checkpoint "$EXP_CKPT" --save-report \
            2>&1 | tee "$EXP_DIR/eval.log" || true

        # Backtest
        python scripts/backtest.py --checkpoint "$EXP_CKPT" \
            2>&1 | tee "$EXP_DIR/backtest.log" || true

        # Copy artifacts
        cp models/tft/backtest_equity.png "$EXP_DIR/equity.png" 2>/dev/null || true

        # Extract metrics for CSV (grep from eval log)
        VAL_LOSS=$(grep -oP 'val_loss[=:]\s*\K[0-9.]+' "$EXP_DIR/train.log" | tail -1 || echo "")
        DA_FINAL=$(grep -iP 'directional.*step.*(15|final|last)[^0-9]*\K[0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        DA_ALL=$(grep -iP 'directional.*all[^0-9]*\K[0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        SHARPE=$(grep -iP 'sharpe[^0-9]*\K[-0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        PF=$(grep -iP 'profit.factor[^0-9]*\K[0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        WR=$(grep -iP 'win.rate[^0-9]*\K[0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        MDD=$(grep -iP 'max.draw[^0-9]*\K[-0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        PNL=$(grep -iP 'cumulative.*p.l[^0-9]*\K[-0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")
        RMSE=$(grep -iP 'rmse[^0-9]*\K[0-9.]+' "$EXP_DIR/eval.log" | head -1 || echo "")

        PHASE=$(echo "$EXP_NAME" | grep -oP 'p\K\d+')
        echo "$EXP_NAME,$PHASE,$VAL_LOSS,$DA_FINAL,$DA_ALL,$SHARPE,$PF,$WR,$MDD,$PNL,$RMSE" >> "$SUMMARY_CSV"

        # Clean up checkpoints from models/tft/ to save disk
        rm -f models/tft/tft-*.ckpt

        echo "  Done: $EXP_NAME"
    else
        echo "  FAILED: $EXP_NAME"
        echo "$EXP_NAME,FAILED,,,,,,,,,," >> "$SUMMARY_CSV"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

# -----------------------------------------------
# Final Summary
# -----------------------------------------------
echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "============================================"
echo ""
echo "Completed: $((CURRENT - FAILED))/$TOTAL"
echo "Failed: $FAILED/$TOTAL"
echo ""
echo "Results summary CSV: $SUMMARY_CSV"
echo ""
echo "--- TOP 5 BY DIRECTIONAL ACCURACY (step final) ---"
sort -t',' -k4 -rn "$SUMMARY_CSV" | head -6
echo ""
echo "--- TOP 5 BY SHARPE RATIO ---"
sort -t',' -k6 -rn "$SUMMARY_CSV" | head -6
echo ""
echo "--- TOP 5 BY PROFIT FACTOR ---"
sort -t',' -k7 -rn "$SUMMARY_CSV" | head -6
echo ""
echo "Full results in: $RESULTS_DIR/"
echo ""
echo "To copy results to your local machine:"
echo "  scp -r <instance>:$(pwd)/results ."
