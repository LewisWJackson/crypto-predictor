#!/usr/bin/env bash
set -e

echo "============================================"
echo "  Crypto-Predictor: Feature Discovery Mode"
echo "  $(date)"
echo "  Finding alpha nobody else has"
echo "============================================"
echo ""

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: No GPU detected"
fi
echo ""

# -----------------------------------------------
# Step 1: Download ALL historical BTC data
# -----------------------------------------------
echo "============================================"
echo "  Step 1: Download full BTC/USDT history"
echo "  (Sept 2017 → present, ~3.9M candles)"
echo "============================================"
if [ ! -f "data/raw/btc_usdt_1m.parquet" ]; then
    python scripts/download_data.py --method bulk --start 2017-09-01
else
    echo "  Raw data already exists, skipping download"
fi
echo ""

# -----------------------------------------------
# Step 2: Compute standard features
# -----------------------------------------------
echo "============================================"
echo "  Step 2: Compute standard technical features"
echo "============================================"
if [ ! -f "data/processed/btc_usdt_features.parquet" ]; then
    python scripts/compute_features.py
else
    echo "  Features already computed, skipping"
fi
echo ""

# -----------------------------------------------
# Step 3: Fetch ALL alternative data
# -----------------------------------------------
echo "============================================"
echo "  Step 3: Fetch alternative data sources"
echo "  (Moon phases, Fear/Greed, Gold, DXY, Oil,"
echo "   S&P500, VIX, Corn, Copper, Google Trends,"
echo "   On-chain metrics, Exchange flows, Temporal)"
echo "============================================"
python scripts/fetch_alternative_data.py \
    --all \
    --start 2017-09-01 \
    --merge data/processed/btc_usdt_features.parquet \
    --output data/processed/btc_usdt_features_enhanced.parquet
echo ""

# -----------------------------------------------
# Step 4: Run feature discovery
# -----------------------------------------------
echo "============================================"
echo "  Step 4: Automated Feature Discovery"
echo "  Testing each alternative feature vs baseline"
echo "============================================"
python scripts/feature_discovery.py \
    --data data/processed/btc_usdt_features_enhanced.parquet \
    --epochs 5 \
    --max-trials 100
echo ""

# -----------------------------------------------
# Step 5: Run standard experiments with best features
# -----------------------------------------------
echo "============================================"
echo "  Step 5: Full experiment suite"
echo "  (Horizon, Loss, Architecture, LR, Batch)"
echo "============================================"
python scripts/generate_experiments.py
bash run_experiments.sh
echo ""

# -----------------------------------------------
# Summary
# -----------------------------------------------
echo "============================================"
echo "  DISCOVERY COMPLETE"
echo "  $(date)"
echo "============================================"
echo ""
echo "Key outputs:"
echo "  Feature rankings:  results/feature_discovery/discovery_report.json"
echo "  Best features:     configs/best_features.yaml"
echo "  Experiment results: results/summary.csv"
echo ""
echo "To copy everything to your Mac:"
echo "  scp -r <instance>:$(pwd)/results ."
echo "  scp -r <instance>:$(pwd)/configs/best_features.yaml ."
