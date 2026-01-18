#!/bin/bash
# Phase 1: Train next-action prediction model (go/no-go validation)

set -e

# Configuration
DATA_PATH=${1:-"data/trajectories/webshop_baseline.jsonl"}
OUTPUT_DIR=${2:-"checkpoints/phase1"}
CONFIG="configs/training.yaml"

echo "=== Phase 1: Next-Action Prediction Training ==="
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"

# Check data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    echo "Run ./scripts/collect_data.sh first"
    exit 1
fi

# Run training
python -m training.train_next_action \
    --config $CONFIG \
    --data $DATA_PATH \
    --output $OUTPUT_DIR

echo "=== Training complete ==="
echo "Model saved to: $OUTPUT_DIR/best_model.pt"
