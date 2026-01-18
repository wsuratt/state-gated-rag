#!/bin/bash
# Collect trajectories from WebShop

set -e

# Configuration
NUM_EPISODES=${1:-1000}
OUTPUT_DIR="data/trajectories"
CONFIG="configs/webshop.yaml"

echo "=== Collecting $NUM_EPISODES WebShop episodes ==="

# Create output directory
mkdir -p $OUTPUT_DIR

# Run collection
python -m data.collect_trajectories \
    --config $CONFIG \
    --output "$OUTPUT_DIR/webshop_baseline.jsonl" \
    --num_episodes $NUM_EPISODES

echo "=== Collection complete ==="
echo "Trajectories saved to: $OUTPUT_DIR/webshop_baseline.jsonl"

# Print stats
echo ""
echo "=== Dataset stats ==="
python -c "
import jsonlines
from collections import Counter

success = 0
total = 0
steps = []

with jsonlines.open('$OUTPUT_DIR/webshop_baseline.jsonl') as reader:
    for ep in reader:
        total += 1
        if ep.get('success', False):
            success += 1
        steps.append(ep.get('total_steps', 0))

print(f'Total episodes: {total}')
print(f'Success rate: {success/total:.2%}')
print(f'Avg steps: {sum(steps)/len(steps):.1f}')
"
