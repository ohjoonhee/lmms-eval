#!/usr/bin/env bash
# Compare Qwen3-VL-4B-Thinking base vs SFT on vstar_bench
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output"

# Base model
PATH1="$OUTPUT_DIR/vstar_bench-base/Qwen__Qwen3-VL-4B-Thinking/20251224_220655_samples_vstar_bench.jsonl"

# SFT variant (default: Viscot46k-SFT-v1)
PATH2="$OUTPUT_DIR/vstar_bench-base/ohjoonhee__Qwen3-VL-4B-Thinking-Viscot46k-SFT-v1/20260128_154752_samples_vstar_bench.jsonl"

# Default score key for vstar_bench
SCORE_KEY="vstar_overall_acc.score"

# Allow overriding via CLI args
PATH1="${1:-$PATH1}"
PATH2="${2:-$PATH2}"
SCORE_KEY="${3:-$SCORE_KEY}"

echo "Left:  $PATH1"
echo "Right: $PATH2"
echo "Score: $SCORE_KEY"
echo "Starting diff viewer at http://localhost:5001"

python "$SCRIPT_DIR/app.py" --path1 "$PATH1" --path2 "$PATH2" --score_key "$SCORE_KEY"
