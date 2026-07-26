#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-.}"
CHECKPOINT_DIR="${2:?checkpoint directory required}"
OUTPUT_DIR="${3:?output directory required}"
EPISODES="${4:-100}"
SEED="${5:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

for checkpoint in "$CHECKPOINT_DIR"/*.pt; do
    name="$(basename "$checkpoint" .pt)"
    output="$OUTPUT_DIR/${name}_phase_${EPISODES}_seed${SEED}.json"
    echo "DIAGNOSE $checkpoint"
    "$PYTHON_BIN" scripts/diagnose_stack_cube_checkpoint.py \
        --checkpoint "$checkpoint" \
        --episodes "$EPISODES" \
        --seed "$SEED" \
        --output-json "$output"
done

echo "SWEEP_COMPLETE $OUTPUT_DIR"
