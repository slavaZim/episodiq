#!/bin/bash
# Step 7: Train dead-end model (full train set, no test split)
set -euo pipefail
cd "$(dirname "$0")"

DEAD_END_MODEL=output/dead_end_model.joblib

echo "=== Step 7: Train dead-end model ==="

PYTHONUNBUFFERED=1 uv run episodiq dead-end train --env .env \
  --output "$DEAD_END_MODEL" \
  --test-size 0

echo "=== Done: $DEAD_END_MODEL ==="
