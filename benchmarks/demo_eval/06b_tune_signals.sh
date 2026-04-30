#!/bin/bash
# Step 6b: Tune signal-thresholds and path-freq (uses PREFETCH_N/TOP_K from .env)
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 6b: Tune signal-thresholds ==="
PYTHONUNBUFFERED=1 uv run episodiq tune signal-thresholds --env .env --sample 500 2>&1 | tee output/tune_signals.txt

echo ""
echo "=== Step 6b: Tune path-freq ==="
PYTHONUNBUFFERED=1 uv run episodiq tune path-freq --env .env 2>&1 | tee output/tune_pathfreq.txt

echo ""
echo ">>> Update .env with suggestions from output/tune_signals.txt and output/tune_pathfreq.txt"
echo "=== Done ==="
