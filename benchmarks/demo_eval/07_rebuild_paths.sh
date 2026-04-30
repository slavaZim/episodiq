#!/bin/bash
# Step 6: Rebuild paths with --fill-signals, then compute signal AUC
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 6: Rebuild paths with signals ==="

PYTHONUNBUFFERED=1 uv run episodiq cluster build-paths --env .env --fill-signals

echo ""
echo "--- Signal AUC ---"
PYTHONUNBUFFERED=1 uv run python signal_auc.py --env .env 2>&1 | tee output/signal_auc.txt

echo "=== Done ==="
