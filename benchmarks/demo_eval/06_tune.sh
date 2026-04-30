#!/bin/bash
# Step 6: Tune prefetch/topk — update .env before running 06b_tune_signals.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 6: Tune prefetch-topk ==="
PYTHONUNBUFFERED=1 uv run episodiq tune prefetch-topk --env .env 2>&1 | tee output/tune_prefetch.txt

echo ""
SUGGESTION=$(grep "^Suggested:" output/tune_prefetch.txt | tail -1)
echo ">>> Update .env with: ${SUGGESTION}"
echo ">>> Then run 06b_tune_signals.sh"
echo "=== Done ==="
