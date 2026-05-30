#!/bin/bash
# Step 5: AO tokenizer grid search (HDBSCAN+UMAP params on the act_obs
# concat embeddings). Writes a CSV ranking; the winner row is picked
# automatically by step 06.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 5: Tokenizer grid search ==="
PYTHONUNBUFFERED=1 uv run episodiq cluster tokenize-grid \
  --env .env \
  --save-output output/tokenize_grid.csv
echo "=== Done: output/tokenize_grid.csv ==="
