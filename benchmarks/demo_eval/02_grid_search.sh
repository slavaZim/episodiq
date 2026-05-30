#!/bin/bash
# Step 2: Grid search clustering parameters → output/grid_search.csv
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 2: Grid search clustering ==="

PYTHONUNBUFFERED=1 uv run episodiq cluster grid-search --env .env \
  --save-output output/grid_search.csv

echo ""
echo ">>> Review output/grid_search.csv and save chosen params to output/cluster_config.json"
echo '>>> Example: {"min_cluster_size": 10, "min_samples": 5, "umap_dims": 30, "umap_n_neighbors": 15, "selection_method": "eom", "selection_epsilon": 0.0}'
echo "=== Done ==="
