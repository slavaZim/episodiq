#!/bin/bash
# Step 6: cluster AO tokens using params from output/tokenize_config.json
# (manually saved from the grid search winner). Writes token_clusters +
# token_mapping (one extra row with cluster_id=-1 collapses HDBSCAN-noise
# pairs).
#
# tokenize_config.json shape:
#   {"min_cluster_size": 10, "min_samples": 5,
#    "umap_dims": 50, "umap_n_neighbors": 15,
#    "cluster_selection_method": "eom", "cluster_selection_epsilon": 0.0}
set -euo pipefail
cd "$(dirname "$0")"

CONFIG=output/tokenize_config.json
if [ ! -f "$CONFIG" ]; then
  echo "Error: $CONFIG not found."
  echo "Pick a winner from output/tokenize_grid.csv and save its params to $CONFIG."
  exit 1
fi

MCS=$(uv run python -c "import json; print(json.load(open('$CONFIG'))['min_cluster_size'])")
MS=$(uv run python -c "import json; print(json.load(open('$CONFIG'))['min_samples'])")
UD=$(uv run python -c "import json; print(json.load(open('$CONFIG'))['umap_dims'])")
UNN=$(uv run python -c "import json; print(json.load(open('$CONFIG'))['umap_n_neighbors'])")
SM=$(uv run python -c "import json; print(json.load(open('$CONFIG'))['cluster_selection_method'])")
SE=$(uv run python -c "import json; print(json.load(open('$CONFIG'))['cluster_selection_epsilon'])")

echo "=== Step 6: Tokenize (AO clusters) ==="
echo "  config: min_cs=$MCS min_s=$MS umap_d=$UD umap_nn=$UNN method=$SM eps=$SE"
PYTHONUNBUFFERED=1 uv run episodiq cluster tokenize \
  --env .env \
  --min-cs "$MCS" \
  --min-s "$MS" \
  --umap-dims "$UD" \
  --umap-nn "$UNN" \
  --selection-method "$SM" \
  --selection-epsilon "$SE"
echo "=== Done ==="
