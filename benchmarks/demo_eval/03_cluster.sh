#!/bin/bash
# Step 3: Run clustering with manually chosen params per type/category.
# Config format: list of {type, category, min_cluster_size, min_samples, ...}
set -euo pipefail
cd "$(dirname "$0")"

CONFIG=output/cluster_config.json

if [ ! -f "$CONFIG" ]; then
  echo "Error: $CONFIG not found."
  echo "Create it manually after reviewing grid search results."
  echo 'Example:'
  echo '[
  {"type": "observation", "category": "text", "min_cluster_size": 10, "min_samples": 5, "umap_dims": 30, "umap_n_neighbors": 15},
  {"type": "observation", "category": "tool", "min_cluster_size": 15, "min_samples": 5, "umap_dims": 30, "umap_n_neighbors": 15},
  {"type": "action", "category": "text", "min_cluster_size": 10, "min_samples": 5, "umap_dims": 30, "umap_n_neighbors": 15},
  {"type": "action", "category": "tool", "min_cluster_size": 15, "min_samples": 5, "umap_dims": 30, "umap_n_neighbors": 15}
]'
  exit 1
fi

echo "=== Step 3: Run clustering ==="

python3 -c "
import json, subprocess, sys

entries = json.load(open('$CONFIG'))
for e in entries:
    cmd = ['uv', 'run', 'episodiq', 'cluster', 'run', '--env', '.env']
    cmd += ['-t', e['type'], '-c', e['category']]
    if 'min_cluster_size' in e: cmd += ['--min-cs', str(e['min_cluster_size'])]
    if 'min_samples' in e: cmd += ['--min-s', str(e['min_samples'])]
    if 'umap_dims' in e: cmd += ['--umap-dims', str(e['umap_dims'])]
    if 'umap_n_neighbors' in e: cmd += ['--umap-nn', str(e['umap_n_neighbors'])]
    if 'selection_method' in e: cmd += ['--selection-method', e['selection_method']]
    if 'selection_epsilon' in e: cmd += ['--selection-epsilon', str(e['selection_epsilon'])]
    print(f\"--- {e['type']}:{e['category']} ---\")
    print(' '.join(cmd))
    subprocess.check_call(cmd)
"

echo "=== Done ==="
