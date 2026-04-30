#!/bin/bash
# Step 4: Build trajectory paths (without signals — tune first)
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 4: Build paths ==="

PYTHONUNBUFFERED=1 uv run episodiq cluster build-paths --env .env

echo "=== Done ==="
