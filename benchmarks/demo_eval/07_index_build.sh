#!/bin/bash
# Step 7: build per-window MinHash LSH bands for every trajectory_path
# from token_mapping. Required before tune/eval (both read trace_tokens
# and rebuild alt LSH tables from them).
#
# Bench overrides the production band layout because the 275-traj
# demo corpus is too thin for the narrower default (B=32, R=2,
# threshold ≈ 0.18). A wider layout (B=64, R=1, threshold ≈ 0.015)
# surfaces enough candidates to keep eval AUC above the noise floor.
set -euo pipefail
cd "$(dirname "$0")"

export EPISODIQ_WMH_SIG_SIZE="${EPISODIQ_WMH_SIG_SIZE:-64}"
export EPISODIQ_WMH_NUM_BANDS="${EPISODIQ_WMH_NUM_BANDS:-64}"
export EPISODIQ_RETRIEVAL_WINDOW="${EPISODIQ_RETRIEVAL_WINDOW:-10}"

echo "=== Step 7: Index build (W=$EPISODIQ_RETRIEVAL_WINDOW, sig=$EPISODIQ_WMH_SIG_SIZE, bands=$EPISODIQ_WMH_NUM_BANDS) ==="
PYTHONUNBUFFERED=1 uv run episodiq index build --env .env
echo "=== Done ==="
