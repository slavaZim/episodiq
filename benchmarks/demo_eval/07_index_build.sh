#!/bin/bash
# Step 7: build trace_tokens + minhash_sig columns for every trajectory_path
# from the token_mapping table. Required before tune and eval (both read
# these columns).
#
# Both EPISODIQ_MINHASH_K and EPISODIQ_NGRAM_N are exported here so the
# pipeline reproduces without requiring users to manually pre-load the .env
# (which is gitignored). The .env values, if set, take precedence.
set -euo pipefail
cd "$(dirname "$0")"

export EPISODIQ_MINHASH_K="${EPISODIQ_MINHASH_K:-512}"
export EPISODIQ_NGRAM_N="${EPISODIQ_NGRAM_N:-3}"

echo "=== Step 7: Index build (trace_tokens + minhash_sig, K=$EPISODIQ_MINHASH_K, n=$EPISODIQ_NGRAM_N) ==="
PYTHONUNBUFFERED=1 uv run episodiq index build --env .env
echo "=== Done ==="
