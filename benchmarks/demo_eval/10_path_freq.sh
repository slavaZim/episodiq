#!/bin/bash
# Step 10: Suggest action-variance thresholds (EPISODIQ_LOW_ENTROPY /
# EPISODIQ_HIGH_ENTROPY) from the path-frequency entropy distribution.
# Persists the suggestion to output/path_freq_config.json so step 11 can
# export it — otherwise the demo would read stale .env values and every
# action ends up tagged "high".
#
# Uses the SAME retrieval config (top_k + similarity_threshold) that step
# 08 picked, so entropy thresholds calibrated here match what step 11's
# report will actually see in production.
set -euo pipefail
cd "$(dirname "$0")"

TUNE_CONFIG=output/tune_config.json
PATH_FREQ_CONFIG=output/path_freq_config.json

if [ ! -f "$TUNE_CONFIG" ]; then
  echo "Error: $TUNE_CONFIG not found — run step 08 first."
  exit 1
fi

export EPISODIQ_MINHASH_K="${EPISODIQ_MINHASH_K:-512}"
export EPISODIQ_NGRAM_N="${EPISODIQ_NGRAM_N:-3}"
export EPISODIQ_RETRIEVAL_TOP_K=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['top_k'])")
export EPISODIQ_RETRIEVAL_SIMILARITY_THRESHOLD=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['similarity_threshold'])")

echo "=== Step 10: Tune path-frequency thresholds (top_k=$EPISODIQ_RETRIEVAL_TOP_K sim=$EPISODIQ_RETRIEVAL_SIMILARITY_THRESHOLD) ==="

TMP_OUT=$(mktemp)
trap 'rm -f "$TMP_OUT"' EXIT
PYTHONUNBUFFERED=1 uv run episodiq tune path-freq --env .env | tee "$TMP_OUT"

LOW=$(grep -oE 'EPISODIQ_LOW_ENTROPY=[0-9.]+' "$TMP_OUT" | tail -1 | cut -d= -f2)
HIGH=$(grep -oE 'EPISODIQ_HIGH_ENTROPY=[0-9.]+' "$TMP_OUT" | tail -1 | cut -d= -f2)
if [ -z "$LOW" ] || [ -z "$HIGH" ]; then
  echo "Error: could not parse suggested thresholds from path-freq output"
  exit 1
fi
uv run python -c "import json; json.dump({'low_entropy': float('$LOW'), 'high_entropy': float('$HIGH')}, open('$PATH_FREQ_CONFIG','w'), indent=2)"
echo ""
echo "  saved -> $PATH_FREQ_CONFIG (low=$LOW high=$HIGH)"

echo "=== Done ==="
