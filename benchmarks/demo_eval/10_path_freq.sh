#!/bin/bash
# Step 10 (optional): suggest action-variance thresholds
# (EPISODIQ_LOW_ENTROPY / EPISODIQ_HIGH_ENTROPY) from the path-frequency
# entropy distribution. Persists the suggestion to
# output/path_freq_config.json so step 11 can export it — otherwise the
# demo would read stale .env values and every action ends up tagged
# "high".
#
# Exports the cascade retrieval config picked by step 08 so entropy
# thresholds match what step 11's report will see.
set -euo pipefail
cd "$(dirname "$0")"

TUNE_CONFIG=output/tune_config.json
PATH_FREQ_CONFIG=output/path_freq_config.json

if [ ! -f "$TUNE_CONFIG" ]; then
  echo "Error: $TUNE_CONFIG not found — run step 08 first."
  exit 1
fi

# Pull cascade params from the tune winner.
export EPISODIQ_RETRIEVAL_WINDOW=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['window'])")
export EPISODIQ_CASCADE_AGGREGATION=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['aggregation'])")
export EPISODIQ_CASCADE_PREFETCH_N_UNIQ=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['prefetch_n_uniq'])")
export EPISODIQ_CASCADE_JACCARD_N_UNIQ=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['jaccard_n_uniq'])")
export EPISODIQ_CASCADE_TOP_K=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['top_k'])")
export EPISODIQ_AS_LAM=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['lam'])")
export EPISODIQ_AS_PENALTY_SHAPE=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['penalty_shape'])")
export EPISODIQ_AS_GAP_OPEN=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['gap_open'])")
export EPISODIQ_AS_GAP_EXTEND=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['gap_extend'])")
export EPISODIQ_AS_SIGMA=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['sigma'])")
# LSH layout — bench uses the wider 64-band variant.
export EPISODIQ_WMH_SIG_SIZE="${EPISODIQ_WMH_SIG_SIZE:-64}"
export EPISODIQ_WMH_NUM_BANDS="${EPISODIQ_WMH_NUM_BANDS:-64}"

echo "=== Step 10: Tune path-frequency thresholds (W=$EPISODIQ_RETRIEVAL_WINDOW agg=$EPISODIQ_CASCADE_AGGREGATION top_k=$EPISODIQ_CASCADE_TOP_K) ==="

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
