#!/bin/bash
# Step 4: Annotate clusters via LLM, then rebuild trajectory_paths.
# Annotation may merge near-duplicate clusters → paths must be rebuilt so
# their downstream cluster_id references match the merged state.
#
# Annotation is OPTIONAL — it only adds human-readable labels and merges
# near-duplicate clusters. If you skip it (no LLM key / cost reasons),
# you still need `episodiq cluster build-paths` to materialize the paths
# from the raw cluster_ids before tokenize/index can run.
#
# Annotator:  claude-sonnet-4-5  (contrastive cluster labeling)
# Summarizer: claude-haiku-4-5   (map-reduce for long messages)
# Both via OpenRouter. NOTE: overrides EPISODIQ_OPENAI_BASE_URL/KEY
# from .env (those point at the local mock for the proxy adapter).
set -euo pipefail
cd "$(dirname "$0")"

source .env  # EPISODIQ_EMBEDDER_URL / EPISODIQ_EMBEDDER_API_KEY / EPISODIQ_DATABASE_URL

PSQL_URL=$(echo "$EPISODIQ_DATABASE_URL" | sed 's|postgresql+asyncpg://|postgresql://|')
ANNOTATE_URL="${EPISODIQ_EMBEDDER_URL%/}/v1"
ANNOTATE_KEY="${EPISODIQ_EMBEDDER_API_KEY}"
ANNOTATE_MODEL="anthropic/claude-sonnet-4-5"
SUMMARIZER_MODEL="anthropic/claude-haiku-4-5"
NAIVE_SAMPLE=30

mkdir -p output

echo "=== Step 4a: Annotate clusters ==="
echo "  Annotator:  $ANNOTATE_MODEL (OpenRouter)"
echo "  Summarizer: $SUMMARIZER_MODEL (OpenRouter)"
echo ""

ANNOT_TMPFILE=$(mktemp)
EPISODIQ_OPENAI_BASE_URL="$ANNOTATE_URL" \
EPISODIQ_OPENAI_API_KEY="$ANNOTATE_KEY" \
PYTHONUNBUFFERED=1 uv run episodiq annotate --env .env \
  --adapter openai \
  --annotate-model "$ANNOTATE_MODEL" \
  --summarizer-model "$SUMMARIZER_MODEL" \
  --workers 20 \
  2>&1 | tee "$ANNOT_TMPFILE"

echo ""
echo "--- Token efficiency (Episodiq vs naive per-message) ---"
echo "  Sampling $NAIVE_SAMPLE messages for naive cost estimate..."

EPISODIQ_OPENAI_BASE_URL="$ANNOTATE_URL" \
EPISODIQ_OPENAI_API_KEY="$ANNOTATE_KEY" \
PSQL_URL="$PSQL_URL" \
ANNOTATE_MODEL="$ANNOTATE_MODEL" \
SUMMARIZER_MODEL="$SUMMARIZER_MODEL" \
NAIVE_SAMPLE="$NAIVE_SAMPLE" \
ANNOT_OUTPUT_FILE="$ANNOT_TMPFILE" \
OUTPUT_FILE="output/annotate_tokens.txt" \
PYTHONUNBUFFERED=1 uv run python token_efficiency.py

rm -f "$ANNOT_TMPFILE"

echo ""
echo "=== Step 4b: Rebuild trajectory_paths ==="
echo "  Annotation may merge clusters → rebuild so paths reference merged ids."

PYTHONUNBUFFERED=1 uv run episodiq cluster build-paths --env .env

echo ""
echo "=== Done: output/annotate_tokens.txt ==="
