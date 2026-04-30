#!/bin/bash
# Step 4: Annotate clusters with LLM-generated labels
# Annotator:  claude-sonnet-4-5 via OpenRouter (contrastive annotation)
# Summarizer: claude-haiku-4-5  via OpenRouter (map-reduce for long messages)
# NOTE: overrides OPENAI_BASE_URL/KEY from .env (those are mock-server values for proxy)
set -euo pipefail
cd "$(dirname "$0")"

source .env  # loads EPISODIQ_EMBEDDER_URL / EPISODIQ_EMBEDDER_API_KEY / EPISODIQ_DATABASE_URL

PSQL_URL=$(echo "$EPISODIQ_DATABASE_URL" | sed 's|postgresql+asyncpg://|postgresql://|')
ANNOTATE_URL="${EPISODIQ_EMBEDDER_URL%/}/v1"
ANNOTATE_KEY="${EPISODIQ_EMBEDDER_API_KEY}"
ANNOTATE_MODEL="anthropic/claude-sonnet-4-5"
SUMMARIZER_MODEL="anthropic/claude-haiku-4-5"
NAIVE_SAMPLE=30

echo "=== Step 4: Annotate clusters ==="
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
ANNOT_OUTPUT=$(cat "$ANNOT_TMPFILE")

# --- Token comparison: Episodiq vs naive per-message ---
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
echo "=== Done ==="
