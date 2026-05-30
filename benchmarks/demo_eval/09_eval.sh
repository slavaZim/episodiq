#!/bin/bash
# Step 9: leakage-free eval on the held-out 220 trajectories using the
# peak-AUC config picked in step 08. Uses the same stratified ordering
# (output/stratified_order.json) that step 08 generated, so the tune
# corpus and eval queries stay on disjoint stratified slices.
set -euo pipefail
cd "$(dirname "$0")"

export EPISODIQ_MINHASH_K="${EPISODIQ_MINHASH_K:-512}"
export EPISODIQ_NGRAM_N="${EPISODIQ_NGRAM_N:-3}"

ENV=.env
ORDER=output/stratified_order.json
TUNE_CONFIG=output/tune_config.json
EVAL_REPORT=output/eval_report.json
TUNE_OFFSET=0
TUNE_LIMIT=55
EVAL_OFFSET=55
EVAL_LIMIT=220

if [ ! -f "$ORDER" ]; then
  echo "Error: $ORDER not found — run step 08 first (it generates the stratified ordering)."
  exit 1
fi

echo "=== Step 10: Eval (corpus=$TUNE_LIMIT tune trajs, queries=$EVAL_LIMIT eval trajs, stratified) ==="
PYTHONUNBUFFERED=1 uv run python eval_sweep.py \
  --env "$ENV" \
  --config "$TUNE_CONFIG" \
  --order-file "$ORDER" \
  --tune-offset "$TUNE_OFFSET" --tune-limit "$TUNE_LIMIT" \
  --eval-offset "$EVAL_OFFSET" --eval-limit "$EVAL_LIMIT" \
  --output "$EVAL_REPORT"

echo "=== Done: $EVAL_REPORT ==="
