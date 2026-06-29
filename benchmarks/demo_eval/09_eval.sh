#!/bin/bash
# Step 9: evaluate the tune-winner cascade config from step 8 on the
# held-out eval slice. ``eval_cascade.py`` reapplies the same
# ``Random(shuffle_seed).shuffle`` and takes trajectories at indices
# ``[tune_limit:]`` as eval queries. Corpus = all completed
# trajectories (self-trajectory excluded server-side) — the split
# lives in hyperparameter selection, not the corpus.
set -euo pipefail
cd "$(dirname "$0")"

ENV=.env
TUNE_CONFIG=output/tune_config.json
EVAL_REPORT=output/eval_report.json
N_WORKERS=4
EVAL_MIN_STEP=50
N_BOOT=2000   # final run: tighter bootstrap CI on the headline AUC

if [ ! -f "$TUNE_CONFIG" ]; then
  echo "Error: $TUNE_CONFIG not found — run step 08 first."
  exit 1
fi

echo "=== Step 9: eval cascade on held-out slice ==="
PYTHONUNBUFFERED=1 uv run python eval_cascade.py \
  --env "$ENV" \
  --config "$TUNE_CONFIG" \
  --output "$EVAL_REPORT" \
  --n-workers "$N_WORKERS" \
  --eval-min-step "$EVAL_MIN_STEP" \
  --n-boot "$N_BOOT"

echo "=== Done: $EVAL_REPORT ==="
