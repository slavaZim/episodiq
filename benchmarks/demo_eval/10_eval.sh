#!/bin/bash
# Step 9: Generate per-trajectory reports + compute aggregate metrics
set -euo pipefail
cd "$(dirname "$0")"

TRAJ_IDS=output/eval_traj_ids.json
REPORTS=output/reports.jsonl
DEAD_END_MODEL=output/dead_end_model.joblib

if [ ! -f "$TRAJ_IDS" ]; then
  echo "Error: $TRAJ_IDS not found. Run 08_seed_eval.sh first."
  exit 1
fi

echo "=== Step 9: Generate reports + eval metrics ==="

# 1. Generate all reports into a single JSONL file
echo "--- Generating reports ---"
> "$REPORTS"  # truncate
N=0
TOTAL=$(python3 -c "import json; print(len(json.load(open('$TRAJ_IDS'))))")

for tid in $(python3 -c "import json; [print(k) for k in json.load(open('$TRAJ_IDS'))]"); do
  N=$((N + 1))
  EPISODIQ_DEAD_END_MODEL="$DEAD_END_MODEL" episodiq report "$tid" --env .env --format json >> "$REPORTS" 2>/dev/null || true
  if [ $((N % 25)) -eq 0 ]; then
    echo "  $N / $TOTAL reports generated"
  fi
done
echo "  $N / $TOTAL reports generated"

# 2. Compute aggregate metrics
echo ""
echo "--- Computing metrics ---"
PYTHONUNBUFFERED=1 uv run python eval_metrics.py \
  --traj-ids "$TRAJ_IDS" \
  --reports "$REPORTS" \
  --output output/eval_summary.json

echo "=== Done: output/eval_summary.json ==="
