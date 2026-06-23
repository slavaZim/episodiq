#!/bin/bash
# Step 8: retrieval-sweep over the cascade hyperparameters on a tune
# slice (first TUNE_LIMIT trajectories after a deterministic
# Random(SHUFFLE_SEED).shuffle of all completed trajectories). The
# same seed in basic.py reproduces a parallel tune/eval split on the
# naive baseline.
#
# Optuna TPE searches per (W, agg) over prefetch_n_uniq /
# jaccard_n_uniq / top_k / penalty_shape / lam / gap_open / gap_extend
# / sigma + metric-as-categorical. The overall best trial's params +
# AUC land in output/tune_config.json for step 9 to eval against the
# remainder of the shuffled list.
set -euo pipefail
cd "$(dirname "$0")"

ENV=.env
TRIALS_CSV=output/sweep_trials.csv
TUNE_CONFIG=output/tune_config.json
TUNE_LIMIT=100
TUNE_OFFSET=0
SHUFFLE_SEED=42
# Sort by Trajectory.meta.instance_id BEFORE the seed shuffle so the
# pre-shuffle order is dataset-derived and matches what basic.py sees
# when it sorts HF rows on the same key. Same (field, seed) on both
# sides ⇒ identical tune/eval slices.
SHUFFLE_FIELD=meta.instance_id
STRATIFY_FIELD=status

N_TRIALS=30
EARLY_STOP_PATIENCE=10
N_JOBS=4
N_WORKERS=4
W_GRID=10,14
AGG_GRID=mean,min_distance
OBJECTIVE_METRIC=cummean

mkdir -p output

echo "=== Step 8: retrieval-sweep (tune slice: $TUNE_LIMIT trajs, seed=$SHUFFLE_SEED) ==="
PYTHONUNBUFFERED=1 uv run episodiq tune retrieval-sweep \
  --env "$ENV" \
  --shuffle-seed "$SHUFFLE_SEED" \
  --shuffle-field "$SHUFFLE_FIELD" \
  --stratify-field "$STRATIFY_FIELD" \
  --limit "$TUNE_LIMIT" \
  --offset "$TUNE_OFFSET" \
  --w-grid "$W_GRID" \
  --agg-grid "$AGG_GRID" \
  --n-trials "$N_TRIALS" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --n-jobs "$N_JOBS" \
  --n-workers "$N_WORKERS" \
  --multivariate \
  --objective-metric "$OBJECTIVE_METRIC" \
  --save-trials "$TRIALS_CSV"

echo ""
echo "=== Step 8: pick champion (argmax target_auc across all trials) ==="
PYTHONUNBUFFERED=1 uv run python - "$TRIALS_CSV" "$TUNE_CONFIG" "$TUNE_LIMIT" "$SHUFFLE_SEED" "$SHUFFLE_FIELD" "$STRATIFY_FIELD" <<'EOF'
import csv, json, sys
from pathlib import Path

trials_path, cfg_path, tune_limit, seed, shuffle_field, stratify_field = sys.argv[1:7]
rows = list(csv.DictReader(open(trials_path)))
if not rows:
    raise SystemExit("no trials in CSV — sweep produced no signal")

best = max(rows, key=lambda r: float(r["target_auc"]))
cfg = {
    "window": int(best["window"]),
    "aggregation": best["aggregation"],
    "metric": best["target_metric"],
    "prefetch_n_uniq": int(best["prefetch_n_uniq"]),
    "jaccard_n_uniq": int(best["jaccard_n_uniq"]),
    "top_k": int(best["top_k"]),
    "penalty_shape": best["penalty_shape"],
    "lam": float(best["lam"]),
    "gap_open": float(best["gap_open"]),
    "gap_extend": float(best["gap_extend"]),
    "sigma": float(best["sigma"]),
    "tune_auc": float(best["target_auc"]),
    "tune_limit": int(tune_limit),
    "shuffle_seed": int(seed),
    "shuffle_field": shuffle_field or None,
    "stratify_field": stratify_field or None,
}
Path(cfg_path).write_text(json.dumps(cfg, indent=2))
print(
    f"  champion: W={cfg['window']} agg={cfg['aggregation']} "
    f"metric={cfg['metric']} AUC={cfg['tune_auc']:.4f}"
)
print(
    f"  params: prefetch={cfg['prefetch_n_uniq']} jaccard={cfg['jaccard_n_uniq']} "
    f"top_k={cfg['top_k']} penalty={cfg['penalty_shape']} lam={cfg['lam']} "
    f"sigma={cfg['sigma']} gap_open={cfg['gap_open']} gap_extend={cfg['gap_extend']}"
)
print(f"  wrote -> {cfg_path}")
EOF

echo "=== Done ==="
