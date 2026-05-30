#!/bin/bash
# Step 8: retrieval-sweep on the first 55 trajectories of a stratified
# ordering (balanced by status × length-quartile so the tune slice mirrors
# the full-population distribution). Leave-one-out against the rest of the
# completed trajectories. Picks the single config with the highest AUC@s60
# in the target coverage band and writes it to output/tune_config.json for
# step 10 to eval.
set -euo pipefail
cd "$(dirname "$0")"

export EPISODIQ_MINHASH_K="${EPISODIQ_MINHASH_K:-512}"
export EPISODIQ_NGRAM_N="${EPISODIQ_NGRAM_N:-3}"

ENV=.env
ORDER=output/stratified_order.json
SWEEP=output/sweep.csv
TUNE_CONFIG=output/tune_config.json
TUNE_LIMIT=55
TUNE_OFFSET=0
TOPK_GRID=5,10,15,25
SIM_GRID=0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50

echo "=== Step 8a: Compute stratified trajectory ordering ==="
if [ -f "$ORDER" ]; then
  echo "  using existing $ORDER (delete to regenerate)"
else
  PYTHONUNBUFFERED=1 uv run python stratify.py --env "$ENV" --output "$ORDER"
fi

echo ""
echo "=== Step 8b: Retrieval sweep on $TUNE_LIMIT tune queries (offset=$TUNE_OFFSET, stratified) ==="
PYTHONUNBUFFERED=1 uv run episodiq tune retrieval-sweep \
  --env "$ENV" \
  --order-file "$ORDER" \
  --limit "$TUNE_LIMIT" \
  --offset "$TUNE_OFFSET" \
  --top-k "$TOPK_GRID" \
  --sim "$SIM_GRID" \
  --save-output "$SWEEP"

TARGET_COV=0.60
COV_TOL=0.20

echo ""
echo "=== Step 8: Pick best AUC@s60 in coverage band $TARGET_COV +/- $COV_TOL ==="
# CSV header: top_k,similarity_threshold,coverage_step60,auc_step60_current,n_snapshots
PYTHONUNBUFFERED=1 uv run python - "$SWEEP" "$TUNE_CONFIG" "$TARGET_COV" "$COV_TOL" <<'EOF'
import csv, json, sys
from pathlib import Path

sweep_path, cfg_path = sys.argv[1], sys.argv[2]
target_cov, tol = float(sys.argv[3]), float(sys.argv[4])
rows = list(csv.DictReader(open(sweep_path)))

def auc(r):
    v = r.get("auc_step60_current") or ""
    return float(v) if v else -1.0

def cov(r):
    v = r.get("coverage_step60") or ""
    return float(v) if v else -1.0

band = [r for r in rows if abs(cov(r) - target_cov) <= tol]
if not band:
    print(f"  no sweep rows in coverage band {target_cov} +/- {tol}; "
          f"falling back to best AUC overall")
    band = rows
best = max(band, key=auc)
cfg = {
    "top_k": int(best["top_k"]),
    "similarity_threshold": float(best["similarity_threshold"]),
    "tune_coverage_step60": cov(best),
    "tune_auc_step60_current": auc(best),
}
Path(cfg_path).write_text(json.dumps(cfg, indent=2))
print(
    f"  top_k={cfg['top_k']} sim={cfg['similarity_threshold']:.3f} "
    f"cov@s60={cfg['tune_coverage_step60']:.3f} "
    f"AUC@s60={cfg['tune_auc_step60_current']:.4f}"
)
print(f"  picked from {len(band)} / {len(rows)} band rows -> {cfg_path}")
EOF

echo "=== Done ==="
