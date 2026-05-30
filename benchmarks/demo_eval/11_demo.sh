#!/bin/bash
# Step 11: Demo — render every seeded trajectory with analytics (-a)
# enabled and dump as JSONL. Each record carries `instance_id` and
# `dataset_traj_id` from the seed mapping so reports map back to the
# source SWE-rebench instance.
#
# Pulls retrieval params (top_k, similarity_threshold) from
# output/tune_config.json and action-variance thresholds (LOW/HIGH_ENTROPY)
# from output/path_freq_config.json — without the latter the report falls
# back to stale .env thresholds and the action_variance buckets degenerate
# (e.g. everything tagged "high").
set -euo pipefail
cd "$(dirname "$0")"

ENV=.env
TRAJ_IDS=output/sqlglot_traj_ids.json
TUNE_CONFIG=output/tune_config.json
PATH_FREQ_CONFIG=output/path_freq_config.json
OUT=output/demo_reports.jsonl

mkdir -p output

if [ ! -f "$TUNE_CONFIG" ]; then
  echo "Error: $TUNE_CONFIG not found — run step 08 first."
  exit 1
fi
if [ ! -f "$PATH_FREQ_CONFIG" ]; then
  echo "Error: $PATH_FREQ_CONFIG not found — run step 10 first."
  exit 1
fi

export EPISODIQ_MINHASH_K="${EPISODIQ_MINHASH_K:-512}"
export EPISODIQ_NGRAM_N="${EPISODIQ_NGRAM_N:-3}"
export EPISODIQ_RETRIEVAL_TOP_K=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['top_k'])")
export EPISODIQ_RETRIEVAL_SIMILARITY_THRESHOLD=$(uv run python -c "import json; print(json.load(open('$TUNE_CONFIG'))['similarity_threshold'])")
export EPISODIQ_LOW_ENTROPY=$(uv run python -c "import json; print(json.load(open('$PATH_FREQ_CONFIG'))['low_entropy'])")
export EPISODIQ_HIGH_ENTROPY=$(uv run python -c "import json; print(json.load(open('$PATH_FREQ_CONFIG'))['high_entropy'])")

echo "=== Step 11: Demo — render all trajectories with -a ==="

PYTHONUNBUFFERED=1 uv run python - "$TRAJ_IDS" "$ENV" "$OUT" <<'PY'
import json, subprocess, sys
from pathlib import Path

ids_path, env, out_path = sys.argv[1:]
if not Path(ids_path).exists():
    sys.exit(f"seed mapping not found at {ids_path} — run step 01 first")

with open(ids_path) as f:
    mapping = json.load(f)

total = len(mapping)
print(f"  {total} trajectories to render")
if total == 0:
    sys.exit("seed mapping is empty")

written, skipped = 0, 0
with open(out_path, "w") as out:
    for i, (uuid, meta) in enumerate(mapping.items(), 1):
        proc = subprocess.run(
            ["uv", "run", "episodiq", "report", uuid,
             "--env", env, "--format", "json", "-a"],
            capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0:
            skipped += 1
            continue
        for line in proc.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["instance_id"] = meta.get("instance_id")
            rec["dataset_traj_id"] = meta.get("traj_id")
            out.write(json.dumps(rec) + "\n")
            written += 1
        if i % 25 == 0:
            print(f"  {i}/{total} trajectories, {written} entries, {skipped} skipped")

print(f"done: {written} entries, {skipped} trajectories skipped")
PY

echo ""
echo "=== Done: $OUT ==="
