#!/bin/bash
# Step 1 (nebius/sqlglot): seed all 275 instance-disjoint trajectories.
# No train/eval split -- retrieval-engine framing uses one pool.
set -euo pipefail
cd "$(dirname "$0")"

ENV=.env
PROXY_PORT=8081
MOCK_PORT=9999
REPO=tobymao/sqlglot
DATASET=nebius
TRAIN_LIMIT=10000   # take all instances (sqlglot has 275)

kill_tree() {
  local pid="$1"
  pkill -P "$pid" 2>/dev/null || true
  kill "$pid" 2>/dev/null || true
}

wait_for_url() {
  local url="$1" name="$2" max_wait="${3:-120}"
  echo "  Waiting for $name ($url) ..."
  for i in $(seq 1 "$max_wait"); do
    if curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -q "^[23]"; then
      echo "  $name ready (${i}s)"
      return 0
    fi
    sleep 1
  done
  echo "  ERROR: $name not ready after ${max_wait}s"
  return 1
}

echo "=== Step 1 (sqlglot): seed all instance-disjoint trajectories ==="

mkdir -p output

echo "--- DB init ---"
uv run episodiq db init --env "$ENV"

echo "--- Mock server on $MOCK_PORT (dataset=$DATASET, repo=$REPO) ---"
PYTHONUNBUFFERED=1 uv run python mock_server.py \
  --port "$MOCK_PORT" --dataset "$DATASET" --repo-filter "$REPO" &
MOCK_PID=$!
trap "kill_tree $MOCK_PID; wait $MOCK_PID 2>/dev/null" EXIT
wait_for_url "http://localhost:$MOCK_PORT/health" "mock server" 180

echo "--- Proxy on $PROXY_PORT ---"
uv run episodiq up --env "$ENV" --port "$PROXY_PORT" &
PROXY_PID=$!
trap "kill_tree $MOCK_PID; kill_tree $PROXY_PID; wait $MOCK_PID $PROXY_PID 2>/dev/null" EXIT
wait_for_url "http://localhost:$PROXY_PORT/episodiq/health" "proxy" 30

echo "--- Seeding all $REPO instance-disjoint trajectories ---"
PYTHONUNBUFFERED=1 uv run python seed_via_proxy.py \
  --phase train \
  --dataset "$DATASET" \
  --repo-filter "$REPO" \
  --train-limit "$TRAIN_LIMIT" \
  --eval-limit 0 \
  --proxy-url "http://localhost:$PROXY_PORT" \
  --concurrency 4 \
  --output output/sqlglot_traj_ids.json

echo "--- Waiting for all embeddings to complete ---"
PSQL_URL=$(grep '^EPISODIQ_DATABASE_URL=' "$ENV" | cut -d= -f2- | sed 's|postgresql+asyncpg://|postgresql://|')
TOTAL=$(PGPASSWORD=optimaizr psql "$PSQL_URL" -t -A -c "SELECT COUNT(*) FROM messages")
for i in $(seq 1 240); do
  EMBEDDED=$(PGPASSWORD=optimaizr psql "$PSQL_URL" -t -A -c "SELECT COUNT(*) FROM messages WHERE embedding IS NOT NULL")
  echo "  [$i] Embeddings: $EMBEDDED / $TOTAL"
  if [ "$EMBEDDED" -ge "$TOTAL" ]; then
    echo "  All embeddings complete"
    break
  fi
  sleep 5
done

echo "=== Done: output/sqlglot_traj_ids.json ==="
kill_tree $MOCK_PID; kill_tree $PROXY_PID
wait $MOCK_PID $PROXY_PID 2>/dev/null || true
