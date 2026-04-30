#!/bin/bash
# Step 1: Seed 500 training trajectories via mock server + proxy
set -euo pipefail
cd "$(dirname "$0")"

ENV=.env
PROXY_PORT=8081
MOCK_PORT=9999

# Kill a process and all its children
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

echo "=== Step 1: Seed training trajectories ==="

# 1. Create database + run migrations
echo "--- Setting up database ---"
uv run episodiq db init --env "$ENV"

# 2. Start mock server
echo "--- Starting mock server on port $MOCK_PORT ---"
PYTHONUNBUFFERED=1 uv run python mock_server.py --port "$MOCK_PORT" &
MOCK_PID=$!
trap "kill_tree $MOCK_PID; wait $MOCK_PID 2>/dev/null" EXIT
wait_for_url "http://localhost:$MOCK_PORT/health" "mock server" 120

# 3. Start proxy
echo "--- Starting proxy on port $PROXY_PORT ---"
uv run episodiq up --env "$ENV" --port "$PROXY_PORT" &
PROXY_PID=$!
trap "kill_tree $MOCK_PID; kill_tree $PROXY_PID; wait $MOCK_PID $PROXY_PID 2>/dev/null" EXIT
wait_for_url "http://localhost:$PROXY_PORT/episodiq/health" "proxy" 30

# 4. Seed train set (500 trajectories)
echo "--- Seeding 500 train trajectories ---"
PYTHONUNBUFFERED=1 uv run python seed_via_proxy.py \
  --phase train \
  --proxy-url "http://localhost:$PROXY_PORT" \
  --concurrency 2 \
  --output output/train_traj_ids.json

echo "--- Waiting for all embeddings to complete ---"
PSQL_URL=$(echo "$EPISODIQ_DATABASE_URL" | sed 's|postgresql+asyncpg://|postgresql://|')
source "$ENV"
TOTAL=$(psql "$PSQL_URL" -t -A -c "SELECT COUNT(*) FROM messages")
for i in $(seq 1 120); do
  EMBEDDED=$(psql "$PSQL_URL" -t -A -c "SELECT COUNT(*) FROM messages WHERE embedding IS NOT NULL")
  echo "  [$i] Embeddings: $EMBEDDED / $TOTAL"
  if [ "$EMBEDDED" -ge "$TOTAL" ]; then
    echo "  All embeddings complete"
    break
  fi
  if [ "$i" -eq 120 ]; then
    echo "  WARNING: timed out after 600s, $((TOTAL - EMBEDDED)) messages without embeddings"
  fi
  sleep 5
done

echo "=== Done: output/train_traj_ids.json ==="
kill_tree $MOCK_PID; kill_tree $PROXY_PID
wait $MOCK_PID $PROXY_PID 2>/dev/null || true
