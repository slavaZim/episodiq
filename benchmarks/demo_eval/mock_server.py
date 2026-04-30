"""Mock LLM server that replays SWE-smith agent responses.

Loads trajectories from HuggingFace, indexes assistant messages by
(dataset_traj_id, turn_number). Routing: request body model field
encodes "mock::<dataset_traj_id>", turn derived from assistant message
count in the request.

Usage:
    uv run python mock_server.py [--port 9999] [--repo-filter getmoto__moto]
"""

import argparse
import json
import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Mock LLM Server")

# Global index: (dataset_traj_id, turn) → assistant message dict
_responses: dict[tuple[str, int], dict] = {}
# Set of known traj_ids for validation
_known_traj_ids: set[str] = set()
_ready = False


def _sanitize(s: str) -> str:
    """Remove null bytes that PostgreSQL cannot store."""
    return s.replace("\u0000", "")


def _load_dataset(repo_filter: str = "getmoto__moto") -> int:
    """Load SWE-smith trajectories and build response index."""
    from datasets import load_dataset

    logger.info("Loading SWE-bench/SWE-smith-trajectories split=tool ...")
    ds = load_dataset("SWE-bench/SWE-smith-trajectories", split="tool")
    logger.info("Loaded %d trajectories total", len(ds))

    rows = [r for r in ds if repo_filter in r["instance_id"]]
    logger.info("Filtered to %d trajectories for repo=%s", len(rows), repo_filter)

    for row in rows:
        traj_id = row["traj_id"]
        _known_traj_ids.add(traj_id)

        messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
        turn = 0
        for msg in messages:
            if msg.get("role") == "assistant":
                _responses[(traj_id, turn)] = msg
                turn += 1

    logger.info("Indexed %d assistant responses from %d trajectories", len(_responses), len(rows))
    return len(rows)


@app.get("/health")
async def health():
    if not _ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ready", "trajectories": len(_known_traj_ids), "responses": len(_responses)}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body = await request.json()

    # Extract routing: model = "mock::<dataset_traj_id>"
    model = body.get("model", "")
    if not model.startswith("mock::"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Expected model='mock::<traj_id>', got '{model}'"},
        )

    dataset_traj_id = model[len("mock::"):]

    if dataset_traj_id not in _known_traj_ids:
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown traj_id: {dataset_traj_id}"},
        )

    # Determine turn from number of assistant messages in request
    messages = body.get("messages", [])
    turn = sum(1 for m in messages if m.get("role") == "assistant")

    response_msg = _responses.get((dataset_traj_id, turn))
    if response_msg is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"No response for traj={dataset_traj_id} turn={turn}"},
        )

    # Build OpenAI-compatible response
    choice_message: dict = {"role": "assistant"}

    content = response_msg.get("content")
    if content:
        choice_message["content"] = _sanitize(content) if isinstance(content, str) else content

    tool_calls = response_msg.get("tool_calls")
    if tool_calls:
        # Ensure each tool_call has an id
        for i, tc in enumerate(tool_calls):
            if not tc.get("id"):
                tc["id"] = f"call_{dataset_traj_id[-8:]}_{turn}_{i}"
        choice_message["tool_calls"] = tool_calls

    return JSONResponse(content={
        "id": f"mock-{dataset_traj_id}-{turn}",
        "object": "chat.completion",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": choice_message,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    })


def main():
    parser = argparse.ArgumentParser(description="Mock LLM server for SWE-smith replay")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--repo-filter", type=str, default="getmoto__moto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    n = _load_dataset(repo_filter=args.repo_filter)
    if n == 0:
        logger.error("No trajectories found for filter '%s'", args.repo_filter)
        raise SystemExit(1)

    global _ready
    _ready = True
    logger.info("Mock server ready on port %d", args.port)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
