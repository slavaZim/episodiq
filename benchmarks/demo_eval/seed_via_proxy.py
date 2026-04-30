"""Replay SWE-smith trajectories through the Optimaizr proxy turn-by-turn.

For each trajectory:
  1. Generate a UUID
  2. Parse messages from dataset (JSON string)
  3. Replay conversation turn-by-turn, sending cumulative messages
  4. Mark trajectory status via PATCH /episodiq/trajectories/{uuid}
  5. Save trajectory mapping to output JSON

Usage:
    uv run python seed_via_proxy.py --phase train --output output/train_traj_ids.json
    uv run python seed_via_proxy.py --phase eval --output output/eval_traj_ids.json
"""

import argparse
import asyncio
import json
import logging
import uuid

import httpx

logger = logging.getLogger(__name__)

TRAIN_LIMIT = 500
EVAL_OFFSET = 500
EVAL_LIMIT = 250


def _sanitize(s: str) -> str:
    return s.replace("\u0000", "")


def _fix_tool_call_ids(messages: list[dict]) -> list[dict]:
    """Ensure all tool messages have tool_call_id and assistant tool_calls have id.

    SWE-smith dataset may have missing tool_call_id fields.
    Generate synthetic IDs where missing to satisfy OpenAI format.
    """
    call_counter = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if not tc.get("id"):
                    tc["id"] = f"call_{call_counter}"
                    call_counter += 1
        elif msg.get("role") == "tool" and not msg.get("tool_call_id"):
            # Find the last assistant tool_call without a matching tool response
            for prev in reversed(messages[:messages.index(msg)]):
                if prev.get("role") == "assistant" and prev.get("tool_calls"):
                    # Use the first tool_call id that hasn't been claimed yet
                    for tc in prev["tool_calls"]:
                        tc_id = tc.get("id", f"call_{call_counter}")
                        if not tc.get("id"):
                            tc["id"] = tc_id
                            call_counter += 1
                        msg["tool_call_id"] = tc_id
                        break
                    break
            if not msg.get("tool_call_id"):
                msg["tool_call_id"] = f"call_{call_counter}"
                call_counter += 1
    return messages


def _load_trajectories(phase: str, repo_filter: str) -> list[dict]:
    """Load and slice SWE-smith trajectories for the given phase."""
    from datasets import load_dataset

    logger.info("Loading SWE-bench/SWE-smith-trajectories split=tool ...")
    ds = load_dataset("SWE-bench/SWE-smith-trajectories", split="tool")

    rows = [r for r in ds if repo_filter in r["instance_id"]]
    logger.info("Filtered to %d trajectories for repo=%s", len(rows), repo_filter)

    if phase == "train":
        rows = rows[:TRAIN_LIMIT]
    elif phase == "eval":
        rows = rows[EVAL_OFFSET : EVAL_OFFSET + EVAL_LIMIT]
    else:
        raise ValueError(f"Unknown phase: {phase}")

    logger.info("Phase=%s: %d trajectories", phase, len(rows))
    return rows


def _parse_turns(messages: list[dict]) -> list[list[dict]]:
    """Split message list into turns: each turn ends with an assistant message.

    Returns list of cumulative message lists to send to the proxy.
    Each entry is messages_so_far up to and including the input for that turn
    (everything before the assistant response).
    """
    turns: list[list[dict]] = []
    messages_so_far: list[dict] = []

    for msg in messages:
        if msg["role"] == "assistant":
            # Current messages_so_far is the input for this turn
            if messages_so_far:
                turns.append(list(messages_so_far))
            # Add assistant to running state for next turn's context
            messages_so_far.append(msg)
        else:
            messages_so_far.append(msg)

    return turns


async def _replay_trajectory(
    row: dict,
    proxy_url: str,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> dict:
    """Replay a single trajectory through the proxy."""
    async with sem:
        traj_uuid = str(uuid.uuid4())
        dataset_traj_id = row["traj_id"]
        instance_id = row["instance_id"]
        resolved = row["resolved"]
        status = "success" if resolved else "failure"

        raw_messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
        raw_messages = _fix_tool_call_ids(raw_messages)

        turns = _parse_turns(raw_messages)
        if not turns:
            logger.warning("No turns for traj=%s, skipping", dataset_traj_id)
            return {
                "uuid": traj_uuid,
                "instance_id": instance_id,
                "traj_id": dataset_traj_id,
                "status": status,
                "turns": 0,
                "error": "no_turns",
            }

        # Replay each turn
        errors = 0
        for i, messages_input in enumerate(turns):
            body = {
                "model": f"mock::{dataset_traj_id}",
                "messages": messages_input,
            }
            try:
                resp = await client.post(
                    f"{proxy_url}/openai/v1/chat/completions",
                    json=body,
                    headers={
                        "X-Trajectory-ID": traj_uuid,
                        "Content-Type": "application/json",
                        "Authorization": "Bearer mock-key",
                    },
                    timeout=60.0,
                )
                if resp.status_code != 200:
                    errors += 1
                    if errors <= 3:
                        logger.warning(
                            "Turn %d/%d failed for %s: %d %s",
                            i + 1, len(turns), dataset_traj_id[:30],
                            resp.status_code, resp.text[:200],
                        )
            except httpx.TimeoutException:
                errors += 1
                logger.warning("Turn %d/%d timed out for %s", i + 1, len(turns), dataset_traj_id[:30])

        # Mark trajectory status — use internal_error if any turns failed
        patch_status = "internal_error" if errors > 0 else status
        try:
            resp = await client.patch(
                f"{proxy_url}/episodiq/trajectories/{traj_uuid}",
                json={"status": patch_status},
                timeout=10.0,
            )
            if resp.status_code not in (200, 409):
                logger.warning("Status mark failed for %s: %d", traj_uuid[:8], resp.status_code)
        except httpx.TimeoutException:
            logger.warning("Status mark timed out for %s", traj_uuid[:8])

        logger.info(
            "Replayed %s → %s (%d turns, %d errors) status=%s",
            dataset_traj_id[:30], traj_uuid[:8], len(turns), errors, patch_status,
        )

        return {
            "uuid": traj_uuid,
            "instance_id": instance_id,
            "traj_id": dataset_traj_id,
            "status": patch_status,
            "turns": len(turns),
        }


async def run(
    phase: str,
    proxy_url: str,
    concurrency: int,
    output_path: str,
    repo_filter: str,
):
    rows = _load_trajectories(phase, repo_filter)
    if not rows:
        logger.error("No trajectories to seed")
        return

    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        tasks = [_replay_trajectory(row, proxy_url, client, sem) for row in rows]
        results = await asyncio.gather(*tasks)

    # Save mapping
    mapping = {r["uuid"]: {k: v for k, v in r.items() if k != "uuid"} for r in results}
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    n_errors = sum(1 for r in results if r.get("error"))
    total_turns = sum(r.get("turns", 0) for r in results)
    logger.info(
        "Seeding complete: %d trajectories, %d total turns, %d errors. Saved to %s",
        len(results), total_turns, n_errors, output_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Seed trajectories via proxy")
    parser.add_argument("--phase", required=True, choices=["train", "eval"])
    parser.add_argument("--proxy-url", default="http://localhost:8081")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--repo-filter", default="getmoto__moto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    asyncio.run(run(
        phase=args.phase,
        proxy_url=args.proxy_url,
        concurrency=args.concurrency,
        output_path=args.output,
        repo_filter=args.repo_filter,
    ))


if __name__ == "__main__":
    main()
