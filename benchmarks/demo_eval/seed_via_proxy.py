"""Replay SWE-smith trajectories through Episodiq proxy turn-by-turn.

For each trajectory:
  1. Generate a UUID
  2. Parse messages from dataset (JSON string)
  3. Replay conversation turn-by-turn, sending cumulative messages
  4. Mark trajectory status via PATCH /episodiq/trajectories/{uuid}
  5. Save trajectory mapping to output JSON

Train/eval split is instance-disjoint to prevent the "same PR was already solved
by another agent" leakage that inflates message-level retrieval scores. See
README "Dataset" section for details.

Usage:
    uv run python seed_via_proxy.py --phase train --output output/train_traj_ids.json
    uv run python seed_via_proxy.py --phase eval --output output/eval_traj_ids.json
"""

import argparse
import asyncio
import json
import logging
import uuid
from collections import defaultdict

import httpx

logger = logging.getLogger(__name__)

# Symmetric 1-traj-per-instance split: both train and eval take the first
# instance-order occurrence of each instance_id, on disjoint instance sets.
# Eliminates the train-vs-eval asymmetry from multi-traj-per-instance training.
# getmoto/moto has 305 unique instances in SWE-smith -> 205 train / 100 eval.
TRAIN_INST_LIMIT = 205
EVAL_INST_LIMIT = 100


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


DATASETS = {
    # name -> (hf_id, split, repo-field, instance-field, traj-id-field, messages-field, messages-is-json-string)
    "swe-smith": (
        "SWE-bench/SWE-smith-trajectories", "tool",
        # repo filter matches on instance_id substring (e.g., "getmoto__moto")
        "instance_id", "instance_id", "traj_id", "messages", True,
    ),
    "nebius": (
        "nebius/SWE-rebench-openhands-trajectories", "train",
        # repo filter exact-matches the `repo` field (e.g., "tobymao/sqlglot")
        "repo", "instance_id", "trajectory_id", "trajectory", False,
    ),
}


def _load_trajectories(
    phase: str,
    repo_filter: str,
    dataset: str,
    train_limit: int | None = None,
    eval_limit: int | None = None,
) -> list[dict]:
    """Load trajectories for the given dataset, repo filter, and phase.

    1-traj-per-instance, stratified by outcome (split inherits dataset failure
    rate), instance-disjoint. Train gets the first ``train_limit`` instances
    of each outcome group, eval the next ``eval_limit``, in dataset order.

    Pass ``eval_limit=0`` to disable the eval split entirely (train takes the
    whole repo). When ``train_limit`` is ``None``, falls back to the module
    constant ``TRAIN_INST_LIMIT``.
    """
    from datasets import load_dataset

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    hf_id, split, repo_field, inst_field, traj_field, msgs_field, _ = DATASETS[dataset]
    logger.info("Loading %s split=%s ...", hf_id, split)
    ds = load_dataset(hf_id, split=split)

    if dataset == "swe-smith":
        rows = [r for r in ds if repo_filter in r[inst_field]]
    else:
        rows = [r for r in ds if r.get(repo_field) == repo_filter]
    logger.info("Filtered to %d trajectories for repo=%s", len(rows), repo_filter)

    by_inst: dict[str, list[dict]] = defaultdict(list)
    inst_order: list[str] = []
    for r in rows:
        inst = r[inst_field]
        if inst not in by_inst:
            inst_order.append(inst)
        by_inst[inst].append(r)
    logger.info("Unique instances: %d (avg %.2f traj/inst)",
                len(inst_order), len(rows) / max(1, len(inst_order)))

    if phase not in ("train", "eval"):
        raise ValueError(f"Unknown phase: {phase}")

    inst_first = [by_inst[inst][0] for inst in inst_order]
    fail = [t for t in inst_first if not t["resolved"]]
    succ = [t for t in inst_first if t["resolved"]]
    fail_rate = len(fail) / len(inst_first) if inst_first else 0.0
    tl = TRAIN_INST_LIMIT if train_limit is None else train_limit
    el = EVAL_INST_LIMIT if eval_limit is None else eval_limit
    n_train_fail = round(tl * fail_rate)
    n_eval_fail = round(el * fail_rate)
    n_train_succ = tl - n_train_fail
    n_eval_succ = el - n_eval_fail
    if phase == "train":
        out = fail[:n_train_fail] + succ[:n_train_succ]
    else:
        out = (fail[n_train_fail:n_train_fail + n_eval_fail]
               + succ[n_train_succ:n_train_succ + n_eval_succ])

    n_fail = sum(1 for r in out if not r["resolved"])
    logger.info(
        "Phase=%s: %d trajectories on %d unique instances, fail rate %.3f",
        phase, len(out), len({r[inst_field] for r in out}),
        n_fail / len(out) if out else 0.0,
    )
    return out


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
    dataset: str = "swe-smith",
) -> dict:
    """Replay a single trajectory through the proxy."""
    async with sem:
        _, _, _, inst_field, traj_field, msgs_field, msgs_json = DATASETS[dataset]
        traj_uuid = str(uuid.uuid4())
        dataset_traj_id = row[traj_field]
        instance_id = row[inst_field]
        resolved = row["resolved"]
        status = "success" if resolved else "failure"

        raw = row[msgs_field]
        raw_messages = json.loads(raw) if (msgs_json and isinstance(raw, str)) else raw
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

        # Mark trajectory status — use internal_error if any turns failed.
        # Persist instance_id in meta so basic.py / cascade can align
        # their tune/eval splits by stable key, not by parallel order.
        patch_status = "internal_error" if errors > 0 else status
        try:
            resp = await client.patch(
                f"{proxy_url}/episodiq/trajectories/{traj_uuid}",
                json={
                    "status": patch_status,
                    "meta": {"instance_id": instance_id},
                },
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
    dataset: str = "swe-smith",
    train_limit: int | None = None,
    eval_limit: int | None = None,
):
    rows = _load_trajectories(phase, repo_filter, dataset, train_limit, eval_limit)
    if not rows:
        logger.error("No trajectories to seed")
        return

    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        tasks = [_replay_trajectory(row, proxy_url, client, sem, dataset) for row in rows]
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
    parser.add_argument("--dataset", default="swe-smith", choices=list(DATASETS))
    parser.add_argument("--train-limit", type=int, default=None,
                        help=f"Override TRAIN_INST_LIMIT (default {TRAIN_INST_LIMIT})")
    parser.add_argument("--eval-limit", type=int, default=None,
                        help=f"Override EVAL_INST_LIMIT (default {EVAL_INST_LIMIT}; "
                             "set 0 to disable eval split and let train take all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    asyncio.run(run(
        phase=args.phase,
        proxy_url=args.proxy_url,
        concurrency=args.concurrency,
        output_path=args.output,
        repo_filter=args.repo_filter,
        dataset=args.dataset,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
    ))


if __name__ == "__main__":
    main()
