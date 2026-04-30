"""Re-seed only internal_error eval trajectories.

1. Load eval_traj_ids.json
2. Query DB for which are internal_error
3. Delete those trajectories (cascade: messages, paths)
4. Re-replay those specific SWE-smith trajectories through proxy
5. Update eval_traj_ids.json with new UUIDs
"""

import argparse
import asyncio
import json
import logging

import httpx

from seed_via_proxy import (
    _load_trajectories,
    _replay_trajectory,
)

logger = logging.getLogger(__name__)

PSQL_URL_TEMPLATE = "postgresql://{user}:{password}@{host}:{port}/{db}"


async def run(
    eval_ids_path: str,
    proxy_url: str,
    concurrency: int,
    db_url: str,
):
    # 1. Load eval mapping
    with open(eval_ids_path) as f:
        eval_ids = json.load(f)

    # 2. Find internal_error UUIDs via DB
    import asyncpg
    conn = await asyncpg.connect(db_url)
    rows = await conn.fetch(
        "SELECT id::text FROM trajectories WHERE status = 'internal_error'"
    )
    ie_uuids = {r["id"] for r in rows}
    await conn.close()

    # 3. Filter to eval trajectories that are internal_error
    failed = {uuid: meta for uuid, meta in eval_ids.items() if uuid in ie_uuids}
    if not failed:
        logger.info("No internal_error eval trajectories found")
        return

    logger.info("Found %d internal_error eval trajectories to reseed", len(failed))

    # 4. Delete from DB (cascade handles messages + paths)
    conn = await asyncpg.connect(db_url)
    for uuid in failed:
        await conn.execute(
            "DELETE FROM trajectory_paths WHERE trajectory_id = $1::uuid", uuid
        )
        await conn.execute("""
            DELETE FROM origin_responses WHERE message_id IN (
                SELECT id FROM messages WHERE trajectory_id = $1::uuid
            )
        """, uuid)
        await conn.execute(
            "DELETE FROM messages WHERE trajectory_id = $1::uuid", uuid
        )
        await conn.execute(
            "DELETE FROM trajectories WHERE id = $1::uuid", uuid
        )
        logger.info("Deleted trajectory %s", uuid[:8])
    await conn.close()

    # 5. Load dataset to get the specific rows
    traj_id_to_meta = {meta["traj_id"]: (uuid, meta) for uuid, meta in failed.items()}
    all_rows = _load_trajectories("eval", "getmoto__moto")
    rows_to_reseed = [r for r in all_rows if r["traj_id"] in traj_id_to_meta]

    logger.info("Matched %d rows from dataset", len(rows_to_reseed))

    # 6. Replay
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [_replay_trajectory(row, proxy_url, client, sem) for row in rows_to_reseed]
        results = await asyncio.gather(*tasks)

    # 7. Update eval_traj_ids.json: remove old UUIDs, add new ones
    for old_uuid in failed:
        del eval_ids[old_uuid]

    for r in results:
        eval_ids[r["uuid"]] = {k: v for k, v in r.items() if k != "uuid"}

    with open(eval_ids_path, "w") as f:
        json.dump(eval_ids, f, indent=2)

    n_errors = sum(1 for r in results if r.get("error"))
    logger.info(
        "Reseed complete: %d trajectories, %d errors. Updated %s",
        len(results), n_errors, eval_ids_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Reseed internal_error eval trajectories")
    parser.add_argument("--eval-ids", default="output/eval_traj_ids.json")
    parser.add_argument("--proxy-url", default="http://localhost:8081")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--db-url", required=True, help="PostgreSQL URL (not asyncpg)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    asyncio.run(run(
        eval_ids_path=args.eval_ids,
        proxy_url=args.proxy_url,
        concurrency=args.concurrency,
        db_url=args.db_url,
    ))


if __name__ == "__main__":
    main()
