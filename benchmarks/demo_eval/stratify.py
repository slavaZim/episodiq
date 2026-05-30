"""Compute a stratified trajectory ordering balanced across
``(status, length_quartile)`` and write it as a JSON list of UUIDs.

The tune/eval slicing in the demo pipeline (``offset/limit`` over completed
trajectories) defaults to ``ORDER BY trajectory_id`` — a UUID order, i.e.
random for the purposes of fail/succ balance and trajectory length. Round-
robin interleaving across status × length-quartile buckets keeps each
prefix of the order representative of the whole population.

Output is consumed by ``tune retrieval-sweep --order-file`` and
``eval_sweep.py --order-file``.

Usage:
    uv run python stratify.py --env .env --output output/stratified_order.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from episodiq.cli.env import _load_dotenv
from episodiq.config import get_config
from episodiq.storage.postgres.models import Trajectory, TrajectoryPath

logger = logging.getLogger(__name__)


async def _load(database_url: str) -> list[tuple[str, str, int]]:
    """Return ``[(traj_id, status, n_paths)]`` for trajectories with at
    least one completed path."""
    engine = create_async_engine(database_url, poolclass=NullPool)
    sf = async_sessionmaker(engine, expire_on_commit=False)
    async with sf() as session:
        stmt = (
            select(
                Trajectory.id,
                Trajectory.status,
                func.count(TrajectoryPath.id).label("n_paths"),
            )
            .join(TrajectoryPath, TrajectoryPath.trajectory_id == Trajectory.id)
            .where(TrajectoryPath.to_observation_id.isnot(None))
            .where(Trajectory.status.in_(("success", "failure")))
            .group_by(Trajectory.id, Trajectory.status)
            .order_by(Trajectory.id)
        )
        rows = (await session.execute(stmt)).all()
    await engine.dispose()
    return [(str(r.id), r.status, int(r.n_paths)) for r in rows]


def _stratified_order(
    rows: list[tuple[str, str, int]],
    *,
    by_status: bool = True,
    by_length: bool = True,
    shuffle_seed: int | None = None,
) -> list[str]:
    """Round-robin interleave across (status?, length_quartile?) buckets.

    Toggle ``by_status`` / ``by_length`` to drop that dimension. With both
    off and a ``shuffle_seed`` set, this becomes a pure shuffle (single
    bucket). Within each bucket items are sorted by UUID for determinism
    unless ``shuffle_seed`` is set, in which case it's a random shuffle
    seeded with that value.
    """
    import random

    if by_length:
        lengths = np.asarray([r[2] for r in rows], dtype=np.float64)
        edges = np.quantile(lengths, [0.25, 0.50, 0.75])

        def length_bucket(n_paths: int) -> int:
            for i, e in enumerate(edges):
                if n_paths <= e:
                    return i
            return len(edges)
    else:
        def length_bucket(_n_paths: int) -> int:  # type: ignore[misc]
            return 0

    by_bucket: dict[tuple, list[str]] = defaultdict(list)
    for tid, status, n_paths in rows:
        key = (status if by_status else "*", length_bucket(n_paths))
        by_bucket[key].append(tid)

    bucket_lists: list[list[str]] = []
    rng = random.Random(shuffle_seed) if shuffle_seed is not None else None
    for v in by_bucket.values():
        if rng is None:
            bucket_lists.append(sorted(v))
        else:
            shuffled = list(v)
            rng.shuffle(shuffled)
            bucket_lists.append(shuffled)

    # Stride-based interleave: each bucket's items are placed at evenly-spaced
    # positions across the full range, so the bucket's share of any prefix
    # stays close to its global share. Plain zip_longest round-robin would
    # exhaust the smaller bucket first and dump the larger bucket's tail at
    # the end, breaking proportional balance.
    total = sum(len(b) for b in bucket_lists)
    placed: list[tuple[float, int, str]] = []
    for b_idx, bucket in enumerate(bucket_lists):
        n = len(bucket)
        if n == 0:
            continue
        for j, tid in enumerate(bucket):
            pos = (j + 0.5) * total / n
            placed.append((pos, b_idx, tid))
    placed.sort(key=lambda x: (x[0], x[1]))
    return [tid for _pos, _b, tid in placed]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=".env")
    p.add_argument("--output", default="output/stratified_order.json")
    p.add_argument("--no-status", action="store_true",
                   help="Skip status (fail/succ) bucketing.")
    p.add_argument("--no-length", action="store_true",
                   help="Skip length-quartile bucketing.")
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="If set, shuffle within each bucket (or globally if "
                        "no bucketing) with this seed instead of sorting by "
                        "UUID.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(args.env)
    cfg = get_config()

    rows = asyncio.run(_load(cfg.get_database_url()))
    logger.info(
        "loaded %d trajectories (by_status=%s, by_length=%s, shuffle_seed=%s)",
        len(rows), not args.no_status, not args.no_length, args.shuffle_seed,
    )

    order = _stratified_order(
        rows,
        by_status=not args.no_status,
        by_length=not args.no_length,
        shuffle_seed=args.shuffle_seed,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(order, f, indent=2)

    # Sanity: fail/succ balance + length-quartile spread of the first 55 (the
    # tune slice in 08_tune.sh) and the remaining 220 (the eval slice).
    by_id = {tid: (status, n_paths) for tid, status, n_paths in rows}
    for label, sl in [("tune (first 55)", order[:55]), ("eval (next 220)", order[55:275])]:
        stat = [by_id[t][0] for t in sl]
        paths = [by_id[t][1] for t in sl]
        n_fail = sum(1 for s in stat if s == "failure")
        logger.info(
            "%s: %d trajs, %d failure (%.1f%%), n_paths quartiles=%s",
            label, len(sl), n_fail, 100.0 * n_fail / max(1, len(sl)),
            np.quantile(paths, [0.25, 0.5, 0.75]).tolist() if paths else None,
        )

    print(f"saved {len(order)} entries to {args.output}")


if __name__ == "__main__":
    main()
