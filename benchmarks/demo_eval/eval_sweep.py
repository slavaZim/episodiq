"""Leakage-free eval for the peak-AUC tune config from step 08.

Pipeline:
  1. Load tune trajectories (offset 0, limit TUNE_LIMIT) as the retrieval
     corpus. Each eval trajectory's prediction depends only on tune
     signatures — eval queries don't see one another.
  2. Load eval trajectories (offset TUNE_LIMIT, limit EVAL_LIMIT) as queries.
  3. For the tune config (top_k, similarity_threshold):
       a. For every eval snapshot, compute MinHash similarity against the
          tune corpus (numpy-vectorised), MAX-pool per tune trajectory,
          apply the threshold, slice to top_k.
       b. fail_frac = fraction of the top_k whose trajectory_status is
          'failure'.
       c. Build per-trajectory (snapshot_step, fail_frac) sequences.
       d. Compute AUC@s60_current = mean ROC AUC over steps s in [60,
          max_step] (current = fail_frac at most-recent snapshot <= s).
       e. Coverage @ s60 = fraction of snapshots at step >= 60 that
          themselves passed the similarity filter.
  4. Dump result to JSON.

Usage:
    uv run python eval_sweep.py \\
      --env benchmarks/demo_eval_1/.env \\
      --config benchmarks/demo_eval_1/output/tune_config.json \\
      --tune-offset 0 --tune-limit 55 \\
      --eval-offset 55 --eval-limit 220 \\
      --output benchmarks/demo_eval_1/output/eval_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from uuid import UUID

import numpy as np
from sklearn.metrics import roc_auc_score
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from episodiq.cli.env import _load_dotenv
from episodiq.config import get_config
from episodiq.storage.postgres.repository import TrajectoryRepository

logger = logging.getLogger(__name__)

EVAL_STEP = 60


async def _fetch_corpus_and_eval(
    session_factory, tune_offset: int, tune_limit: int,
    eval_offset: int, eval_limit: int,
    ordered_traj_ids: list[UUID] | None = None,
):
    """Return:
      tune_corpus: list[(traj_id, traj_status, sig)]
      eval_trajs:  list[(traj_id, traj_status, [(step, sig), ...])]
    """
    async with session_factory() as session:
        repo = TrajectoryRepository(session)
        if ordered_traj_ids is not None:
            # Load all once, reorder by external list, then slice.
            all_trajs = await repo.get_with_completed_paths(
                ["success", "failure"],
            )
            by_id = {t.id: t for t in all_trajs}
            ordered_set = set(ordered_traj_ids)
            reordered = [by_id[tid] for tid in ordered_traj_ids if tid in by_id]
            unseen = [t for t in all_trajs if t.id not in ordered_set]
            all_ordered = reordered + unseen
            tune_trajs = all_ordered[tune_offset : tune_offset + tune_limit]
            eval_trajs_raw = all_ordered[eval_offset : eval_offset + eval_limit]
        else:
            tune_trajs = await repo.get_with_completed_paths(
                ["success", "failure"], limit=tune_limit, offset=tune_offset,
            )
            eval_trajs_raw = await repo.get_with_completed_paths(
                ["success", "failure"], limit=eval_limit, offset=eval_offset,
            )

    tune_corpus: list[tuple[UUID, str, list[int]]] = []
    for t in tune_trajs:
        for p in t.paths:
            if p.minhash_sig:
                tune_corpus.append((t.id, t.status, list(p.minhash_sig)))

    eval_trajs: list[tuple[UUID, str, list[tuple[int, list[int]]]]] = []
    for t in eval_trajs_raw:
        snaps = []
        for p in t.paths:
            if p.minhash_sig and p.index is not None:
                snaps.append((int(p.index), list(p.minhash_sig)))
        if snaps:
            eval_trajs.append((t.id, t.status, snaps))

    return tune_corpus, eval_trajs


def _stack_corpus(
    corpus: list[tuple[UUID, str, list[int]]],
) -> tuple[np.ndarray, np.ndarray, list[UUID], dict[UUID, str]]:
    """Build the numpy matrices needed for vectorised per-query lookup."""
    if not corpus:
        return np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=np.int64), [], {}
    k = len(corpus[0][2])
    sigs = []
    tids = []
    tid_status: dict[UUID, str] = {}
    for tid, status, sig in corpus:
        if not sig or len(sig) != k:
            continue
        sigs.append(sig)
        tids.append(tid)
        if tid not in tid_status:
            tid_status[tid] = status
    sig_matrix = np.asarray(sigs, dtype=np.int64)
    tid_arr = np.asarray(tids, dtype=object)
    uniq_tids, inverse = np.unique(tid_arr, return_inverse=True)
    return sig_matrix, inverse, list(uniq_tids), tid_status


def _per_traj_fail_fracs(
    eval_trajs: list[tuple[UUID, str, list[tuple[int, list[int]]]]],
    sig_matrix: np.ndarray,
    inverse: np.ndarray,
    uniq_tids: list[UUID],
    tid_status: dict[UUID, str],
    top_k: int,
    sim_threshold: float,
) -> tuple[
    dict[UUID, list[tuple[int, float]]],
    list[tuple[UUID, int]],
    int, int,
]:
    """For each eval snapshot, retrieve top_k tune trajectories by MinHash sim
    (above threshold) and record fail_frac. Returns:
      - per_traj: per-eval-traj (step, fail_frac) sequence (filtered)
      - all_snaps: list of every eval (traj_id, step), regardless of filter
      - total / kept: raw snapshot counts (kept = had a non-empty shortlist)
    """
    if sig_matrix.size == 0:
        return {}, [], 0, 0
    k = sig_matrix.shape[1]
    n_uniq = len(uniq_tids)
    statuses = np.asarray(
        [1 if tid_status[t] == "failure" else 0 for t in uniq_tids],
        dtype=np.int64,
    )
    per_traj: dict[UUID, list[tuple[int, float]]] = defaultdict(list)
    all_snaps: list[tuple[UUID, int]] = []
    total = 0
    kept = 0
    for q_tid, _q_status, snaps in eval_trajs:
        for step, q_sig in snaps:
            if len(q_sig) != k:
                continue
            total += 1
            all_snaps.append((q_tid, step))
            q_arr = np.asarray(q_sig, dtype=np.int64)
            path_sims = (sig_matrix == q_arr).sum(axis=1) / k
            traj_max = np.full(n_uniq, -1.0, dtype=np.float64)
            np.maximum.at(traj_max, inverse, path_sims)
            mask = traj_max >= sim_threshold
            if not mask.any():
                continue
            sims_kept = traj_max[mask]
            status_kept = statuses[mask]
            order = np.argsort(-sims_kept, kind="stable")[:top_k]
            if order.size == 0:
                continue
            fail_count = int(status_kept[order].sum())
            ff = fail_count / order.size
            per_traj[q_tid].append((step, ff))
            kept += 1
    return per_traj, all_snaps, total, kept


def _coverage_at_step(
    all_snaps: list[tuple[UUID, int]],
    per_traj: dict[UUID, list[tuple[int, float]]],
    min_step: int,
) -> float | None:
    """Per-snapshot density at step s: fraction of eval snapshots with
    step >= min_step that themselves passed the similarity filter.
    """
    total = sum(1 for _tid, step in all_snaps if step >= min_step)
    if total == 0:
        return None
    filtered_set = {
        (tid, st) for tid, snaps in per_traj.items() for st, _ in snaps
    }
    covered = sum(
        1 for tid, step in all_snaps
        if step >= min_step and (tid, step) in filtered_set
    )
    return covered / total


def _current_auc_at_step(
    per_traj: dict[UUID, list[tuple[int, float]]],
    traj_label: dict[UUID, int],
    min_step: int,
) -> float | None:
    """Mean ROC AUC over s >= min_step using current aggregation: a
    trajectory contributes at step s only if it has a filtered snapshot
    at step <= s AND its max filtered step is >= s (i.e. the trajectory
    is still active at s by the filter). Steps with single-class
    survivors are skipped.
    """
    if not per_traj:
        return None
    max_step_per_traj = {
        tid: max(s for s, _ in snaps) for tid, snaps in per_traj.items()
    }
    max_step = max(max_step_per_traj.values())
    aucs = []
    for s in range(min_step, max_step + 1):
        y, scores = [], []
        for tid, snaps in per_traj.items():
            if max_step_per_traj[tid] < s:
                continue
            ordered = sorted([(st, ff) for st, ff in snaps if st <= s])
            if not ordered:
                continue
            y.append(traj_label[tid])
            scores.append(ordered[-1][1])
        if len(set(y)) >= 2:
            aucs.append(roc_auc_score(y, scores))
    return float(np.mean(aucs)) if aucs else None


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tune-offset", type=int, default=0)
    p.add_argument("--tune-limit", type=int, default=55)
    p.add_argument("--eval-offset", type=int, default=55)
    p.add_argument("--eval-limit", type=int, default=220)
    p.add_argument("--order-file", type=Path, default=None,
                   help="JSON list of trajectory UUIDs; overrides default UUID "
                        "ordering for tune/eval slicing.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _load_dotenv(args.env)
    cfg = get_config()
    engine = create_async_engine(cfg.get_database_url(), poolclass=NullPool)
    sf = async_sessionmaker(engine, expire_on_commit=False)

    configs = json.loads(args.config.read_text())
    logger.info("loaded tune config: %s", configs)

    ordered_ids: list[UUID] | None = None
    if args.order_file is not None:
        ordered_ids = [UUID(s) for s in json.loads(args.order_file.read_text())]
        logger.info("ordering: %s (%d ids)", args.order_file, len(ordered_ids))

    tune_corpus, eval_trajs = await _fetch_corpus_and_eval(
        sf, args.tune_offset, args.tune_limit,
        args.eval_offset, args.eval_limit,
        ordered_traj_ids=ordered_ids,
    )
    await engine.dispose()
    logger.info(
        "tune corpus: %d paths from %d trajs; eval: %d trajs",
        len(tune_corpus), args.tune_limit, len(eval_trajs),
    )

    sig_matrix, inverse, uniq_tids, tid_status = _stack_corpus(tune_corpus)
    traj_label = {
        tid: 1 if status == "failure" else 0 for tid, status, _ in eval_trajs
    }

    top_k = int(configs["top_k"])
    sim_threshold = float(configs["similarity_threshold"])
    logger.info("config: top_k=%d sim=%.4f", top_k, sim_threshold)
    per_traj, all_snaps, total_snaps, kept_snaps = _per_traj_fail_fracs(
        eval_trajs, sig_matrix, inverse, uniq_tids, tid_status,
        top_k, sim_threshold,
    )
    auc = _current_auc_at_step(per_traj, traj_label, EVAL_STEP)
    cov = _coverage_at_step(all_snaps, per_traj, EVAL_STEP)
    result = {
        "config": {"top_k": top_k, "similarity_threshold": sim_threshold},
        "tune": {
            "coverage_step60": configs.get("tune_coverage_step60"),
            "auc_step60_current": configs.get("tune_auc_step60_current"),
        },
        "eval": {
            "n_snapshots_total": total_snaps,
            "n_snapshots_with_shortlist": kept_snaps,
            "n_trajectories": len(per_traj),
            "coverage_step60": cov,
            "auc_step60_current": auc,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    def _fmt(v, suffix=""):
        return f"{v:.4f}{suffix}" if v is not None else "n/a"

    cov_str = f"{cov * 100:.1f}%" if cov is not None else "n/a"
    print("\n=== Eval result ===")
    print(f"top_k={top_k} sim={sim_threshold:.3f}  "
          f"cov@s60={cov_str}  AUC@s60={_fmt(auc)}")
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
