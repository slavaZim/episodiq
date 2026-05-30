"""(top_k, similarity_threshold) sweep on per-step ROC AUC + coverage.

Tune queries are a deterministic subset of completed trajectories (ordered by
trajectory_id, sliced by offset/limit). Leave-one-out: each query path's own
trajectory is excluded from the retrieval corpus.

Each query path computes ``fail_frac`` = fraction of the top-k retrieved
trajectories whose status is "failure". For step S ∈ {50, 60}, AUC is
computed over trajectories active at S using the most-recent snapshot's
fail_frac (``current`` aggregation). Coverage = fraction of query snapshots
with a non-empty shortlist at the threshold.

The corpus is loaded once into memory; all (sim, top_k) combos reuse the
same precomputed per-snapshot similarity vectors via post-hoc filtering.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from sklearn.metrics import roc_auc_score
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.storage.postgres.repository import TrajectoryRepository

logger = logging.getLogger(__name__)

DEFAULT_TOPK_GRID = [5, 10, 25]
DEFAULT_SIM_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_TUNE_LIMIT = 100
EVAL_STEP = 60


@dataclass(frozen=True)
class SweepPoint:
    top_k: int
    similarity_threshold: float
    auc_step60_current: float | None
    coverage_step60: float | None
    n_snapshots: int


@dataclass
class SweepReport:
    points: list[SweepPoint] = field(default_factory=list)

    @property
    def best(self) -> SweepPoint | None:
        scored = [(p.auc_step60_current or -1.0, p) for p in self.points]
        return max(scored, key=lambda x: x[0])[1] if scored else None


class RetrievalSweep:
    """Sweep (top_k, similarity_threshold) on a deterministic tune-query slice
    of completed trajectories. Reports per-step current AUC + coverage.

    The tune slice is ``ordered_traj_ids[offset : offset+limit]`` when
    ``ordered_traj_ids`` is set, otherwise ``all_completed[offset : offset+limit]``
    in UUID order. The full set of completed trajectories is always the
    retrieval corpus regardless of ordering.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        limit: int | None = None,
        offset: int = 0,
        ordered_traj_ids: list[UUID] | None = None,
    ):
        self._sf = session_factory
        self._limit = limit
        self._offset = offset
        self._ordered_traj_ids = ordered_traj_ids

    async def run(
        self, topk_grid: list[int], sim_grid: list[float],
    ) -> SweepReport:
        if not topk_grid or not sim_grid:
            return SweepReport()
        topk_grid = sorted(set(topk_grid))
        sim_grid = sorted(set(sim_grid))

        async with self._sf() as session:
            tune_paths, traj_status, corpus = await self._fetch_tune_and_corpus(
                session,
            )
        if not tune_paths:
            logger.warning("no tune paths found")
            return SweepReport()
        logger.info(
            "tune slice: %d snapshots from %d trajs; corpus %d paths",
            len(tune_paths), len(traj_status), len(corpus),
        )

        per_snapshot_shortlist = self._compute_shortlists(
            tune_paths, corpus,
        )
        logger.info(
            "shortlists computed: %d non-empty (of %d query snapshots)",
            len(per_snapshot_shortlist), len(tune_paths),
        )

        all_query_snaps = [(tid, step) for tid, step, _sig in tune_paths]
        report = SweepReport()
        for sim in sim_grid:
            for top_k in topk_grid:
                point = self._score_combo(
                    per_snapshot_shortlist, all_query_snaps, sim, top_k,
                    traj_status,
                )
                report.points.append(point)
                logger.info(
                    "sim=%.2f top_k=%d: cov@s60=%s s60=%s",
                    sim, top_k,
                    f"{point.coverage_step60 * 100:.1f}%" if point.coverage_step60 is not None else "n/a",
                    f"{point.auc_step60_current:.3f}" if point.auc_step60_current else "n/a",
                )
        return report

    async def _fetch_tune_and_corpus(
        self, session: AsyncSession,
    ) -> tuple[
        list[tuple[UUID, int, list[int]]],
        dict[UUID, str],
        list[tuple[UUID, str, list[int]]],
    ]:
        """Single-pass corpus + tune split: load every completed trajectory
        with eager-loaded paths once, slice [offset:offset+limit] for tune
        queries, use the full set as the retrieval corpus.
        """
        traj_repo = TrajectoryRepository(session)
        all_trajs = await traj_repo.get_with_completed_paths(
            ["success", "failure"],
        )
        if not all_trajs:
            return [], {}, []

        if self._ordered_traj_ids is not None:
            by_id = {t.id: t for t in all_trajs}
            ordered = [by_id[tid] for tid in self._ordered_traj_ids if tid in by_id]
            unseen = [t for t in all_trajs if t.id not in set(self._ordered_traj_ids)]
            all_trajs = ordered + unseen

        end = (
            self._offset + self._limit if self._limit is not None else len(all_trajs)
        )
        tune_trajs = all_trajs[self._offset : end]
        traj_status = {t.id: t.status for t in tune_trajs}

        tune_paths: list[tuple[UUID, int, list[int]]] = []
        for t in tune_trajs:
            for p in t.paths:
                if p.minhash_sig and p.index is not None:
                    tune_paths.append((t.id, int(p.index), list(p.minhash_sig)))

        corpus: list[tuple[UUID, str, list[int]]] = []
        for t in all_trajs:
            for p in t.paths:
                if p.minhash_sig:
                    corpus.append((t.id, t.status, list(p.minhash_sig)))
        return tune_paths, traj_status, corpus

    def _compute_shortlists(
        self,
        tune_paths: list[tuple[UUID, int, list[int]]],
        corpus: list[tuple[UUID, str, list[int]]],
    ) -> dict[tuple[UUID, int], list[tuple[UUID, float, str]]]:
        """For each tune snapshot, compute MinHash similarity to every
        corpus path, MAX-pool per trajectory (leave-one-out: skip the query
        traj's own paths), and return the per-snapshot list sorted by
        similarity descending. Format: dict[(q_tid, step)] = [(cand_tid,
        sim, status), ...].

        The corpus is stacked into a 2D numpy matrix once and reused across
        all query snapshots.
        """
        if not corpus:
            return {}
        k = len(corpus[0][2])
        # Filter to signatures of expected length, build parallel arrays.
        path_tids = []
        path_sigs = []
        tid_status: dict[UUID, str] = {}
        for tid, status, sig in corpus:
            if not sig or len(sig) != k:
                continue
            if tid not in tid_status:
                tid_status[tid] = status
            path_tids.append(tid)
            path_sigs.append(sig)
        if not path_sigs:
            return {}
        sig_matrix = np.asarray(path_sigs, dtype=np.int64)
        tid_arr = np.asarray(path_tids, dtype=object)

        # Map every unique trajectory_id to a contiguous int, so MAX-pool can
        # be done with np.maximum.at (no Python loop over candidates).
        uniq_tids, inverse = np.unique(tid_arr, return_inverse=True)
        n_uniq = len(uniq_tids)

        out: dict[tuple[UUID, int], list[tuple[UUID, float, str]]] = {}
        for i, (q_tid, q_step, q_sig) in enumerate(tune_paths):
            if len(q_sig) != k:
                continue
            query_arr = np.asarray(q_sig, dtype=np.int64)
            path_sims = (sig_matrix == query_arr).sum(axis=1) / k
            traj_max = np.full(n_uniq, -1.0, dtype=np.float64)
            np.maximum.at(traj_max, inverse, path_sims)
            entries: list[tuple[UUID, float, str]] = []
            for ti in range(n_uniq):
                cand_tid = uniq_tids[ti]
                if cand_tid == q_tid:
                    continue
                sim = float(traj_max[ti])
                if sim <= 0:
                    continue
                entries.append((cand_tid, sim, tid_status[cand_tid]))
            entries.sort(key=lambda x: -x[1])
            if entries:
                out[(q_tid, q_step)] = entries
            if (i + 1) % 1000 == 0:
                logger.info("shortlists: %d/%d query snapshots", i + 1, len(tune_paths))
        return out

    def _score_combo(
        self,
        shortlists: dict[tuple[UUID, int], list[tuple[UUID, float, str]]],
        all_query_snaps: list[tuple[UUID, int]],
        sim_threshold: float,
        top_k: int,
        traj_status: dict[UUID, str],
    ) -> SweepPoint:
        per_traj: dict[UUID, list[tuple[int, float]]] = defaultdict(list)
        n_kept = 0
        for (q_tid, q_step), shortlist in shortlists.items():
            filtered = [(c, s, st) for c, s, st in shortlist
                        if s >= sim_threshold]
            if not filtered:
                continue
            top = filtered[:top_k]
            fail = sum(1 for _c, _s, st in top if st == "failure")
            per_traj[q_tid].append((q_step, fail / len(top)))
            n_kept += 1

        auc = self._current_auc_at_step(per_traj, EVAL_STEP, traj_status)
        cov = self._coverage_at_step(all_query_snaps, per_traj, EVAL_STEP)
        return SweepPoint(
            top_k=top_k,
            similarity_threshold=sim_threshold,
            auc_step60_current=auc,
            coverage_step60=cov,
            n_snapshots=n_kept,
        )

    @staticmethod
    def _coverage_at_step(
        all_snaps: list[tuple[UUID, int]],
        per_traj: dict[UUID, list[tuple[int, float]]],
        min_step: int,
    ) -> float | None:
        """Per-snapshot density at step s: fraction of query snapshots with
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

    @staticmethod
    def _current_auc_at_step(
        per_traj: dict[UUID, list[tuple[int, float]]],
        min_step: int,
        traj_status: dict[UUID, str],
    ) -> float | None:
        """Mean ROC AUC over s >= min_step using current aggregation."""
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
                y.append(1 if traj_status.get(tid) == "failure" else 0)
                scores.append(ordered[-1][1])
            if len(set(y)) >= 2:
                aucs.append(roc_auc_score(y, scores))
        return float(np.mean(aucs)) if aucs else None
