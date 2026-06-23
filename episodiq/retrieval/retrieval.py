"""Retrieval cascade: ``lsh_lookup`` → ``minhash_jaccard_rerank`` →
``cross_anchor_aggregate`` → ``aggshift_rerank``.

``lsh_lookup``: place query anchors at center steps spaced by ``W = 2w``,
with the last anchor aligned so its window ends at the current snapshot.
Each anchor's MinHash bands hit a Postgres LSH index filtered to
``window_center ∈ [s - w, s + w]`` so the candidate pool is both
hash-similar AND temporally aligned to the query's current step. Each
row carries the candidate's ``best_step`` (window with most band hits).

``minhash_jaccard_rerank``: per (anchor, candidate), recompute exact
MinHash jaccard between the query's signature at the anchor's window
and the candidate's signature at its ``best_step`` — replaces the
band-hit approximation with the precise jaccard. Candidate tokens are
fetched once and reused below.

``cross_anchor_aggregate``: per candidate, fold per-anchor jaccards
with ``min`` or ``mean`` (0-fill for anchors where the candidate
didn't appear). Keep top ``RetrievalConfig.jaccard_n_uniq``.

``aggshift_rerank``: exact Lev grid + agg-shift on survivors using the
already-fetched candidate tokens. Return top ``top_k``.

Caching
-------
``Retrieval`` optionally accepts a ``RetrievalCache``. When set, the
LSH lookups, candidate ``trace_tokens`` fetches, and dense MinHash
jaccard scores are memoised. ``RetrievalQuery.path_id`` is the cache
key for per-snapshot entries and is required whenever the cache is
enabled. Pre-warm by issuing one ``search()`` per snapshot with a
``RetrievalConfig`` whose ``prefetch_n_uniq`` is the widest the caller
will ever need; smaller subsequent ``prefetch_n_uniq`` values slice
down from the cached lists.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

import numpy as np

from episodiq.config.retrieval_config import RetrievalConfig, WindowMinHashConfig
from episodiq.config.scoring_config import AggShiftConfig
from episodiq.retrieval.cache import RetrievalCache
from episodiq.retrieval.candidate import RetrievalCandidate
from episodiq.retrieval.scoring import score_pair
from episodiq.retrieval.window_minhash import WindowMinHasher
from episodiq.storage.postgres.models import TrajectoryPath
from episodiq.storage.postgres.repository import (
    TrajectoryPathRepository,
    TrajectoryWindowLSHRepository,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalQuery:
    """One query's input to the cascade.

    ``tokens`` are the trajectory's cumulative ``trace_tokens`` for the
    snapshot being retrieved against. ``trajectory_id`` (when not None)
    is excluded from the candidate pool server-side. ``path_id`` is
    the cache key for per-snapshot entries and is required whenever
    ``Retrieval`` is constructed with a ``RetrievalCache``.
    """
    tokens: np.ndarray
    trajectory_id: UUID | None = None
    path_id: UUID | None = None


class Retrieval:
    """3-stage retrieval pipeline backed by a Postgres LSH band index."""

    def __init__(
        self,
        path_repo: TrajectoryPathRepository,
        lsh_repo: TrajectoryWindowLSHRepository,
        *,
        minhash_config: WindowMinHashConfig,
        retrieval_config: RetrievalConfig,
        scoring_config: AggShiftConfig,
        cache: RetrievalCache | None = None,
    ) -> None:
        self._paths = path_repo
        self._lsh = lsh_repo
        self._mh_cfg = minhash_config
        self._cas_cfg = retrieval_config
        self._scoring = scoring_config
        self._hasher = WindowMinHasher(self._mh_cfg)
        self._cache = cache

    @property
    def use_cache(self) -> bool:
        return self._cache is not None

    async def search(
        self, query: "RetrievalQuery | TrajectoryPath",
    ) -> list[RetrievalCandidate]:
        # Accept TrajectoryPath directly — caller usually has the path row
        # in hand and we adapt to RetrievalQuery internally.
        if not isinstance(query, RetrievalQuery):
            tokens = np.asarray(query.trace_tokens or [], dtype=np.int64)
            query = RetrievalQuery(
                tokens=tokens,
                trajectory_id=query.trajectory_id,
                path_id=query.id,
            )
        q_tokens = np.ascontiguousarray(query.tokens, dtype=np.int64)
        W = self._mh_cfg.window
        if q_tokens.size < W:
            return []

        # Place query anchors at center steps spaced by W = 2w. Last anchor
        # aligns so its window ends at snap_len; earlier anchors step
        # backward by W until they no longer fit.
        w = self._mh_cfg.half_window
        last_center = int(q_tokens.size) - w
        anchor_centers: list[int] = []
        c = last_center
        while c >= w:
            anchor_centers.append(c)
            c -= W
        anchor_centers.reverse()
        if not anchor_centers:
            return []

        # Per-anchor MinHash bands. Each anchor's window starts at
        # ``center - w`` (so it covers tokens[center-w : center+w]).
        anchor_bands: list[np.ndarray] = [
            self._hasher.bands_for_window(q_tokens, center - w)
            for center in anchor_centers
        ]

        anchor_rows = await self._lsh_lookup(
            anchor_centers, anchor_bands, query.trajectory_id,
            path_id=query.path_id,
        )
        if not anchor_rows:
            return []

        all_tids = list({
            tid for rows in anchor_rows.values() for tid, _ in rows
        })
        cand_tokens = await self._fetch_candidate_tokens(all_tids)

        anchor_scores = self._minhash_jaccard_rerank(
            anchor_rows, q_tokens, anchor_centers, cand_tokens,
            path_id=query.path_id,
        )
        if not anchor_scores:
            return []

        ranked = self._cross_anchor_aggregate(
            anchor_scores, len(anchor_centers),
        )
        if not ranked:
            return []

        # Off-load to a worker thread — the Numba JIT inside
        # ``score_pair`` releases the GIL, so concurrent ``search()``
        # calls under ``asyncio.gather`` see real CPU parallelism on
        # this Lev kernel instead of serialising on the event loop.
        return await asyncio.to_thread(
            self._aggshift_rerank, q_tokens, ranked, cand_tokens,
            query.path_id,
        )

    async def _lsh_lookup(
        self,
        anchor_centers: list[int],
        anchor_bands: list[np.ndarray],
        exclude_tid: UUID | None,
        *,
        path_id: UUID | None = None,
    ) -> dict[int, list[tuple[UUID, float]]]:
        """Wide LSH prefilter. ``{anchor_idx: [(tid, lsh_score), ...]}`` —
        per anchor, top ``prefetch_n_uniq`` candidates by ``cfg.aggregation``
        of band-hit counts. With ``self.use_cache`` true, cached entries
        are reused when their stored width is at least ``prefetch_n_uniq``
        and sliced; otherwise a fresh DB lookup repopulates the slot.
        """
        num_bands = self._mh_cfg.num_bands
        w = self._mh_cfg.half_window
        agg = self._cas_cfg.aggregation
        top_uniq = self._cas_cfg.prefetch_n_uniq

        anchor_rows: dict[int, list[tuple[UUID, float]]] = {}
        for qi, center in enumerate(anchor_centers):
            if self.use_cache:
                key = (path_id, qi, exclude_tid, agg)
                cached = self._cache.lsh.get(key)
                if cached is not None and len(cached) >= top_uniq:
                    anchor_rows[qi] = cached[:top_uniq]
                    continue
            band_pairs = [
                (b, int(anchor_bands[qi][b])) for b in range(num_bands)
            ]
            rows = await self._lsh.lookup(
                band_pairs,
                step_min=center - w,
                step_max=center + w,
                top_uniq=top_uniq,
                exclude_trajectory_id=exclude_tid,
                aggregation=agg,
            )
            if self.use_cache:
                self._cache.lsh.put(key, rows)
            anchor_rows[qi] = rows
        return anchor_rows

    async def _fetch_candidate_tokens(
        self, tids: list[UUID],
    ) -> dict[UUID, tuple[UUID | None, str, list[int]]]:
        """Fetch the latest ``trace_tokens`` per candidate trajectory.
        Hits ``CandidateTokensCache`` first; misses go to the DB and
        are written back.
        """
        if not self.use_cache:
            return await self._paths.get_latest_trace_tokens_for_trajectories(
                tids,
            )
        hits, misses = self._cache.tokens.get_many(tids)
        if not misses:
            return hits
        fresh = await self._paths.get_latest_trace_tokens_for_trajectories(
            misses,
        )
        self._cache.tokens.put_many(fresh)
        return {**hits, **fresh}

    def _minhash_jaccard_rerank(
        self,
        anchor_rows: dict[int, list[tuple[UUID, float]]],
        q_tokens: np.ndarray,
        anchor_centers: list[int],
        cand_tokens: dict[UUID, tuple[UUID | None, str, list[int]]],
        *,
        path_id: UUID | None = None,
    ) -> dict[int, dict[UUID, float]]:
        """Per (anchor, candidate in LSH pool), exact MinHash jaccard
        aggregated over the neighborhood per ``cfg.aggregation``. With
        ``self.use_cache`` true, scores are memoised per
        (path_id, anchor_idx, candidate, agg) so a later widened LSH
        pool only computes the new candidates. No per-anchor cap —
        cross_anchor_aggregate filters globally.
        """
        w = self._mh_cfg.half_window
        W = self._mh_cfg.window
        sig_size = self._mh_cfg.signature_size
        agg = self._cas_cfg.aggregation
        anchor_scores: dict[int, dict[UUID, float]] = {}
        for qi, center in enumerate(anchor_centers):
            tids = [t for t, _ in anchor_rows.get(qi, [])]
            # Fast path: every candidate at this anchor is already cached.
            # Skip the per-anchor query-side ``signature_for_window`` JIT
            # call entirely — it's pure overhead when nothing needs to be
            # recomputed.
            if self.use_cache:
                cached_all = {}
                miss = False
                for tid in tids:
                    val = self._cache.jaccard.get((path_id, qi, tid, agg))
                    if val is None:
                        miss = True
                        break
                    cached_all[tid] = val
                if not miss:
                    anchor_scores[qi] = cached_all
                    continue

            q_sig = self._hasher.signature_for_window(q_tokens, center - w)
            per_tid_score: dict[UUID, float] = {}
            for tid in tids:
                if self.use_cache:
                    cache_key = (path_id, qi, tid, agg)
                    cached = self._cache.jaccard.get(cache_key)
                    if cached is not None:
                        per_tid_score[tid] = cached
                        continue
                entry = cand_tokens.get(tid)
                if entry is None:
                    continue
                _action_cid, _status, tokens = entry
                if not tokens:
                    continue
                c_arr = np.asarray(tokens, dtype=np.int64)
                jaccs: list[float] = []
                for start in range(center - 2 * w, center + 1):
                    if start < 0 or start + W > c_arr.size:
                        continue
                    c_sig = self._hasher.signature_for_window(c_arr, start)
                    jaccs.append(float((q_sig == c_sig).sum()) / sig_size)
                if not jaccs:
                    continue
                # ``min_distance`` aggregation is optimistic across the
                # whole cascade: stage 4 agg-shift picks the best shift
                # in the band, so here we mirror that by picking the
                # best window in the neighborhood (max jaccard).
                score = (
                    max(jaccs) if agg == "min_distance"
                    else sum(jaccs) / len(jaccs)
                )
                per_tid_score[tid] = score
                if self.use_cache:
                    self._cache.jaccard.put(cache_key, score)
            anchor_scores[qi] = per_tid_score
        return anchor_scores

    def _cross_anchor_aggregate(
        self,
        anchor_scores: dict[int, dict[UUID, float]],
        n_anchors: int,
    ) -> list[tuple[UUID, float]]:
        """Cross-anchor mean of per-anchor dense scores (sum/n_anchors
        with 0-fill). Returns top ``jaccard_n_uniq`` — all go to
        ``aggshift_rerank``.
        """
        per_tid: dict[UUID, float] = defaultdict(float)
        for scores in anchor_scores.values():
            for tid, score in scores.items():
                per_tid[tid] += score
        aggregated = [(tid, total / n_anchors) for tid, total in per_tid.items()]
        aggregated.sort(key=lambda x: -x[1])
        return aggregated[:self._cas_cfg.jaccard_n_uniq]

    def _aggshift_rerank(
        self,
        q_tokens: np.ndarray,
        ranked: list[tuple[UUID, float]],
        cand_tokens: dict[UUID, tuple[UUID | None, str, list[int]]],
        path_id: UUID | None = None,
    ) -> list[RetrievalCandidate]:
        """Exact Lev grid + agg-shift on survivors. ``cfg.aggregation``
        also picks the per-row band reduction (``"min_distance"`` =
        best-shift wins, ``"mean"`` = average over the band) so the
        whole cascade — LSH lookup, jaccard rerank, agg-shift — uses
        one consistent aggregation. With ``self.use_cache`` true,
        ``(path_id, tid, AggShiftConfig, aggregation)`` keys the
        per-pair score; sweep trials sampling adjacent params on a
        coarse grid reuse the JIT output.
        """
        agg = self._cas_cfg.aggregation
        scored: list[RetrievalCandidate] = []
        for tid, _agg in ranked:
            entry = cand_tokens.get(tid)
            if entry is None:
                continue
            action_cid, status, tokens = entry
            if not tokens:
                continue
            if self.use_cache:
                cache_key = (path_id, tid, self._scoring, agg)
                cached = self._cache.aggshift.get(cache_key)
                if cached is not None:
                    sim = cached
                else:
                    c_arr = np.asarray(tokens, dtype=np.int64)
                    sim = score_pair(
                        q_tokens, c_arr, self._scoring,
                        band_aggregation=agg,
                    )
                    self._cache.aggshift.put(cache_key, sim)
            else:
                c_arr = np.asarray(tokens, dtype=np.int64)
                sim = score_pair(
                    q_tokens, c_arr, self._scoring,
                    band_aggregation=agg,
                )
            if sim <= -1.0:
                continue
            scored.append(RetrievalCandidate(
                trajectory_id=tid,
                score=sim,
                best_path_action_cluster_id=action_cid,
                trajectory_status=status,
            ))
        scored.sort(key=lambda r: -r.score)
        return scored[:self._cas_cfg.top_k]
