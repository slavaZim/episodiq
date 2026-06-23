"""In-memory caches for the Retrieval cascade.

The cascade pipeline has three points where state can be reused across
search calls:

  - **LSH lookup** — per (query path, anchor index, exclude trajectory,
    aggregation). Stored list is the widest fetched so far; callers
    slice down to their current ``top_uniq``. When a later search needs
    more candidates than cached, the DB lookup runs again at the new
    width and replaces the entry.

  - **Dense MinHash jaccard** — per (query path, anchor index, candidate
    trajectory, aggregation). The value is a single float; the cache
    grows incrementally as wider LSH pools surface new candidates for
    each anchor.

  - **Candidate trace tokens** — per candidate trajectory. Read-mostly
    snapshot of the latest path's ``(action_cluster_id, status,
    trace_tokens)`` tuple.

Plus a fourth slot for agg-shift / Lev scores keyed by the full
``AggShiftConfig`` — sweep trials sampling adjacent params on a coarse
grid reuse the JIT output across (path, candidate) pairs.

The container is intentionally process-local and unbounded — sweep runs
construct one ``RetrievalCache`` per W and discard it. None of the
maps are thread-safe; cooperative concurrency (single asyncio loop) is
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class LSHCache:
    """``(path_id, anchor_idx, exclude_tid, aggregation) → [(tid, score), ...]``.

    Values are stored at the widest ``top_uniq`` seen so far. Use
    ``get`` to read; the caller slices down to its current width.
    Calls to ``put`` only replace existing entries when the new list is
    strictly longer than the stored one.
    """
    _data: dict[
        tuple[UUID, int, UUID | None, str],
        list[tuple[UUID, float]],
    ] = field(default_factory=dict)

    def get(
        self, key: tuple[UUID, int, UUID | None, str],
    ) -> list[tuple[UUID, float]] | None:
        return self._data.get(key)

    def put(
        self,
        key: tuple[UUID, int, UUID | None, str],
        value: list[tuple[UUID, float]],
    ) -> None:
        existing = self._data.get(key)
        if existing is None or len(value) > len(existing):
            self._data[key] = value


@dataclass
class JaccardCache:
    """``(path_id, anchor_idx, candidate_tid, aggregation) → float``.

    One entry per (snapshot, anchor, candidate, agg). Filled lazily as
    the LSH layer surfaces candidates for each anchor.
    """
    _data: dict[
        tuple[UUID, int, UUID, str],
        float,
    ] = field(default_factory=dict)

    def get(
        self, key: tuple[UUID, int, UUID, str],
    ) -> float | None:
        return self._data.get(key)

    def put(
        self,
        key: tuple[UUID, int, UUID, str],
        value: float,
    ) -> None:
        self._data[key] = value


@dataclass
class AggShiftCache:
    """``(path_id, candidate_tid, AggShiftConfig, aggregation) → float``.

    ``AggShiftConfig`` is a frozen dataclass — hashable, used directly
    as part of the composite key. ``aggregation`` ("min_distance" or
    "mean") selects the per-row band reduction inside
    ``agg_shift_score`` and is therefore also part of the key. With
    sweep trials discretising ``lam`` / ``gap_open`` / ``gap_extend``
    on a coarse grid, adjacent trials in the same ``(W, agg)`` slot
    hit the same configs and skip the Numba JIT ``score_pair`` call.
    """
    _data: dict[
        tuple[UUID, UUID, object, str],
        float,
    ] = field(default_factory=dict)

    def get(
        self, key: tuple[UUID, UUID, object, str],
    ) -> float | None:
        return self._data.get(key)

    def put(
        self,
        key: tuple[UUID, UUID, object, str],
        value: float,
    ) -> None:
        self._data[key] = value


@dataclass
class CandidateTokensCache:
    """``tid → (action_cluster_id, status, trace_tokens)``.

    Read-mostly; one entry per candidate trajectory ever seen.
    """
    _data: dict[UUID, tuple[UUID | None, str, list[int]]] = field(
        default_factory=dict,
    )

    def get(self, tid: UUID) -> tuple[UUID | None, str, list[int]] | None:
        return self._data.get(tid)

    def put(
        self, tid: UUID, value: tuple[UUID | None, str, list[int]],
    ) -> None:
        self._data[tid] = value

    def get_many(
        self, tids: list[UUID],
    ) -> tuple[dict[UUID, tuple[UUID | None, str, list[int]]], list[UUID]]:
        """Split ``tids`` into ``(hits, misses)`` against the cache."""
        hits: dict[UUID, tuple[UUID | None, str, list[int]]] = {}
        misses: list[UUID] = []
        for tid in tids:
            entry = self._data.get(tid)
            if entry is None:
                misses.append(tid)
            else:
                hits[tid] = entry
        return hits, misses

    def put_many(
        self,
        entries: dict[UUID, tuple[UUID | None, str, list[int]]],
    ) -> None:
        self._data.update(entries)


@dataclass
class RetrievalCache:
    """Container — pass one of these into ``Retrieval(cache=...)`` to
    enable cross-call reuse. Defaults are empty caches.
    """
    lsh: LSHCache = field(default_factory=LSHCache)
    jaccard: JaccardCache = field(default_factory=JaccardCache)
    tokens: CandidateTokensCache = field(default_factory=CandidateTokensCache)
    aggshift: AggShiftCache = field(default_factory=AggShiftCache)
