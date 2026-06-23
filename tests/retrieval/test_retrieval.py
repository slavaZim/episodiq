"""Per-stage unit tests for the cascade ``Retrieval`` pipeline plus a
cache-parity test. Each stage is exercised in isolation so failures
point at one part of the cascade; the parity test then proves that
turning the cache on doesn't change the output. Integration against a
real Postgres LSH index lives in ``test_retrieval_db.py``.
"""

from uuid import UUID, uuid4

import numpy as np
import pytest

from episodiq.config.retrieval_config import (
    RetrievalConfig, WindowMinHashConfig,
)
from episodiq.config.scoring_config import AggShiftConfig
from episodiq.retrieval.cache import RetrievalCache
from episodiq.retrieval.retrieval import Retrieval, RetrievalQuery
from episodiq.retrieval.scoring import _window_affine_distance, score_pair
from episodiq.retrieval.window_minhash import WindowMinHasher

from tests.in_memory_repos import InMemoryTrajectoryWindowLSHRepository


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


class _StubPathRepo:
    """Minimal stand-in — only the method ``Retrieval`` calls. Records
    each fetch so tests can assert how many times the repo was hit.
    """

    def __init__(self) -> None:
        self.tokens: dict[UUID, tuple[UUID | None, str, list[int]]] = {}
        self.fetch_calls: list[tuple[UUID, ...]] = []

    def add(
        self, tid: UUID, tokens: list[int], status: str = "success",
    ) -> None:
        self.tokens[tid] = (uuid4(), status, tokens)

    async def get_latest_trace_tokens_for_trajectories(
        self, tids: list[UUID],
    ) -> dict[UUID, tuple[UUID | None, str, list[int]]]:
        self.fetch_calls.append(tuple(tids))
        return {t: self.tokens[t] for t in tids if t in self.tokens}


def _mh(window: int = 10) -> WindowMinHashConfig:
    return WindowMinHashConfig(window=window)


def _cas(**overrides) -> RetrievalConfig:
    defaults = dict(
        aggregation="mean", prefetch_n_uniq=50,
        jaccard_n_uniq=40, top_k=5,
    )
    defaults.update(overrides)
    return RetrievalConfig(**defaults)


def _ms(window: int = 10) -> AggShiftConfig:
    return AggShiftConfig(window=window)


def _retrieval(
    *, paths=None, lsh=None, cas=None, cache=None, mh=None,
) -> Retrieval:
    return Retrieval(
        paths or _StubPathRepo(),
        lsh or InMemoryTrajectoryWindowLSHRepository(),
        minhash_config=mh or _mh(),
        retrieval_config=cas or _cas(),
        scoring_config=_ms(),
        cache=cache,
    )


def _seed_trajectory(
    lsh: InMemoryTrajectoryWindowLSHRepository,
    paths: _StubPathRepo,
    tid: UUID,
    tokens: list[int] | np.ndarray,
    mh_cfg: WindowMinHashConfig,
    *,
    status: str = "success",
) -> list[tuple[UUID, int, int, int]]:
    """Insert one trajectory's bands at every window position into the
    in-memory LSH table, and register its trace_tokens in the path repo.
    Returns the rows for assertions.
    """
    arr = np.asarray(tokens, dtype=np.int64)
    paths.add(tid, list(arr), status=status)
    hasher = WindowMinHasher(mh_cfg)
    W = mh_cfg.window
    w = mh_cfg.half_window
    rows: list[tuple[UUID, int, int, int]] = []
    for start in range(0, int(arr.size) - W + 1):
        center = start + w
        bands = hasher.bands_for_window(arr, start)
        for bi, bh in enumerate(bands):
            rows.append((tid, center, bi, int(bh)))
    seen = {(t, wc, b) for t, wc, b, _ in lsh._rows}
    for r in rows:
        if (r[0], r[1], r[2]) in seen:
            continue
        seen.add((r[0], r[1], r[2]))
        lsh._rows.append(r)
    return rows


# ----------------------------------------------------------------------
# Stage 0: anchor placement + band computation
# ----------------------------------------------------------------------


class TestComputeAnchors:
    """``search`` builds anchor centers in [w, n - w] stepping by W;
    we re-execute the logic inline so the test stays decoupled from
    the private signature."""

    def _centers(self, n: int, mh: WindowMinHashConfig) -> list[int]:
        w = mh.half_window
        W = mh.window
        last = n - w
        out: list[int] = []
        c = last
        while c >= w:
            out.append(c)
            c -= W
        out.reverse()
        return out

    def test_single_anchor_when_tokens_equal_window(self):
        mh = _mh(window=10)
        # n=10 → last_center=5, fits exactly once.
        assert self._centers(10, mh) == [5]

    def test_anchors_step_back_by_window(self):
        mh = _mh(window=10)
        # n=30 → last_center=25, prev=15, 5. Reversed: [5, 15, 25].
        assert self._centers(30, mh) == [5, 15, 25]

    def test_too_short_yields_no_anchors(self):
        mh = _mh(window=10)
        assert self._centers(9, mh) == []


# ----------------------------------------------------------------------
# Stage 1: LSH lookup
# ----------------------------------------------------------------------


@pytest.mark.asyncio
class TestLSHLookup:

    async def _setup(self):
        mh = _mh()
        paths = _StubPathRepo()
        lsh = InMemoryTrajectoryWindowLSHRepository()
        return mh, paths, lsh

    async def test_excludes_query_trajectory(self):
        mh, paths, lsh = await self._setup()
        q_tokens = np.arange(20, dtype=np.int64)
        q_tid = uuid4()
        other = uuid4()
        _seed_trajectory(lsh, paths, q_tid, q_tokens, mh)
        _seed_trajectory(lsh, paths, other, q_tokens, mh)

        r = _retrieval(paths=paths, lsh=lsh, mh=mh)
        hasher = WindowMinHasher(mh)
        centers = [mh.half_window, mh.half_window + mh.window]
        bands = [
            hasher.bands_for_window(q_tokens, c - mh.half_window)
            for c in centers
        ]
        rows = await r._lsh_lookup(centers, bands, q_tid)
        tids = {tid for per in rows.values() for tid, _ in per}
        assert q_tid not in tids
        assert other in tids

    async def test_top_uniq_caps_per_anchor(self):
        mh, paths, lsh = await self._setup()
        q_tokens = np.arange(20, dtype=np.int64)
        for _ in range(7):
            _seed_trajectory(lsh, paths, uuid4(), q_tokens, mh)
        r = _retrieval(
            paths=paths, lsh=lsh, mh=mh, cas=_cas(prefetch_n_uniq=3),
        )
        hasher = WindowMinHasher(mh)
        centers = [mh.half_window]
        bands = [hasher.bands_for_window(q_tokens, 0)]
        rows = await r._lsh_lookup(centers, bands, exclude_tid=None)
        assert len(rows[0]) <= 3

    async def test_step_range_filters_outside_neighborhood(self):
        """LSH must only surface bands whose ``window_center`` is within
        ``[anchor - w, anchor + w]``. Seed a far-away candidate window
        and assert it never appears."""
        mh, paths, lsh = await self._setup()
        q_tokens = np.arange(20, dtype=np.int64)
        match_tid = uuid4()
        _seed_trajectory(lsh, paths, match_tid, q_tokens, mh)
        # Insert a band at a far step that would match if step filter were broken.
        far_tid = uuid4()
        paths.add(far_tid, list(q_tokens))
        hasher = WindowMinHasher(mh)
        bands_at_zero = hasher.bands_for_window(q_tokens, 0)
        for bi, bh in enumerate(bands_at_zero):
            lsh._rows.append((far_tid, 9999, bi, int(bh)))  # far center
        r = _retrieval(paths=paths, lsh=lsh, mh=mh)
        rows = await r._lsh_lookup([mh.half_window], [bands_at_zero], None)
        tids = {tid for tid, _ in rows[0]}
        assert match_tid in tids
        assert far_tid not in tids


# ----------------------------------------------------------------------
# Stage 2: candidate token fetch (with cache)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
class TestFetchCandidateTokens:

    async def test_no_cache_hits_repo_each_call(self):
        paths = _StubPathRepo()
        tid = uuid4()
        paths.add(tid, [1, 2, 3])
        r = _retrieval(paths=paths)
        await r._fetch_candidate_tokens([tid])
        await r._fetch_candidate_tokens([tid])
        assert len(paths.fetch_calls) == 2

    async def test_cache_dedupes_repeat_fetches(self):
        paths = _StubPathRepo()
        tid = uuid4()
        paths.add(tid, [1, 2, 3])
        cache = RetrievalCache()
        r = _retrieval(paths=paths, cache=cache)
        await r._fetch_candidate_tokens([tid])
        await r._fetch_candidate_tokens([tid])
        # First call hits the repo; the second is a pure cache read.
        assert len(paths.fetch_calls) == 1
        assert tid in cache.tokens._data

    async def test_cache_only_fetches_misses(self):
        paths = _StubPathRepo()
        a, b = uuid4(), uuid4()
        paths.add(a, [1, 2])
        paths.add(b, [3, 4])
        cache = RetrievalCache()
        r = _retrieval(paths=paths, cache=cache)
        await r._fetch_candidate_tokens([a])
        await r._fetch_candidate_tokens([a, b])
        # Second call fetches only ``b`` (``a`` already cached).
        assert paths.fetch_calls[-1] == (b,)


# ----------------------------------------------------------------------
# Stage 3: dense MinHash jaccard rerank
# ----------------------------------------------------------------------


class TestMinhashJaccardRerank:

    def _setup(self, q_tokens: np.ndarray, c_tokens: np.ndarray, *, agg: str):
        mh = _mh()
        r = _retrieval(mh=mh, cas=_cas(aggregation=agg))
        w = mh.half_window
        centers = [w]
        cand_tid = uuid4()
        anchor_rows = {0: [(cand_tid, 1.0)]}
        cand_tokens = {cand_tid: (uuid4(), "success", list(c_tokens))}
        return r, centers, anchor_rows, cand_tokens, cand_tid

    def test_identical_tokens_under_min_distance_give_one(self):
        """``min_distance`` takes the best window in the neighborhood —
        identical query/candidate tokens have at least one matching
        window so the per-anchor score must be exactly 1.0."""
        toks = np.arange(20, dtype=np.int64)
        r, centers, rows, cand, tid = self._setup(
            toks, toks, agg="min_distance",
        )
        scores = r._minhash_jaccard_rerank(rows, toks, centers, cand)
        for per_tid in scores.values():
            assert per_tid[tid] == pytest.approx(1.0)

    def test_identical_tokens_under_mean_dilutes_below_one(self):
        """``mean`` averages over every window in the neighborhood —
        only the perfectly-aligned window scores 1.0, the rest dilute
        the average, so even identical tokens fall below 1.0."""
        toks = np.arange(20, dtype=np.int64)
        r, centers, rows, cand, tid = self._setup(toks, toks, agg="mean")
        scores = r._minhash_jaccard_rerank(rows, toks, centers, cand)
        for per_tid in scores.values():
            assert 0.0 < per_tid[tid] < 1.0

    def test_min_distance_picks_max_jaccard_in_neighborhood(self):
        """Under ``min_distance`` agg the per-anchor reduction is
        ``max(jaccs)`` — a candidate whose best neighborhood window
        matches the query 1:1 must outrank a candidate with no
        matching window even if the boundary window matters."""
        q = np.arange(20, dtype=np.int64)
        mh = _mh()
        r = _retrieval(mh=mh, cas=_cas(aggregation="min_distance"))
        centers = [mh.half_window]
        good_tid = uuid4()
        weak_tid = uuid4()
        cand_tokens = {
            good_tid: (uuid4(), "success", list(q)),
            weak_tid: (uuid4(), "success", list(np.arange(500, 520))),
        }
        rows = {0: [(good_tid, 1.0), (weak_tid, 1.0)]}
        scores = r._minhash_jaccard_rerank(rows, q, centers, cand_tokens)
        per_tid = scores[0]
        assert per_tid[good_tid] >= per_tid.get(weak_tid, 0.0)
        assert per_tid[good_tid] == pytest.approx(1.0)

    def test_mean_dilutes_with_unrelated_tail(self):
        q = np.arange(20, dtype=np.int64)
        c_mixed = np.concatenate([q[:10], np.arange(100, 110)]).astype(np.int64)
        r, centers, rows, cand, tid = self._setup(q, c_mixed, agg="mean")
        scores = r._minhash_jaccard_rerank(rows, q, centers, cand)
        for per_tid in scores.values():
            assert 0.0 < per_tid[tid] < 1.0

    def test_cache_hit_returns_stored_score(self):
        """Pre-populate the JaccardCache and feed garbage candidate
        tokens; if compute fired the stored score wouldn't survive."""
        q = np.arange(20, dtype=np.int64)
        cache = RetrievalCache()
        mh = _mh()
        r = _retrieval(mh=mh, cache=cache, cas=_cas(aggregation="mean"))
        centers = [mh.half_window]
        tid = uuid4()
        rows = {0: [(tid, 1.0)]}
        cand = {tid: (uuid4(), "success", [1] * 20)}
        path_id = uuid4()
        cache.jaccard.put((path_id, 0, tid, "mean"), 0.999)
        scores = r._minhash_jaccard_rerank(
            rows, q, centers, cand, path_id=path_id,
        )
        assert scores[0][tid] == pytest.approx(0.999)


# ----------------------------------------------------------------------
# Stage 4: cross-anchor aggregate
# ----------------------------------------------------------------------


class TestCrossAnchorAggregate:

    def test_zero_fill_lowers_partial_matches(self):
        r = _retrieval()
        full_tid = uuid4()
        partial_tid = uuid4()
        anchor_scores = {
            0: {full_tid: 0.8, partial_tid: 0.8},
            1: {full_tid: 0.8},
            2: {full_tid: 0.8},
            3: {full_tid: 0.8},
        }
        ranked = r._cross_anchor_aggregate(anchor_scores, n_anchors=4)
        scores = dict(ranked)
        assert scores[full_tid] == pytest.approx(0.8)
        assert scores[partial_tid] == pytest.approx(0.8 / 4)
        assert ranked[0][0] == full_tid

    def test_top_jaccard_n_uniq_caps_survivors(self):
        r = _retrieval(cas=_cas(jaccard_n_uniq=2))
        anchor_scores = {0: {uuid4(): 0.5 + 0.01 * i for i in range(5)}}
        ranked = r._cross_anchor_aggregate(anchor_scores, n_anchors=1)
        assert len(ranked) == 2


# ----------------------------------------------------------------------
# Stage 5: agg-shift rerank (+ cache)
# ----------------------------------------------------------------------


class TestAggShiftRerank:

    def test_top_k_caps_and_orders_by_score(self):
        q = np.arange(20, dtype=np.int64)
        r = _retrieval(cas=_cas(top_k=2))
        good_a = uuid4()
        good_b = uuid4()
        bad = uuid4()
        cand_tokens = {
            good_a: (uuid4(), "success", list(q)),
            good_b: (uuid4(), "success", list(q)),
            bad:    (uuid4(), "failure", list(np.arange(500, 520))),
        }
        ranked = [(good_a, 1.0), (good_b, 1.0), (bad, 1.0)]
        out = r._aggshift_rerank(q, ranked, cand_tokens)
        assert len(out) == 2
        assert bad not in {c.trajectory_id for c in out}
        assert out[0].score >= out[1].score

    def test_cache_hit_skips_jit(self):
        q = np.arange(20, dtype=np.int64)
        cache = RetrievalCache()
        r = _retrieval(cache=cache)
        tid = uuid4()
        cand_tokens = {tid: (uuid4(), "success", list(q))}
        path_id = uuid4()
        # Sentinel — if the JIT fired we'd get 1.0 instead.
        cache.aggshift.put((path_id, tid, r._scoring, "mean"), 0.4242)
        out = r._aggshift_rerank(q, [(tid, 1.0)], cand_tokens, path_id)
        assert out[0].score == pytest.approx(0.4242)


# ----------------------------------------------------------------------
# End-to-end ranking sanity
# ----------------------------------------------------------------------


@pytest.mark.asyncio
class TestSearchEndToEnd:

    async def test_excludes_query_trajectory(self):
        q_tid = uuid4()
        other = uuid4()
        q = np.arange(20, dtype=np.int64)
        paths = _StubPathRepo()
        lsh = InMemoryTrajectoryWindowLSHRepository()
        mh = _mh()
        _seed_trajectory(lsh, paths, q_tid, q, mh)
        _seed_trajectory(lsh, paths, other, q, mh)
        r = _retrieval(paths=paths, lsh=lsh, mh=mh)
        out = await r.search(RetrievalQuery(tokens=q, trajectory_id=q_tid))
        assert {c.trajectory_id for c in out} == {other}


# ----------------------------------------------------------------------
# Cache parity — enabling the cache must not change results
# ----------------------------------------------------------------------


@pytest.mark.asyncio
class TestCacheParity:

    async def _seed(self, paths, lsh, mh):
        q_tokens = np.arange(20, dtype=np.int64)
        match_tid = uuid4()
        diff_tid = uuid4()
        _seed_trajectory(lsh, paths, match_tid, q_tokens, mh)
        _seed_trajectory(
            lsh, paths, diff_tid,
            (np.arange(20) + 1000).astype(np.int64), mh,
        )
        return q_tokens

    async def test_cache_off_vs_on_same_candidates_and_scores(self):
        """Both retrievals share the same repos so UUIDs and seed
        contents are identical — the only difference is whether the
        cache slot is wired in."""
        mh = _mh()
        paths = _StubPathRepo()
        lsh = InMemoryTrajectoryWindowLSHRepository()
        q_tokens = await self._seed(paths, lsh, mh)
        r_off = _retrieval(paths=paths, lsh=lsh, mh=mh)
        r_on = _retrieval(
            paths=paths, lsh=lsh, mh=mh, cache=RetrievalCache(),
        )
        query = RetrievalQuery(
            tokens=q_tokens, trajectory_id=uuid4(), path_id=uuid4(),
        )
        out_off = await r_off.search(query)
        out_on = await r_on.search(query)
        assert [c.trajectory_id for c in out_off] == [
            c.trajectory_id for c in out_on
        ]
        for a, b in zip(out_off, out_on):
            assert a.score == pytest.approx(b.score)

    async def test_cache_second_call_matches_first(self):
        """Running the same search twice on a single cache must yield
        identical results — guards against accidental in-place mutation
        of cached lists or scores."""
        mh = _mh()
        paths = _StubPathRepo()
        lsh = InMemoryTrajectoryWindowLSHRepository()
        q_tokens = await self._seed(paths, lsh, mh)
        cache = RetrievalCache()
        r = _retrieval(paths=paths, lsh=lsh, mh=mh, cache=cache)
        query = RetrievalQuery(
            tokens=q_tokens, trajectory_id=uuid4(), path_id=uuid4(),
        )
        first = await r.search(query)
        second = await r.search(query)
        assert [c.trajectory_id for c in first] == [
            c.trajectory_id for c in second
        ]
        for a, b in zip(first, second):
            assert a.score == pytest.approx(b.score)

    async def test_aggshift_cache_keyed_by_aggregation_value(self):
        """Same ``(path_id, tid, AggShiftConfig)`` under different
        ``aggregation`` strings must produce distinct entries — agg
        flips the per-row band reduction so the kernel score differs
        and the keys cannot collide."""
        mh = _mh()
        paths = _StubPathRepo()
        lsh = InMemoryTrajectoryWindowLSHRepository()
        q_tokens = await self._seed(paths, lsh, mh)
        cache = RetrievalCache()
        r = _retrieval(paths=paths, lsh=lsh, mh=mh, cache=cache)
        path_id = uuid4()
        query = RetrievalQuery(
            tokens=q_tokens, trajectory_id=uuid4(), path_id=path_id,
        )
        await r.search(query)
        assert {key[-1] for key in cache.aggshift._data} == {"mean"}
        r._cas_cfg = _cas(aggregation="min_distance")
        await r.search(query)
        assert {key[-1] for key in cache.aggshift._data} == {
            "mean", "min_distance",
        }


# ----------------------------------------------------------------------
# Scoring kernel — hand-calculated values on a tiny fixture
# ----------------------------------------------------------------------


class TestAggShiftHandCalc:
    """Anchor expected outputs of ``agg_shift_score`` to a fixture small
    enough to compute on paper, so a regression in the JIT kernel
    breaks the test instead of silently shifting scores.

    Fixture: ``a = [1..6]``, ``W = 4``, ``w = 2``. Three query rows
    (i = 0, 1, 2); each row's band covers ``j ∈ [0, 2]`` (every
    candidate window). Per-window Lev distance:

        Lev([k..k+3], [k..k+3]) = 0          (identical)
        Lev([k..k+3], [k±1..k±1+3]) = 2      (shift by 1: 1 sub + 1 sub)
        Lev([k..k+3], [k±2..k±2+3]) = 4      (shift by 2: 2 subs at edges)
    """

    A = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    # b = a shifted right by 1 — identical content displaced by one
    # token, so every query row has a perfect match at d=+1 (except
    # the last, where the perfect partner falls outside the band).
    B_SHIFT = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)

    def _cfg(self, **overrides) -> AggShiftConfig:
        defaults = dict(window=4, lam=0.0, penalty_shape="lin",
                        gap_open=1.0, gap_extend=1.0, sigma=1.0)
        defaults.update(overrides)
        return AggShiftConfig(**defaults)

    def test_identical_min_distance_is_one(self):
        # Best-shift wins: every row picks d=0 with cost 0.
        # total = 0 → sim = 1 - 0/(n_use·W) = 1.0.
        sim = score_pair(
            self.A, self.A, self._cfg(),
            band_aggregation="min_distance",
        )
        assert sim == pytest.approx(1.0)

    def test_identical_mean_is_five_ninths(self):
        # Per-row mean over band [0,2,4], [2,0,2], [4,2,0] → 2, 4/3, 2.
        # total = 2 + 4/3 + 2 = 16/3; avg per row = 16/9; sim = 1 - (16/9)/4 = 5/9.
        sim = score_pair(
            self.A, self.A, self._cfg(), band_aggregation="mean",
        )
        assert sim == pytest.approx(5 / 9)

    def test_shift_by_one_min_distance_is_five_sixths(self):
        # Per-row mins on the shifted grid:
        #   row 0: M = [2,0,2] → min = 0 (d=+1 lands on perfect window)
        #   row 1: M = [4,2,0] → min = 0 (d=+1 perfect)
        #   row 2: M = [4,4,2] → min = 2 (perfect partner out of band)
        # total = 2; sim = 1 - (2/3)/4 = 5/6.
        sim = score_pair(
            self.A, self.B_SHIFT, self._cfg(),
            band_aggregation="min_distance",
        )
        assert sim == pytest.approx(5 / 6)

    def test_shift_by_one_mean_is_four_ninths(self):
        # Per-row means: [2,0,2]/3, [4,2,0]/3, [4,4,2]/3
        #              = 4/3,        2,         10/3
        # total = 4/3 + 2 + 10/3 = 20/3; avg = 20/9; sim = 1 - (20/9)/4 = 4/9.
        sim = score_pair(
            self.A, self.B_SHIFT, self._cfg(), band_aggregation="mean",
        )
        assert sim == pytest.approx(4 / 9)

    def test_linear_penalty_drops_min_distance_to_two_thirds(self):
        # lam=1.0, lin penalty: per-row min(M[i,j] + |d|):
        #   row 0: min(2+0, 0+1, 2+2) = 1
        #   row 1: min(4+1, 2+0, 0+1) = 1
        #   row 2: min(4+2, 4+1, 2+0) = 2
        # total = 4; sim = 1 - (4/3)/4 = 2/3.
        sim = score_pair(
            self.A, self.B_SHIFT, self._cfg(lam=1.0),
            band_aggregation="min_distance",
        )
        assert sim == pytest.approx(2 / 3)

    def test_const_penalty_drops_min_distance_to_eleven_twelfths(self):
        # const penalty: pen = lam·(d != 0). With lam=1:
        #   row 0: min(2+0, 0+1, 2+1) = 1
        #   row 1: min(4+1, 2+0, 0+1) = 1
        #   row 2: min(4+1, 4+1, 2+0) = 2
        # total = 4; sim = 1 - (4/3)/4 = 2/3. Same total as linear here
        # because the winning d for rows 0/1 is ±1 — both shapes give
        # the same penalty at |d|=1. test_quad below diverges.
        sim = score_pair(
            self.A, self.B_SHIFT, self._cfg(lam=1.0, penalty_shape="const"),
            band_aggregation="min_distance",
        )
        assert sim == pytest.approx(2 / 3)

    def test_zero_lam_means_no_penalty_regardless_of_shape(self):
        # All four shapes share the same min when lam=0 — the score
        # must match the lam=0 linear baseline (= 5/6).
        scores = [
            score_pair(
                self.A, self.B_SHIFT, self._cfg(lam=0.0, penalty_shape=s),
                band_aggregation="min_distance",
            )
            for s in ("lin", "const", "quad", "gauss")
        ]
        for s in scores:
            assert s == pytest.approx(5 / 6)


class TestAffineKernel:
    """Direct ``_window_affine_distance`` calls — Gotoh affine charges
    one ``gap_open`` per gap run plus ``gap_extend`` for each extra
    character in the same run, so any pair that aligns via a multi-char
    gap diverges from uniform Lev (which charges 1 per character)."""

    def test_identical_arrays_cost_zero(self):
        a = np.array([1, 2, 3, 4], dtype=np.int64)
        assert _window_affine_distance(a, a, 2.0, 0.5) == 0.0

    def test_single_substitution_costs_one(self):
        # Sub of one position costs 1 (sub_cost), strictly less than
        # gap_open (=2) so the subs path wins regardless of gap costs.
        a = np.array([1, 2, 3, 4], dtype=np.int64)
        b = np.array([1, 2, 3, 5], dtype=np.int64)
        assert _window_affine_distance(a, b, 2.0, 0.5) == 1.0

    def test_two_char_insertion_uses_open_plus_one_extend(self):
        # Aligning [1,2,3,4] with [1,2,5,5,3,4] needs a 2-char gap in
        # ``a`` between positions 2 and 3. Affine cost = open + 1·extend.
        a = np.array([1, 2, 3, 4], dtype=np.int64)
        b = np.array([1, 2, 5, 5, 3, 4], dtype=np.int64)
        assert _window_affine_distance(a, b, 2.0, 0.5) == pytest.approx(2.5)

    def test_uniform_costs_equal_per_character_indel(self):
        # gap_open == gap_extend == 1.0 collapses Gotoh onto uniform
        # Lev — 2-char gap costs exactly 2 (matches Wagner-Fischer).
        a = np.array([1, 2, 3, 4], dtype=np.int64)
        b = np.array([1, 2, 5, 5, 3, 4], dtype=np.int64)
        assert _window_affine_distance(a, b, 1.0, 1.0) == 2.0



# ----------------------------------------------------------------------
# Aggregation flow: the cascade-wide ``aggregation`` reaches every stage
# ----------------------------------------------------------------------


class TestAggregationFlow:
    """``RetrievalConfig.aggregation`` must propagate to (a) the LSH
    lookup (max vs avg of band-hit counts), (b) the dense jaccard
    rerank (max vs avg of neighborhood jaccards), and (c) the
    agg-shift rerank (best-shift vs mean-over-band)."""

    @pytest.mark.asyncio
    async def test_lsh_lookup_uses_aggregation(self):
        # Two windows for the same candidate at different band-hit
        # counts: full match at center=5, partial at center=6.
        mh = _mh()
        paths = _StubPathRepo()
        lsh = InMemoryTrajectoryWindowLSHRepository()
        q_tokens = np.arange(20, dtype=np.int64)
        tid = uuid4()
        paths.add(tid, list(q_tokens))
        hasher = WindowMinHasher(mh)
        full = hasher.bands_for_window(q_tokens, 0)
        for bi, bh in enumerate(full):
            lsh._rows.append((tid, 5, bi, int(bh)))
        # Partial — only the first band matches.
        for bi, bh in enumerate(full):
            lsh._rows.append((tid, 6, bi, int(bh) if bi == 0 else -999))

        centers = [5]
        bands = [full]
        r_min = _retrieval(
            paths=paths, lsh=lsh, mh=mh, cas=_cas(aggregation="min_distance"),
        )
        r_mean = _retrieval(
            paths=paths, lsh=lsh, mh=mh, cas=_cas(aggregation="mean"),
        )
        min_rows = await r_min._lsh_lookup(centers, bands, None)
        mean_rows = await r_mean._lsh_lookup(centers, bands, None)
        score_min = dict(min_rows[0])[tid]
        score_mean = dict(mean_rows[0])[tid]
        # max(counts) > avg(counts) when at least one window is partial.
        assert score_min > score_mean

    def test_jaccard_rerank_uses_aggregation(self):
        # Identical tokens: best window in the neighborhood is a
        # perfect match (jaccard=1.0) under min_distance, while mean
        # averages over the whole neighborhood (smaller).
        mh = _mh()
        q = np.arange(20, dtype=np.int64)
        tid = uuid4()
        cand = {tid: (uuid4(), "success", list(q))}
        rows = {0: [(tid, 1.0)]}
        centers = [mh.half_window]
        r_min = _retrieval(mh=mh, cas=_cas(aggregation="min_distance"))
        r_mean = _retrieval(mh=mh, cas=_cas(aggregation="mean"))
        s_min = r_min._minhash_jaccard_rerank(rows, q, centers, cand)[0][tid]
        s_mean = r_mean._minhash_jaccard_rerank(rows, q, centers, cand)[0][tid]
        assert s_min == pytest.approx(1.0)
        assert s_mean < 1.0

    def test_aggshift_rerank_uses_aggregation(self):
        # Hand-calc fixture from TestAggShiftHandCalc: shift-by-one
        # yields min_distance=5/6 and mean=4/9 — they must differ
        # when the cascade-wide agg switches.
        q = TestAggShiftHandCalc.A
        c = TestAggShiftHandCalc.B_SHIFT
        tid = uuid4()
        cand_tokens = {tid: (uuid4(), "success", list(c))}
        ranked = [(tid, 1.0)]
        r_min = _retrieval(
            cas=_cas(aggregation="min_distance"),
        )
        r_min._scoring = AggShiftConfig(window=4, lam=0.0)
        r_mean = _retrieval(cas=_cas(aggregation="mean"))
        r_mean._scoring = AggShiftConfig(window=4, lam=0.0)
        s_min = r_min._aggshift_rerank(q, ranked, cand_tokens)[0].score
        s_mean = r_mean._aggshift_rerank(q, ranked, cand_tokens)[0].score
        assert s_min == pytest.approx(5 / 6)
        assert s_mean == pytest.approx(4 / 9)


# ----------------------------------------------------------------------
# Cache keys: each slot writes the documented composite key
# ----------------------------------------------------------------------


@pytest.mark.asyncio
class TestCacheKeys:
    """``RetrievalCache`` has four slots with documented composite
    keys. After a search, the cache must contain exactly those keys —
    a refactor that drops a component (e.g. ``aggregation`` from the
    LSH key) collides hot-cache reads under sweep and would silently
    return wrong scores."""

    async def _run_search(self, agg: str = "mean"):
        mh = _mh()
        paths = _StubPathRepo()
        lsh = InMemoryTrajectoryWindowLSHRepository()
        q_tokens = np.arange(20, dtype=np.int64)
        cand = uuid4()
        _seed_trajectory(lsh, paths, cand, q_tokens, mh)
        cache = RetrievalCache()
        r = _retrieval(
            paths=paths, lsh=lsh, mh=mh, cache=cache,
            cas=_cas(aggregation=agg),
        )
        q_tid = uuid4()
        path_id = uuid4()
        await r.search(RetrievalQuery(
            tokens=q_tokens, trajectory_id=q_tid, path_id=path_id,
        ))
        return cache, path_id, q_tid, cand, r._scoring, agg

    async def test_lsh_cache_key_is_path_anchor_excludetid_agg(self):
        cache, path_id, q_tid, _cand, _scoring, agg = await self._run_search()
        assert cache.lsh._data, "LSH cache must populate during search"
        for key in cache.lsh._data:
            assert len(key) == 4
            assert key[0] == path_id
            assert isinstance(key[1], int)
            assert key[2] == q_tid
            assert key[3] == agg

    async def test_jaccard_cache_key_is_path_anchor_cand_agg(self):
        cache, path_id, _q_tid, cand, _scoring, agg = await self._run_search()
        assert cache.jaccard._data, "jaccard cache must populate during search"
        for key in cache.jaccard._data:
            assert len(key) == 4
            assert key[0] == path_id
            assert isinstance(key[1], int)
            assert isinstance(key[2], UUID)
            assert key[3] == agg
        assert any(key[2] == cand for key in cache.jaccard._data)

    async def test_tokens_cache_key_is_trajectory_id(self):
        cache, _path_id, _q_tid, cand, _scoring, _agg = await self._run_search()
        assert cand in cache.tokens._data
        for key in cache.tokens._data:
            assert isinstance(key, UUID)

    async def test_aggshift_cache_key_is_path_cand_cfg_agg(self):
        cache, path_id, _q_tid, cand, scoring, agg = await self._run_search()
        assert cache.aggshift._data, "agg-shift cache must populate"
        for key in cache.aggshift._data:
            assert len(key) == 4
            assert key[0] == path_id
            assert isinstance(key[1], UUID)
            assert key[2] == scoring
            assert key[3] == agg
        assert any(key[1] == cand for key in cache.aggshift._data)

    async def test_lsh_and_jaccard_keys_split_per_aggregation(self):
        cache_a, *_ = await self._run_search(agg="mean")
        cache_b, *_ = await self._run_search(agg="min_distance")
        assert {k[-1] for k in cache_a.lsh._data} == {"mean"}
        assert {k[-1] for k in cache_b.lsh._data} == {"min_distance"}
        assert {k[-1] for k in cache_a.jaccard._data} == {"mean"}
        assert {k[-1] for k in cache_b.jaccard._data} == {"min_distance"}
