"""Unit tests for ``TokenizerGridSearch.run()`` — sweeps HDBSCAN/UMAP
``Params`` over a single act_obs pool. Focus: that the sweep
**dispatches** each ``Params`` instance into ``Clusterer(params)``
verbatim and calls ``fit(pool.embs)``. Pool building and the cluster
result arithmetic are exercised in their own test modules — here we
only pin orchestration.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from episodiq.clustering.clusterer import ClusterResult
from episodiq.clustering.constants import Params
from episodiq.clustering.tokenizer.grid_search import TokenizerGridSearch


@dataclass
class _Pool:
    embs: np.ndarray


def _result(labels: list[int], dbcv: float = 0.5) -> ClusterResult:
    return ClusterResult(
        labels=np.array(labels, dtype=np.int64),
        noise_count=sum(1 for label in labels if label == -1),
        dbcv=dbcv,
    )


def _session_factory():
    @asynccontextmanager
    async def _factory():
        yield MagicMock()
    return _factory


@pytest.mark.asyncio
class TestDispatch:
    """Each ``Params`` in the list must be forwarded into the
    ``Clusterer`` constructor verbatim (in order) and ``fit`` must run
    on ``pool.embs`` — the only inputs the sweep is responsible for."""

    async def test_clusterer_called_with_each_params_in_order(self):
        pool = _Pool(embs=np.ones((20, 4), dtype=np.float32))
        params_list = [
            Params(min_cluster_size=3, min_samples=2),
            Params(min_cluster_size=5, min_samples=3),
            Params(min_cluster_size=10, min_samples=5),
        ]

        with patch(
            "episodiq.clustering.tokenizer.grid_search.ActObsBuilder",
        ) as builder_cls, patch(
            "episodiq.clustering.tokenizer.grid_search.Clusterer",
        ) as clusterer_cls:
            builder_cls.return_value.build = AsyncMock(return_value=pool)
            clusterer_cls.return_value.fit.return_value = _result(
                [0, 0, 1, 1, -1] * 4,
            )

            sweep = TokenizerGridSearch(_session_factory(), params_list=params_list)
            await sweep.run()

        # Clusterer(...) called once per Params, in iteration order.
        ctor_args = [c.args[0] for c in clusterer_cls.call_args_list]
        assert ctor_args == params_list

    async def test_fit_called_with_pool_embs(self):
        embs = np.arange(40, dtype=np.float32).reshape(10, 4)
        pool = _Pool(embs=embs)

        with patch(
            "episodiq.clustering.tokenizer.grid_search.ActObsBuilder",
        ) as builder_cls, patch(
            "episodiq.clustering.tokenizer.grid_search.Clusterer",
        ) as clusterer_cls:
            builder_cls.return_value.build = AsyncMock(return_value=pool)
            fit = clusterer_cls.return_value.fit
            fit.return_value = _result([0] * 10)

            sweep = TokenizerGridSearch(
                _session_factory(),
                params_list=[Params(min_cluster_size=3)],
            )
            await sweep.run()

        # fit ran once, with the pool's embeddings (same array).
        fit.assert_called_once()
        passed = fit.call_args.args[0]
        assert passed is embs

    async def test_params_above_pool_size_are_skipped(self):
        """``min_cluster_size > n`` is an HDBSCAN-invalid combination —
        sweep skips it without invoking ``Clusterer``."""
        pool = _Pool(embs=np.ones((5, 4), dtype=np.float32))
        keep = Params(min_cluster_size=3)
        skip = Params(min_cluster_size=10)

        with patch(
            "episodiq.clustering.tokenizer.grid_search.ActObsBuilder",
        ) as builder_cls, patch(
            "episodiq.clustering.tokenizer.grid_search.Clusterer",
        ) as clusterer_cls:
            builder_cls.return_value.build = AsyncMock(return_value=pool)
            clusterer_cls.return_value.fit.return_value = _result([0, 0, 1, 1, -1])

            sweep = TokenizerGridSearch(
                _session_factory(), params_list=[keep, skip],
            )
            report = await sweep.run()

        # Only the in-bounds Params reached Clusterer.
        ctor_args = [c.args[0] for c in clusterer_cls.call_args_list]
        assert ctor_args == [keep]
        assert [e.params for e in report.entries] == [keep]

    async def test_empty_pool_short_circuits_before_clusterer(self):
        with patch(
            "episodiq.clustering.tokenizer.grid_search.ActObsBuilder",
        ) as builder_cls, patch(
            "episodiq.clustering.tokenizer.grid_search.Clusterer",
        ) as clusterer_cls:
            builder_cls.return_value.build = AsyncMock(return_value=_Pool(
                embs=np.zeros((0, 4), dtype=np.float32),
            ))
            sweep = TokenizerGridSearch(
                _session_factory(), params_list=[Params()],
            )
            report = await sweep.run()
            clusterer_cls.assert_not_called()
        assert report.entries == []


