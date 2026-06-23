"""TokenizerGridSearch: sweep params on a single act_obs pool, no persistence."""

import logging
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.clustering.clusterer import Clusterer
from episodiq.clustering.constants import Params
from episodiq.clustering.tokenizer.act_obs_builder import ActObsBuilder
from episodiq.clustering.tokenizer.constants import DEFAULT_GRID
from episodiq.storage.postgres.repository import (
    ClusterRepository,
    MessageRepository,
    TrajectoryPathRepository,
)

logger = logging.getLogger(__name__)


@dataclass
class GridSearchEntry:
    params: Params
    n_clusters: int
    noise_count: int
    noise_ratio: float
    dbcv: float
    entropy: float
    score: float


@dataclass
class GridSearchReport:
    # No ``best`` selector: tokenizer params are chosen manually from
    # the entries table (operator inspects ``score`` / ``n_clusters`` /
    # ``noise_ratio`` / ``dbcv`` / ``entropy``). An auto-argmax would
    # encourage misuse since downstream callers don't pick by score.
    entries: list[GridSearchEntry] = field(default_factory=list)


class TokenizerGridSearch:
    """Try multiple Params combos on one act_obs pool; rank by (1-noise) * dbcv * entropy."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        params_list: list[Params] | None = None,
    ):
        self._session_factory = session_factory
        self._params_list = params_list or list(DEFAULT_GRID)

    async def run(self) -> GridSearchReport:
        async with self._session_factory() as session:
            builder = ActObsBuilder(
                TrajectoryPathRepository(session),
                ClusterRepository(session),
                MessageRepository(session),
            )
            pool = await builder.build()

        report = GridSearchReport()
        if pool.embs.shape[0] == 0:
            logger.warning("no act_obs entries; grid search empty")
            return report

        n = pool.embs.shape[0]
        for params in self._params_list:
            if n < params.min_cluster_size:
                continue
            cr = Clusterer(params).fit(pool.embs)
            noise_ratio = cr.noise_count / n
            score = (1 - noise_ratio) * cr.dbcv * cr.entropy
            report.entries.append(GridSearchEntry(
                params=params,
                n_clusters=cr.n_clusters,
                noise_count=cr.noise_count,
                noise_ratio=noise_ratio,
                dbcv=cr.dbcv,
                entropy=cr.entropy,
                score=score,
            ))
            logger.info(
                "params=%s n_clusters=%d noise=%.2f dbcv=%.3f entropy=%.3f score=%.3f",
                params, cr.n_clusters, noise_ratio, cr.dbcv, cr.entropy, score,
            )

        report.entries.sort(key=lambda e: (-e.score, -e.n_clusters))
        return report
