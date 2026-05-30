"""TokenizerPipeline: ActObsBuilder → Clusterer → TokenSaver."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.clustering.clusterer import ClusterResult, Clusterer
from episodiq.clustering.constants import Params
from episodiq.clustering.tokenizer.act_obs_builder import ActObsBuilder
from episodiq.clustering.tokenizer.constants import DEFAULT_PARAMS
from episodiq.clustering.tokenizer.saver import TokenSaver
from episodiq.storage.postgres.repository import (
    ClusterRepository,
    TokenClusterRepository,
    TokenMappingRepository,
    TrajectoryPathRepository,
)

logger = logging.getLogger(__name__)


class TokenizerPipeline:
    """Build act_obs pool → run UMAP+HDBSCAN → persist token_clusters + mapping."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        params: Params = DEFAULT_PARAMS,
    ):
        self._session_factory = session_factory
        self._params = params

    async def run(self, *, dry_run: bool = False) -> ClusterResult:
        async with self._session_factory() as session:
            builder = ActObsBuilder(
                TrajectoryPathRepository(session),
                ClusterRepository(session),
            )
            pool = await builder.build()

        if pool.embs.shape[0] == 0:
            logger.warning("no act_obs entries; nothing to cluster")
            return ClusterResult(labels=[], noise_count=0, dbcv=-1.0)

        result = Clusterer(self._params).fit(pool.embs)
        logger.info(
            "tokenizer: n_clusters=%d noise=%d dbcv=%.3f entropy=%.3f",
            result.n_clusters, result.noise_count, result.dbcv, result.entropy,
        )

        if dry_run:
            return result

        async with self._session_factory() as session:
            saver = TokenSaver(
                TokenClusterRepository(session),
                TokenMappingRepository(session),
            )
            await saver.save(pool, result)
            await session.commit()

        return result
