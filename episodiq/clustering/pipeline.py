"""Clustering pipelines: simple and grid-search."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.clustering.grid_search import ClusterGridSearch, GridJobSpec, GridSearchReport
from episodiq.clustering.manager import (
    CategoryResult,
    ClusteringManager,
    JobSpec,
    resolve_jobs,
)
from episodiq.clustering.saver import ClusterSaver
from episodiq.clustering.updater import MessageUpdater
from episodiq.storage.postgres.repository import (
    ClusterRepository,
    MessageRepository,
)

logger = logging.getLogger(__name__)


class ClusteringPipeline:
    """resolve_jobs → Manager → Saver → Updater."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        specs: list[JobSpec] = [],
    ):
        self._session_factory = session_factory
        self._specs = specs

    async def run(self, *, dry_run: bool = False) -> list[CategoryResult]:
        async with self._session_factory() as session:
            msg_repo = MessageRepository(session)
            jobs = await resolve_jobs(msg_repo, self._specs)
            manager = ClusteringManager(msg_repo, jobs)
            results = await manager.run()

        if dry_run:
            return results

        async with self._session_factory() as session:
            saver = ClusterSaver(ClusterRepository(session))
            assignments = await saver.save(results)

            updater = MessageUpdater(MessageRepository(session))
            await updater.update(assignments)

            await session.commit()

        return results


class GridSearchClusteringPipeline:
    """GridSearch only — returns report without applying winners."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        specs: list[GridJobSpec] = [],
        bucket_size: float = 0.05,
    ):
        self._session_factory = session_factory
        self._specs = specs
        self._bucket_size = bucket_size

    async def run(self) -> GridSearchReport:
        async with self._session_factory() as session:
            msg_repo = MessageRepository(session)
            grid = ClusterGridSearch(msg_repo, self._specs, self._bucket_size)
            _, report = await grid.run()
        return report
