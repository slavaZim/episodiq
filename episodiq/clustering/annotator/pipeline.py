"""AnnotationPipeline: orchestrates annotation + optional merge."""

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.api_adapters.base import Usage
from episodiq.clustering.annotator.annotator import (
    Annotation,
    AnnotatingJobSpec,
    ClusterAnnotator,
    resolve_annotation_jobs,
)
from episodiq.clustering.annotator.generator import Generator
from episodiq.clustering.annotator.saver import AnnotationSaver
from episodiq.clustering.annotator.summarizer import MapReduceSummarizer
from episodiq.inference.embedder import Embedder
from episodiq.storage.postgres.repository import (
    ClusterRepository,
    MessageRepository,
)

logger = logging.getLogger(__name__)


@dataclass
class AnnotationPipelineResult:
    results: list[Annotation]
    merged_count: int  # 0 in dry_run
    usage: Usage
    summarizer_usage: Usage


class AnnotationPipeline:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        generator: Generator,
        embedder: Embedder,
        *,
        summarizer: MapReduceSummarizer | None = None,
        n_examples: int = 5,
        merge_threshold: float = 0.85,
        workers: int = 1,
    ):
        self._session_factory = session_factory
        self._generator = generator
        self._embedder = embedder
        self._summarizer = summarizer
        self._n_examples = n_examples
        self._merge_threshold = merge_threshold
        self._workers = workers

    async def run(
        self,
        specs: list[AnnotatingJobSpec] | None = None,
        dry_run: bool = False,
    ) -> AnnotationPipelineResult:
        """Run annotation pipeline. With dry_run=True, returns results without DB writes."""
        async with self._session_factory() as session:
            message_repo = MessageRepository(session)
            cluster_repo = ClusterRepository(session)

            jobs = await resolve_annotation_jobs(message_repo, specs)

            annotator = ClusterAnnotator(
                message_repo=message_repo,
                cluster_repo=cluster_repo,
                session_factory=self._session_factory,
                generator=self._generator,
                embedder=self._embedder,
                summarizer=self._summarizer,
                n_examples=self._n_examples,
                merge_threshold=self._merge_threshold,
                workers=self._workers,
            )

            annotations = await annotator.annotate(jobs)
            usage = annotator.total_usage
            summarizer_usage = annotator.summarizer_usage

            logger.info(
                "annotation done: %d results, %d merges, %d input tokens, %d output tokens"
                " | summarizer: %d input, %d output",
                len(annotations),
                sum(1 for a in annotations if len(a.merged_ids) > 1),
                usage.input_tokens,
                usage.output_tokens,
                summarizer_usage.input_tokens,
                summarizer_usage.output_tokens,
            )

            merged_count = 0
            if not dry_run:
                saver = AnnotationSaver(cluster_repo, message_repo)
                merged_count = await saver.save(annotations)
                await session.commit()
                logger.info("merge done: %d clusters absorbed", merged_count)

        return AnnotationPipelineResult(
            results=annotations,
            merged_count=merged_count,
            usage=usage,
            summarizer_usage=summarizer_usage,
        )
