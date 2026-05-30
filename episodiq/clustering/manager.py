"""ClusteringManager: loads messages by cluster_type + category, clusters each job."""

import logging
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np

from episodiq.clustering.clusterer import Clusterer
from episodiq.clustering.constants import DEFAULT_PARAMS, MAX_NOISE_RATIO, PREFIXES, Params
from episodiq.storage.postgres.repository import MessageRepository

logger = logging.getLogger(__name__)


@dataclass
class JobSpec:
    """Declarative spec: category="tool" auto-discovers all tool categories from DB."""
    type: str          # "action" | "observation"
    category: str      # "text", specific tool name, or "tool" (discover all)
    params: Params = field(default_factory=lambda: DEFAULT_PARAMS)


_DEFAULT_SPECS = [
    JobSpec(type="action", category="text"),
    JobSpec(type="observation", category="text"),
    JobSpec(type="action", category="tool"),
    JobSpec(type="observation", category="tool"),
]


@dataclass
class ClusteringJob:
    """One concrete clustering task: type + category + resolved params."""
    type: str
    category: str
    params: Params = field(default_factory=lambda: DEFAULT_PARAMS)


async def resolve_jobs(
    repo: MessageRepository,
    specs: list[JobSpec],
) -> list[ClusteringJob]:
    """Resolve JobSpecs into concrete ClusteringJobs.

    Empty specs → default (all types × text + discovered tools).
    category="tool" → discover all non-text categories from DB.
    """
    if not specs:
        specs = _DEFAULT_SPECS

    jobs: list[ClusteringJob] = []
    discovered: dict[str, list[str]] = {}

    for spec in specs:
        categories = await _resolve_categories(
            repo, spec.type, spec.category, discovered,
        )
        for cat in categories:
            jobs.append(ClusteringJob(
                type=spec.type, category=cat, params=spec.params,
            ))

    return jobs


async def _resolve_categories(
    repo: MessageRepository,
    cluster_type: str,
    category: str,
    cache: dict[str, list[str]],
) -> list[str]:
    """Resolve category="tool" into concrete tool names, or pass through."""
    if category == "tool":
        if cluster_type not in cache:
            cats = await repo.get_distinct_categories(cluster_type)
            cache[cluster_type] = [c for c in cats if c != "text"]
        return cache[cluster_type]
    return [category]


@dataclass
class CategoryResult:
    """Clustering result for one job: message_ids aligned with string labels."""
    type: str
    category: str
    message_ids: list[UUID] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    n_clusters: int = 0
    noise_count: int = 0
    noise_ratio: float = 0.0
    dbcv: float = 0.0
    entropy: float = 0.0
    score: float = 0.0


class ClusteringManager:
    """Run clustering jobs: load messages → cluster → produce labels."""

    def __init__(
        self,
        repo: MessageRepository,
        jobs: list[ClusteringJob],
    ):
        self._repo = repo
        self._jobs = jobs

    async def run(self) -> list[CategoryResult]:
        results: list[CategoryResult] = []

        for job in self._jobs:
            messages = await self._repo.get_messages_for_clustering(job.type, job.category)

            if not messages:
                continue

            prefix = PREFIXES[job.type]
            flat_label = f"{prefix}:{job.category}"
            vectors = np.array([m.embedding for m in messages], dtype=np.float32)

            # Too few messages — flat label for all
            if len(vectors) < job.params.min_cluster_size:
                result = CategoryResult(type=job.type, category=job.category)
                for msg in messages:
                    result.message_ids.append(msg.id)
                    result.labels.append(flat_label)
                results.append(result)
                logger.debug(
                    "flat type=%s category=%s count=%d (< min %d)",
                    job.type, job.category, len(vectors), job.params.min_cluster_size,
                )
                continue

            cr = Clusterer(job.params).fit(vectors)
            noise_ratio = cr.noise_count / len(vectors)

            # Single cluster — flat label (sub-cluster IDs are meaningless)
            if cr.n_clusters <= 1:
                result = CategoryResult(
                    type=job.type, category=job.category,
                    n_clusters=cr.n_clusters, noise_count=cr.noise_count,
                    noise_ratio=noise_ratio, dbcv=cr.dbcv, entropy=cr.entropy,
                )
                for msg in messages:
                    result.message_ids.append(msg.id)
                    result.labels.append(flat_label)
                results.append(result)
                logger.debug(
                    "flat type=%s category=%s n_clusters=%d",
                    job.type, job.category, cr.n_clusters,
                )
                continue

            # Too much noise — flat label for all
            if noise_ratio > MAX_NOISE_RATIO:
                result = CategoryResult(
                    type=job.type, category=job.category,
                    n_clusters=cr.n_clusters, noise_count=cr.noise_count,
                    noise_ratio=noise_ratio, dbcv=cr.dbcv, entropy=cr.entropy,
                    score=(1 - noise_ratio) * cr.dbcv * cr.entropy,
                )
                for msg in messages:
                    result.message_ids.append(msg.id)
                    result.labels.append(flat_label)
                results.append(result)
                logger.debug(
                    "flat type=%s category=%s noise=%.2f (> %.2f)",
                    job.type, job.category, noise_ratio, MAX_NOISE_RATIO,
                )
                continue

            score = (1 - noise_ratio) * cr.dbcv * cr.entropy
            result = CategoryResult(
                type=job.type, category=job.category,
                n_clusters=cr.n_clusters, noise_count=cr.noise_count,
                noise_ratio=noise_ratio, dbcv=cr.dbcv, entropy=cr.entropy,
                score=score,
            )
            for msg, lbl in zip(messages, cr.labels):
                result.message_ids.append(msg.id)
                label_id = "?" if lbl == -1 else str(lbl)
                result.labels.append(f"{flat_label}:{label_id}")

            results.append(result)
            logger.debug(
                "clustered type=%s category=%s n=%d clusters=%d noise=%d",
                job.type, job.category, len(vectors), cr.n_clusters, cr.noise_count,
            )

        return results
