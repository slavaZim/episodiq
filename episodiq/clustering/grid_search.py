"""ClusterGridSearch: tries param combos per job, quantized selection."""

import logging
from dataclasses import dataclass, field

import numpy as np

from episodiq.clustering.clusterer import Clusterer
from episodiq.clustering.constants import DEFAULT_GRID, Params
from episodiq.clustering.manager import ClusteringJob, _resolve_categories
from episodiq.storage.postgres.repository import MessageRepository

logger = logging.getLogger(__name__)


@dataclass
class GridJobSpec:
    """Grid search spec: tries multiple param combos per category."""
    type: str          # "action" | "observation"
    category: str      # "text", specific tool name, or "tool" (discover all)
    params_list: list[Params] = field(default_factory=lambda: list(DEFAULT_GRID))
    max_clusters: int | None = None


_DEFAULT_GRID_SPECS = [
    GridJobSpec(type="action", category="text"),
    GridJobSpec(type="observation", category="text"),
    GridJobSpec(type="action", category="tool"),
    GridJobSpec(type="observation", category="tool"),
]


@dataclass
class GridSearchEntry:
    params: Params
    noise_count: int
    noise_ratio: float
    n_clusters: int
    dbcv: float
    entropy: float
    score: float


@dataclass
class GridSearchReport:
    entries: dict[str, list[GridSearchEntry]] = field(default_factory=dict)


def _select_winner(
    entries: list[GridSearchEntry],
    max_clusters: int | None,
    bucket_size: float,
) -> GridSearchEntry | None:
    """Quantized selection: top score bucket → max clusters."""
    candidates = entries
    if max_clusters is not None:
        candidates = [e for e in candidates if e.n_clusters <= max_clusters]
    if not candidates:
        return None

    def bucket(e: GridSearchEntry) -> int:
        return int(e.score / bucket_size)

    top_bucket = max(bucket(e) for e in candidates)
    in_bucket = [e for e in candidates if bucket(e) == top_bucket]

    return max(in_bucket, key=lambda e: e.n_clusters)


async def resolve_grid_jobs(
    repo: MessageRepository,
    specs: list[GridJobSpec],
) -> list[tuple[ClusteringJob, list[Params], int | None]]:
    """Resolve GridJobSpecs into (base_job, params_list, max_clusters) triples."""
    if not specs:
        specs = _DEFAULT_GRID_SPECS

    result: list[tuple[ClusteringJob, list[Params], int | None]] = []
    discovered: dict[str, list[str]] = {}

    for spec in specs:
        categories = await _resolve_categories(
            repo, spec.type, spec.category, discovered,
        )
        for cat in categories:
            job = ClusteringJob(type=spec.type, category=cat)
            result.append((job, spec.params_list, spec.max_clusters))

    return result


class ClusterGridSearch:
    """Try param combos per resolved job, pick best with quantized scoring."""

    def __init__(
        self,
        repo: MessageRepository,
        specs: list[GridJobSpec] = [],
        bucket_size: float = 0.05,
    ):
        self._repo = repo
        self._specs = specs
        self._bucket_size = bucket_size

    async def run(self) -> tuple[list[ClusteringJob], GridSearchReport]:
        grid_jobs = await resolve_grid_jobs(self._repo, self._specs)

        winners: list[ClusteringJob] = []
        report = GridSearchReport()

        for base_job, params_list, max_clusters in grid_jobs:
            key = f"{base_job.type}:{base_job.category}"

            messages = await self._repo.get_messages_for_clustering(
                base_job.type, base_job.category,
            )

            if not messages:
                continue

            vectors = np.array([m.embedding for m in messages], dtype=np.float32)
            n = len(vectors)

            entries: list[GridSearchEntry] = []
            for params in params_list:
                if n < params.min_cluster_size:
                    continue

                cr = Clusterer(params).fit(vectors)

                noise_ratio = cr.noise_count / n
                score = (1 - noise_ratio) * cr.dbcv * cr.entropy

                entries.append(GridSearchEntry(
                    params=params,
                    noise_count=cr.noise_count,
                    noise_ratio=noise_ratio,
                    n_clusters=cr.n_clusters,
                    dbcv=cr.dbcv,
                    entropy=cr.entropy,
                    score=score,
                ))

            report.entries[key] = sorted(entries, key=lambda e: (-e.score, -e.n_clusters))

            winner = _select_winner(entries, max_clusters, self._bucket_size)
            if winner:
                winners.append(ClusteringJob(
                    type=base_job.type,
                    category=base_job.category,
                    params=winner.params,
                ))

        return winners, report
