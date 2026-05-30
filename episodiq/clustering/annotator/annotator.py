"""ClusterAnnotator: LLM-based contrastive annotation with embedding-based merge."""

import asyncio
import logging
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from episodiq.clustering.annotator.constants import API_CONCURRENCY, MERGE_THRESHOLD, get_prompt
from episodiq.clustering.annotator.generator import Generator, system_message, user_message
from episodiq.clustering.annotator.summarizer import MapReduceSummarizer
from episodiq.inference.embedder import Embedder
from episodiq.storage.postgres.models import Message
from episodiq.storage.postgres.repository import ClusterRepository, MessageRepository
from episodiq.utils import json_to_text, l2_normalize

logger = logging.getLogger(__name__)


# --- Job specs ---

@dataclass
class AnnotatingJobSpec:
    """Declarative spec: category="tool" auto-discovers all tool categories."""
    type: str       # "action" | "observation"
    category: str   # "text", specific tool name, or "tool" (discover all)


@dataclass
class AnnotatingJob:
    """One concrete annotation task: type + category."""
    type: str       # "action" | "observation"
    category: str   # resolved concrete category


_DEFAULT_ANNOTATION_SPECS = [
    AnnotatingJobSpec(type="action", category="text"),
    AnnotatingJobSpec(type="observation", category="text"),
    AnnotatingJobSpec(type="action", category="tool"),
    AnnotatingJobSpec(type="observation", category="tool"),
]


async def resolve_annotation_jobs(
    repo: MessageRepository,
    specs: list[AnnotatingJobSpec] | None = None,
) -> list[AnnotatingJob]:
    """Resolve AnnotatingJobSpecs into concrete AnnotatingJobs.

    Empty specs → default (all types × text + discovered tools).
    category="tool" → discover all non-text categories from DB.
    """
    if not specs:
        specs = _DEFAULT_ANNOTATION_SPECS

    jobs: list[AnnotatingJob] = []
    discovered: dict[str, list[str]] = {}

    for spec in specs:
        if spec.category == "tool":
            if spec.type not in discovered:
                cats = await repo.get_distinct_categories(spec.type)
                discovered[spec.type] = [c for c in cats if c != "text"]
            for cat in discovered[spec.type]:
                jobs.append(AnnotatingJob(type=spec.type, category=cat))
        else:
            jobs.append(AnnotatingJob(type=spec.type, category=spec.category))

    return jobs


# --- Annotation dataclass ---

@dataclass
class Annotation:
    """Working state and output of cluster annotation."""
    cluster_id: UUID
    type: str           # "action" | "observation"
    category: str       # "text", tool name
    label: str
    text: str | None = None
    merged_ids: set[UUID] = field(default_factory=set)
    embeddings: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        if not self.merged_ids:
            self.merged_ids = {self.cluster_id}

    @classmethod
    def merge(cls, annotations: list["Annotation"]) -> "Annotation":
        """Merge multiple annotations into one. First annotation is the base."""
        base = annotations[0]
        for other in annotations[1:]:
            base.merged_ids |= other.merged_ids
            base.embeddings.extend(other.embeddings)
        return base


# --- Annotator ---

class ClusterAnnotator:
    def __init__(
        self,
        *,
        message_repo: MessageRepository,
        cluster_repo: ClusterRepository,
        generator: Generator,
        embedder: Embedder,
        summarizer: MapReduceSummarizer | None = None,
        n_examples: int = 5,
        merge_threshold: float = MERGE_THRESHOLD,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        workers: int = API_CONCURRENCY,
    ):
        self._message_repo = message_repo
        self._cluster_repo = cluster_repo
        self._generator = generator
        self._embedder = embedder
        self._summarizer = summarizer or MapReduceSummarizer(generator)
        self._n_examples = n_examples
        self._merge_threshold = merge_threshold
        self._session_factory = session_factory
        self._workers = workers

    @property
    def total_usage(self):
        return self._generator.total_usage

    @property
    def summarizer_usage(self):
        return self._summarizer._generator.total_usage

    async def annotate(self, jobs: list[AnnotatingJob]) -> list[Annotation]:
        """Annotate clusters for given jobs. Returns Annotation list."""
        results: list[Annotation] = []
        for job in jobs:
            annotations = await self._annotate_job(job)
            results.extend(annotations)
        logger.info("annotation_complete results=%d", len(results))
        return results

    async def _annotate_job(self, job: AnnotatingJob) -> list[Annotation]:
        """Annotate all clusters for a single job (type + category)."""
        clusters = await self._cluster_repo.find_by(type=job.type, category=job.category)

        annotations: list[Annotation] = []
        for cluster in clusters:
            if cluster.label.endswith(":?"):
                continue
            annotations.append(Annotation(
                cluster_id=cluster.id,
                type=job.type,
                category=job.category,
                label=cluster.label,
            ))

        if not annotations:
            return []

        # Tool clusters with single cluster: static annotation, no LLM
        if job.category != "text" and len(annotations) < 2:
            for ann in annotations:
                if job.type == "observation":
                    ann.text = f"tool {job.category} response"
                else:
                    ann.text = f"agent used tool {job.category}"
            return annotations

        # Full pipeline: find neighbors → contrastive annotate → embed → merge
        return await self._annotate_group(annotations, job.type, job.category)

    async def _annotate_group(
        self, annotations: list[Annotation], cluster_type: str, category: str,
    ) -> list[Annotation]:
        """Contrastive annotation + embedding + agglomerative merge for a group."""
        ann_by_label = {a.label: a for a in annotations}
        cluster_ids = {a.cluster_id for a in annotations}

        # Compute centroids and build neighbor map: label → nearest label
        neighbors = await self._build_neighbor_map(cluster_ids)
        sem = asyncio.Semaphore(self._workers)

        async def annotate_async(ann: Annotation) -> None:
            async with sem:
                neighbor_label = neighbors.get(ann.label)
                neighbor = ann_by_label.get(neighbor_label) if neighbor_label else None
                if neighbor is not None:
                    prompt = get_prompt(cluster_type, category, contrastive=True)
                    ann.text = await self._annotate_contrastive(ann, neighbor, prompt)
                else:
                    prompt = get_prompt(cluster_type, category, contrastive=False)
                    ann.text = await self._annotate_solo(ann, prompt)
                logger.info("annotated %s → %s", ann.label, ann.text)

        results = await asyncio.gather(
            *[annotate_async(a) for a in annotations], return_exceptions=True,
        )
        for ann, result in zip(annotations, results):
            if isinstance(result, Exception):
                logger.error("annotation failed %s: %s", ann.label, result)
                ann.text = None

        # Embed annotations in parallel
        async def embed_async(ann: Annotation) -> None:
            async with sem:
                emb = await self._embedder.embed_text(ann.text)
                ann.embeddings = [np.array(emb)]

        await asyncio.gather(
            *[embed_async(a) for a in annotations if a.text], return_exceptions=True,
        )

        return _agglomerative_merge(annotations, self._merge_threshold)

    async def _build_neighbor_map(
        self, cluster_ids: set[UUID],
    ) -> dict[str, str | None]:
        """Compute centroids via DB, return {label: nearest_neighbor_label}."""
        rows = await self._cluster_repo.get_centroids(cluster_ids)

        centroids: dict[str, np.ndarray] = {}
        for cluster_id, label, centroid_raw in rows:
            if centroid_raw is None:
                continue
            if isinstance(centroid_raw, str):
                vec = np.fromstring(centroid_raw.strip("[]"), sep=",", dtype=np.float32)
            else:
                vec = np.array(centroid_raw, dtype=np.float32)
            centroids[label] = l2_normalize(vec)

        neighbors: dict[str, str | None] = {}
        labels = list(centroids.keys())
        for label in labels:
            best_label = None
            best_sim = -1.0
            for other_label in labels:
                if other_label == label:
                    continue
                sim = float(np.dot(centroids[label], centroids[other_label]))
                if sim > best_sim:
                    best_sim = sim
                    best_label = other_label
            neighbors[label] = best_label

        logger.info("neighbor_map: %d clusters, %d with neighbors", len(labels), sum(1 for v in neighbors.values() if v))
        return neighbors

    async def _sample_examples(self, ann: Annotation) -> list[str]:
        """Sample random messages from cluster and summarize long ones."""
        if self._session_factory:
            async with self._session_factory() as session:
                msg_repo = MessageRepository(session)
                result = await self._do_sample(ann, msg_repo)
                await session.commit()
                return result
        return await self._do_sample(ann, self._message_repo)

    async def _do_sample(self, ann: Annotation, msg_repo: MessageRepository) -> list[str]:
        messages = await msg_repo.sample_by_cluster(
            ann.cluster_id, self._n_examples,
        )
        for msg in messages:
            if not msg.summary:
                formatted = _format_for_annotation(msg)
                if len(formatted) > 1000:
                    msg.summary = await self._summarizer.summarize(formatted)
                    logger.info("summary msg_id=%s role=%s text=%s", msg.id, msg.role, msg.summary[:120])
                else:
                    msg.summary = formatted
                await msg_repo.update(msg.id, summary=msg.summary)

        return [msg.summary for msg in messages if msg.summary]

    async def _annotate_solo(self, ann: Annotation, prompt: str) -> str:
        examples = await self._sample_examples(ann)
        if not examples:
            return ""
        user_content = "\n\n---\n\n".join(
            f"{i+1}. {ex}" for i, ex in enumerate(examples)
        )
        return await self._generator.generate(
            [system_message(prompt), user_message(user_content)],
            max_tokens=50,
        )

    async def _annotate_contrastive(
        self, ann: Annotation, neighbor: Annotation, prompt: str,
    ) -> str:
        target_examples = await self._sample_examples(ann)
        neighbor_examples = await self._sample_examples(neighbor)
        if not target_examples:
            return ""

        target_section = "\n\n---\n\n".join(
            f"{i+1}. {ex}" for i, ex in enumerate(target_examples)
        )
        neighbor_section = "\n\n---\n\n".join(
            f"{i+1}. {ex}" for i, ex in enumerate(neighbor_examples)
        )
        user_content = (
            "TARGET cluster:\n\n" + target_section
            + "\n\n===\n\nSIMILAR cluster:\n\n" + neighbor_section
        )
        return await self._generator.generate(
            [system_message(prompt), user_message(user_content)],
            max_tokens=50,
        )


# --- Pure functions ---

def _complete_linkage_sim(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    """Min cosine similarity across all pairs (complete-linkage)."""
    return float(min(np.dot(ea, eb) for ea in a for eb in b))


def _agglomerative_merge(items: list[Annotation], threshold: float) -> list[Annotation]:
    """Agglomerative merge with complete-linkage: no centroid drift."""
    active = list(range(len(items)))

    while len(active) >= 2:
        best_sim, best_i, best_j = -1.0, -1, -1
        for ai in range(len(active)):
            for aj in range(ai + 1, len(active)):
                i, j = active[ai], active[aj]
                if not items[i].embeddings or not items[j].embeddings:
                    continue
                sim = _complete_linkage_sim(items[i].embeddings, items[j].embeddings)
                if sim > best_sim:
                    best_sim, best_i, best_j = sim, ai, aj

        if best_sim < threshold:
            break

        i, j = active[best_i], active[best_j]
        Annotation.merge([items[i], items[j]])

        logger.info(
            "merge %s: %s + %s → %s",
            f"{best_sim:.3f}", items[i].label, items[j].label, items[i].text,
        )

        active.pop(best_j)

    return [items[i] for i in active]


def _format_for_annotation(msg: Message) -> str:
    """Format message content for LLM annotation, preserving tool names."""
    parts = []
    for block in msg.content:
        match block.get("type"):
            case "text":
                parts.append(block["text"])
            case "tool_call":
                args = json_to_text(block.get("input", {}))
                parts.append(f"tool: {block['tool_name']}\n{args}")
            case "tool_response":
                resp = json_to_text(block.get("tool_response", ""))
                parts.append(f"tool: {block['tool_name']}\n{resp}")
    return "\n".join(parts)
