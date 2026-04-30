"""Integration test: ClusteringPipeline → AnnotationPipeline with merge."""

from uuid import uuid4

import numpy as np
import pytest

from episodiq.api_adapters.base import Usage
from episodiq.clustering.annotator.annotator import AnnotatingJobSpec
from episodiq.clustering.annotator.pipeline import AnnotationPipeline
from episodiq.clustering.constants import Params
from episodiq.clustering.manager import JobSpec
from episodiq.clustering.pipeline import ClusteringPipeline
from episodiq.storage.postgres.models import Message
from episodiq.storage.postgres.repository import ClusterRepository, MessageRepository
from episodiq.utils import l2_normalize
from tests.helpers import MockEmbedder, make_distant_vector

DIM = 1024
PARAMS = Params(min_cluster_size=3, min_samples=2, umap_dims=10, umap_n_neighbors=4)


def _make_group(base: list[float], content_prefix: str, n: int = 5) -> list[dict]:
    """Create n messages around a base vector with small perturbations."""
    rng = np.random.RandomState(hash(content_prefix) % 2**31)
    msgs = []
    for i in range(n):
        perturbed = np.array(base) + rng.randn(DIM) * 0.01
        embedding = l2_normalize(perturbed.tolist())
        msgs.append({
            "content": [{"type": "text", "text": f"{content_prefix} example {i}"}],
            "embedding": embedding,
        })
    return msgs


class ContentAwareGenerator:
    """Returns annotation based on message content in the prompt."""

    def __init__(self):
        self.total_usage = Usage(input_tokens=0, output_tokens=0)
        self.calls: list = []

    async def generate(self, messages, *, max_tokens=256):
        self.calls.append(messages)
        user_text = messages[-1].text
        if "math" in user_text:
            return "math operations"
        return "greeting phrases"


def _make_annotation_embedder() -> MockEmbedder:
    """Embedder that maps annotation texts to vectors: 'greeting' → same base, 'math' → distant."""
    greeting_base = l2_normalize(np.random.RandomState(99).randn(DIM).tolist())

    def embed_fn(text: str, dim: int, **kwargs) -> list[float]:
        if "math" in text:
            return make_distant_vector(greeting_base, 0.9)
        return greeting_base

    return MockEmbedder(embedding_fn=embed_fn, dims=DIM)


@pytest.mark.asyncio(loop_scope="session")
class TestClusteringAnnotationIntegration:
    """Seed 3 groups → cluster → annotate → verify merge of 2 duplicate clusters."""

    async def test_clustering_then_annotation_merge(self, session_factory):
        # --- Seed data: 3 groups of messages ---
        rng = np.random.RandomState(42)
        base_b = l2_normalize(rng.randn(DIM).tolist())
        base_c = make_distant_vector(base_b, 0.05)  # close to B
        base_a = make_distant_vector(base_b, 0.9)    # far from B/C

        group_a = _make_group(base_a, "math question 2+2")
        group_b = _make_group(base_b, "hello greeting")
        group_c = _make_group(base_c, "hi salutation greeting")

        trajectory_id = uuid4()
        async with session_factory() as session:
            idx = 0
            for group in [group_a, group_b, group_c]:
                for msg_data in group:
                    session.add(Message(
                        trajectory_id=trajectory_id,
                        role="user",
                        content=msg_data["content"],
                        embedding=msg_data["embedding"],
                        category="text",
                        cluster_type="observation",
                        index=idx,
                    ))
                    idx += 1
            await session.commit()

        # --- Step 1: Run clustering ---
        specs = [JobSpec(type="observation", category="text", params=PARAMS)]
        pipeline = ClusteringPipeline(session_factory, specs=specs)
        results = await pipeline.run()

        assert len(results) == 1
        result = results[0]
        assert result.n_clusters == 3, f"Expected 3 clusters, got {result.n_clusters}"
        assert result.noise_count == 0, f"Expected no noise, got {result.noise_count}"

        # Verify clusters in DB
        async with session_factory() as session:
            cluster_repo = ClusterRepository(session)
            clusters = await cluster_repo.find_by(type="observation", category="text")
            assert len(clusters) == 3

            msg_repo = MessageRepository(session)
            msgs = await msg_repo.find_by(trajectory_id=trajectory_id)
            assert all(m.cluster_id is not None for m in msgs), "All messages should be assigned"

        # --- Step 2: Run annotation with merge ---
        generator = ContentAwareGenerator()
        embedder = _make_annotation_embedder()

        ann_pipeline = AnnotationPipeline(
            session_factory,
            generator=generator,
            embedder=embedder,
            merge_threshold=0.85,
            n_examples=3,
        )
        ann_result = await ann_pipeline.run(
            specs=[AnnotatingJobSpec(type="observation", category="text")],
        )

        assert ann_result.merged_count == 1, f"Expected 1 merged cluster, got {ann_result.merged_count}"

        # --- Step 3: Verify final state ---
        async with session_factory() as session:
            cluster_repo = ClusterRepository(session)
            remaining = await cluster_repo.find_by(type="observation", category="text")
            assert len(remaining) == 2, f"Expected 2 clusters after merge, got {len(remaining)}"

            annotations = {c.annotation for c in remaining}
            assert "math operations" in annotations
            assert "greeting phrases" in annotations

            # All messages from groups B+C should point to same cluster
            msg_repo = MessageRepository(session)
            msgs = await msg_repo.find_by(trajectory_id=trajectory_id)

            greeting_cluster = next(c for c in remaining if c.annotation == "greeting phrases")
            math_cluster = next(c for c in remaining if c.annotation == "math operations")

            greeting_msgs = [m for m in msgs if m.cluster_id == greeting_cluster.id]
            math_msgs = [m for m in msgs if m.cluster_id == math_cluster.id]

            assert len(greeting_msgs) == 10, f"Expected 10 greeting msgs, got {len(greeting_msgs)}"
            assert len(math_msgs) == 5, f"Expected 5 math msgs, got {len(math_msgs)}"

        # Generator was called for each of the 3 clusters
        assert len(generator.calls) == 3
