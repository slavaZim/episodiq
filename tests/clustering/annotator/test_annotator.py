"""Tests for ClusterAnnotator, Annotation, resolve_annotation_jobs, and pure functions."""

from uuid import uuid4

import numpy as np
import pytest

from episodiq.clustering.annotator.annotator import (
    Annotation,
    AnnotatingJob,
    AnnotatingJobSpec,
    ClusterAnnotator,
    _agglomerative_merge,
    _complete_linkage_sim,
    _format_for_annotation,
    resolve_annotation_jobs,
)
from tests.clustering.annotator.conftest import MockGenerator, make_cluster
from tests.helpers import MockEmbedder, text_to_embedding
from tests.in_memory_repos import (
    InMemoryClusterRepository,
    InMemoryMessageRepository,
    Message,
)


# --- Pure functions ---

class TestCompleteLinkageSim:

    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert _complete_linkage_sim([v], [v]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _complete_linkage_sim([a], [b]) == pytest.approx(0.0)

    def test_min_across_pairs(self):
        """Complete linkage returns the worst (min) similarity."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.9, 0.1])
        v3 = np.array([0.0, 1.0])
        sim = _complete_linkage_sim([v1, v2], [v3])
        assert sim == pytest.approx(float(np.dot(v1, v3)), abs=0.01)


class TestAgglomerativeMerge:

    def test_merges_similar(self):
        """Items with identical embeddings merge into one."""
        v = np.array([1.0, 0.0, 0.0])
        items = [
            Annotation(cluster_id=uuid4(), type="a", category="text", label="0",
                       text="same", embeddings=[v]),
            Annotation(cluster_id=uuid4(), type="a", category="text", label="1",
                       text="same", embeddings=[v]),
        ]
        result = _agglomerative_merge(items, threshold=0.9)
        assert len(result) == 1
        assert len(result[0].merged_ids) == 2

    def test_keeps_dissimilar(self):
        """Orthogonal items stay separate."""
        items = [
            Annotation(cluster_id=uuid4(), type="a", category="text", label="0",
                       text="a", embeddings=[np.array([1.0, 0.0])]),
            Annotation(cluster_id=uuid4(), type="a", category="text", label="1",
                       text="b", embeddings=[np.array([0.0, 1.0])]),
        ]
        result = _agglomerative_merge(items, threshold=0.5)
        assert len(result) == 2

    def test_skips_empty_embeddings(self):
        items = [
            Annotation(cluster_id=uuid4(), type="a", category="text", label="0",
                       text="a", embeddings=[]),
            Annotation(cluster_id=uuid4(), type="a", category="text", label="1",
                       text="b", embeddings=[np.array([1.0, 0.0])]),
        ]
        result = _agglomerative_merge(items, threshold=0.5)
        assert len(result) == 2


class TestFormatForAnnotation:

    def test_text_block(self):
        msg = Message(
            id=uuid4(), trajectory_id=uuid4(), role="user",
            content=[{"type": "text", "text": "hello world"}], index=0,
        )
        assert _format_for_annotation(msg) == "hello world"

    def test_tool_call_block(self):
        msg = Message(
            id=uuid4(), trajectory_id=uuid4(), role="assistant",
            content=[{"type": "tool_call", "tool_name": "bash", "input": {"cmd": "ls"}}],
            index=0,
        )
        result = _format_for_annotation(msg)
        assert "tool: bash" in result
        assert "ls" in result

    def test_tool_response_block(self):
        msg = Message(
            id=uuid4(), trajectory_id=uuid4(), role="tool",
            content=[{"type": "tool_response", "tool_name": "bash", "tool_response": "file.txt"}],
            index=0,
        )
        result = _format_for_annotation(msg)
        assert "tool: bash" in result
        assert "file.txt" in result


# --- Annotation dataclass ---

class TestAnnotation:

    def test_post_init_sets_merged_ids(self):
        cid = uuid4()
        ann = Annotation(cluster_id=cid, type="a", category="text", label="0")
        assert ann.merged_ids == {cid}

    def test_merge_combines_ids_and_embeddings(self):
        id1, id2 = uuid4(), uuid4()
        v1, v2 = np.array([1.0]), np.array([2.0])
        a1 = Annotation(cluster_id=id1, type="a", category="text", label="0",
                        text="first", embeddings=[v1])
        a2 = Annotation(cluster_id=id2, type="a", category="text", label="1",
                        text="second", embeddings=[v2])
        merged = Annotation.merge([a1, a2])
        assert merged.merged_ids == {id1, id2}
        assert len(merged.embeddings) == 2


# --- resolve_annotation_jobs ---

class TestResolveAnnotationJobs:

    async def test_default_specs(self):
        repo = InMemoryMessageRepository()
        tid = uuid4()
        for role, cat, ct in [("assistant", "bash", "action"), ("user", "text", "observation")]:
            repo.add_message(Message(
                id=uuid4(), trajectory_id=tid, role=role,
                content=[], index=0, category=cat, cluster_type=ct,
            ))

        jobs = await resolve_annotation_jobs(repo)
        types_cats = {(j.type, j.category) for j in jobs}
        assert ("action", "text") in types_cats
        assert ("observation", "text") in types_cats
        assert ("action", "bash") in types_cats

    async def test_specific_category(self):
        repo = InMemoryMessageRepository()
        specs = [AnnotatingJobSpec(type="action", category="bash")]
        jobs = await resolve_annotation_jobs(repo, specs)
        assert len(jobs) == 1
        assert jobs[0].category == "bash"

    async def test_tool_discovers_non_text(self):
        repo = InMemoryMessageRepository()
        tid = uuid4()
        for cat in ["text", "bash", "editor"]:
            repo.add_message(Message(
                id=uuid4(), trajectory_id=tid, role="assistant",
                content=[], index=0, category=cat, cluster_type="action",
            ))
        specs = [AnnotatingJobSpec(type="action", category="tool")]
        jobs = await resolve_annotation_jobs(repo, specs)
        assert {j.category for j in jobs} == {"bash", "editor"}


# --- ClusterAnnotator ---

class TestClusterAnnotator:

    async def test_single_tool_cluster_static_annotation(self):
        """Single tool cluster gets static annotation without LLM."""
        msg_repo = InMemoryMessageRepository()
        cluster_repo = InMemoryClusterRepository()
        gen = MockGenerator()

        await make_cluster(cluster_repo, msg_repo, "action", "bash", "a:bash:0")

        annotator = ClusterAnnotator(
            message_repo=msg_repo,
            cluster_repo=cluster_repo,
            generator=gen,
            embedder=MockEmbedder(),
        )
        results = await annotator.annotate([AnnotatingJob(type="action", category="bash")])

        assert len(results) == 1
        assert results[0].text == "agent used tool bash"
        assert len(gen.calls) == 0

    async def test_single_observation_tool_cluster(self):
        msg_repo = InMemoryMessageRepository()
        cluster_repo = InMemoryClusterRepository()
        gen = MockGenerator()

        await make_cluster(cluster_repo, msg_repo, "observation", "bash", "o:bash:0")

        annotator = ClusterAnnotator(
            message_repo=msg_repo,
            cluster_repo=cluster_repo,
            generator=gen,
            embedder=MockEmbedder(),
        )
        results = await annotator.annotate([AnnotatingJob(type="observation", category="bash")])

        assert len(results) == 1
        assert results[0].text == "tool bash response"

    async def test_skips_noise_clusters(self):
        msg_repo = InMemoryMessageRepository()
        cluster_repo = InMemoryClusterRepository()
        gen = MockGenerator()

        await make_cluster(cluster_repo, msg_repo, "action", "text", "a:text:?")

        annotator = ClusterAnnotator(
            message_repo=msg_repo,
            cluster_repo=cluster_repo,
            generator=gen,
            embedder=MockEmbedder(),
        )
        results = await annotator.annotate([AnnotatingJob(type="action", category="text")])
        assert len(results) == 0

    async def test_multiple_text_clusters_contrastive(self):
        """Multiple text clusters use LLM contrastive annotation."""
        msg_repo = InMemoryMessageRepository()
        cluster_repo = InMemoryClusterRepository()
        gen = MockGenerator(responses=["agent explored code", "agent wrote tests"])
        embedder = MockEmbedder(dims=50)

        await make_cluster(cluster_repo, msg_repo, "action", "text", "a:text:0")
        await make_cluster(cluster_repo, msg_repo, "action", "text", "a:text:1")
        cluster_repo.link_messages(msg_repo)

        annotator = ClusterAnnotator(
            message_repo=msg_repo,
            cluster_repo=cluster_repo,
            generator=gen,
            embedder=embedder,
        )

        results = await annotator.annotate([AnnotatingJob(type="action", category="text")])

        assert len(gen.calls) == 2
        texts = {r.text for r in results}
        assert "agent explored code" in texts

    async def test_multiple_tool_clusters_use_llm(self):
        """2+ tool clusters go through full contrastive pipeline."""
        msg_repo = InMemoryMessageRepository()
        cluster_repo = InMemoryClusterRepository()
        gen = MockGenerator(responses=["bash cluster A", "bash cluster B"])
        embedder = MockEmbedder(dims=50)

        await make_cluster(cluster_repo, msg_repo, "action", "bash", "a:bash:0")
        await make_cluster(cluster_repo, msg_repo, "action", "bash", "a:bash:1")
        cluster_repo.link_messages(msg_repo)

        annotator = ClusterAnnotator(
            message_repo=msg_repo,
            cluster_repo=cluster_repo,
            generator=gen,
            embedder=embedder,
        )

        results = await annotator.annotate([AnnotatingJob(type="action", category="bash")])

        assert len(gen.calls) == 2
        assert all(r.text for r in results)

    async def test_summary_persisted_to_db(self):
        """Long messages get summarized; summarizer is called and summary is persisted."""
        msg_repo = InMemoryMessageRepository()
        cluster_repo = InMemoryClusterRepository()
        gen = MockGenerator(default="annotation text")
        embedder = MockEmbedder(dims=50)

        summarizer_calls = []

        class TrackingSummarizer:
            async def summarize(self, text: str) -> str:
                summarizer_calls.append(text)
                return "summarized content"

        cluster = await cluster_repo.create(type="action", category="text", label="a:text:0")
        msg_repo.add_cluster(cluster)
        long_text = "word " * 600  # >1000 chars
        tid = uuid4()
        msg = Message(
            id=uuid4(), trajectory_id=tid, role="assistant",
            content=[{"type": "text", "text": long_text}], index=0,
            embedding=text_to_embedding("long", 50),
            cluster_id=cluster.id, category="text",
        )
        msg_repo.add_message(msg)

        await make_cluster(cluster_repo, msg_repo, "action", "text", "a:text:1")
        cluster_repo.link_messages(msg_repo)

        annotator = ClusterAnnotator(
            message_repo=msg_repo,
            cluster_repo=cluster_repo,
            generator=gen,
            embedder=embedder,
            summarizer=TrackingSummarizer(),
        )

        await annotator.annotate([AnnotatingJob(type="action", category="text")])

        assert len(summarizer_calls) == 1
        assert msg.summary == "summarized content"

    async def test_empty_clusters_returns_empty(self):
        msg_repo = InMemoryMessageRepository()
        cluster_repo = InMemoryClusterRepository()

        annotator = ClusterAnnotator(
            message_repo=msg_repo,
            cluster_repo=cluster_repo,
            generator=MockGenerator(),
            embedder=MockEmbedder(),
        )
        results = await annotator.annotate([AnnotatingJob(type="action", category="text")])
        assert results == []

    async def test_total_usage_tracked(self):
        msg_repo = InMemoryMessageRepository()
        cluster_repo = InMemoryClusterRepository()
        gen = MockGenerator()

        annotator = ClusterAnnotator(
            message_repo=msg_repo,
            cluster_repo=cluster_repo,
            generator=gen,
            embedder=MockEmbedder(),
        )
        assert annotator.total_usage.input_tokens == 0
