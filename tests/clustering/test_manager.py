"""Tests for ClusteringManager and resolve_jobs."""

from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np

from episodiq.clustering.constants import Params
from episodiq.clustering.manager import (
    ClusteringJob,
    ClusteringManager,
    JobSpec,
    _resolve_categories,
    resolve_jobs,
)
from tests.in_memory_repos import InMemoryMessageRepository, Message


def _messages(repo: InMemoryMessageRepository, n: int, role: str, category: str, dim: int = 50, cluster_type: str = "action") -> list[Message]:
    rng = np.random.RandomState(42)
    tid = uuid4()
    msgs = []
    for i in range(n):
        m = Message(
            id=uuid4(), trajectory_id=tid, role=role,
            content=[], index=i,
            embedding=rng.randn(dim).tolist(),
            category=category, cluster_type=cluster_type,
        )
        repo.add_message(m)
        msgs.append(m)
    return msgs


class TestResolveJobs:

    async def test_empty_specs_uses_defaults(self):
        """Empty specs → default jobs: action/observation × text + discovered tools."""
        repo = InMemoryMessageRepository()
        # Add messages with tool categories so discovery finds them
        tid = uuid4()
        for role, cat, ct in [("assistant", "bash", "action"), ("assistant", "editor", "action"), ("user", "text", "observation")]:
            repo.add_message(Message(
                id=uuid4(), trajectory_id=tid, role=role, content=[], index=0, category=cat, cluster_type=ct,
            ))

        jobs = await resolve_jobs(repo, [])

        assert len(jobs) == 4
        types = {(j.type, j.category) for j in jobs}
        assert ("action", "text") in types
        assert ("observation", "text") in types
        assert ("action", "bash") in types
        assert ("action", "editor") in types
        assert ("observation", "bash") not in types
        assert ("observation", "editor") not in types

    async def test_specific_category_passes_through(self):
        repo = InMemoryMessageRepository()

        jobs = await resolve_jobs(
            repo,
            [JobSpec(type="action", category="bash")],
        )

        assert len(jobs) == 1
        assert jobs[0].category == "bash"

    async def test_tool_category_discovers_from_db(self):
        repo = InMemoryMessageRepository()
        tid = uuid4()
        for cat in ["text", "bash", "editor"]:
            repo.add_message(Message(
                id=uuid4(), trajectory_id=tid, role="assistant", content=[], index=0, category=cat, cluster_type="action",
            ))

        jobs = await resolve_jobs(
            repo,
            [JobSpec(type="action", category="tool")],
        )

        assert {j.category for j in jobs} == {"bash", "editor"}

    async def test_custom_params_propagated(self):
        repo = InMemoryMessageRepository()
        custom = Params(min_cluster_size=3, min_samples=2)

        jobs = await resolve_jobs(
            repo,
            [JobSpec(type="action", category="text", params=custom)],
        )

        assert jobs[0].params == custom


class TestResolveCategories:

    async def test_caches_discovered_categories(self):
        repo = InMemoryMessageRepository()
        tid = uuid4()
        for cat in ["text", "bash"]:
            repo.add_message(Message(
                id=uuid4(), trajectory_id=tid, role="assistant", content=[], index=0, category=cat, cluster_type="action",
            ))

        cache: dict = {}
        r1 = await _resolve_categories(repo, "action", "tool", cache)
        r2 = await _resolve_categories(repo, "action", "tool", cache)

        assert r1 == r2
        assert "action" in cache


class TestClusteringManager:

    async def test_flat_label_when_too_few(self):
        """Messages < min_cluster_size → flat label."""
        repo = InMemoryMessageRepository()
        _messages(repo, 3, "user", "text", cluster_type="observation")

        mgr = ClusteringManager(
            repo,
            [ClusteringJob(type="observation", category="text")],
        )
        results = await mgr.run()

        assert len(results) == 1
        assert all(lbl == "o:text" for lbl in results[0].labels)

    async def test_flat_label_when_single_cluster(self):
        """HDBSCAN returns 1 cluster → flat label (sub-IDs meaningless)."""
        repo = InMemoryMessageRepository()
        _messages(repo, 20, "assistant", "text")

        cr = MagicMock(noise_count=0, labels=np.array([0] * 20), n_clusters=1)

        from unittest.mock import patch
        with patch("episodiq.clustering.manager.Clusterer") as MC:
            MC.return_value.fit.return_value = cr
            mgr = ClusteringManager(
                repo,
                [ClusteringJob(type="action", category="text")],
            )
            results = await mgr.run()

        assert all(lbl == "a:text" for lbl in results[0].labels)

    async def test_flat_label_when_too_much_noise(self):
        repo = InMemoryMessageRepository()
        _messages(repo, 20, "assistant", "bash")

        cr = MagicMock(noise_count=20, n_clusters=2, labels=np.array([-1] * 20))

        from unittest.mock import patch
        with patch("episodiq.clustering.manager.Clusterer") as MC:
            MC.return_value.fit.return_value = cr
            mgr = ClusteringManager(
                repo,
                [ClusteringJob(type="action", category="bash")],
            )
            results = await mgr.run()

        assert all(lbl == "a:bash" for lbl in results[0].labels)

    async def test_clustered_labels(self):
        repo = InMemoryMessageRepository()
        _messages(repo, 20, "assistant", "text")

        labels = np.array([0] * 9 + [1] * 8 + [-1] * 3)
        cr = MagicMock(noise_count=3, labels=labels, n_clusters=2)

        from unittest.mock import patch
        with patch("episodiq.clustering.manager.Clusterer") as MC:
            MC.return_value.fit.return_value = cr
            mgr = ClusteringManager(
                repo,
                [ClusteringJob(type="action", category="text")],
            )
            results = await mgr.run()

        r = results[0]
        assert r.labels[0] == "a:text:0"
        assert r.labels[9] == "a:text:1"
        assert r.labels[17] == "a:text:?"

    async def test_empty_messages_skipped(self):
        repo = InMemoryMessageRepository()

        mgr = ClusteringManager(
            repo,
            [ClusteringJob(type="action", category="text")],
        )
        results = await mgr.run()

        assert results == []

    async def test_message_ids_aligned(self):
        repo = InMemoryMessageRepository()
        msgs = _messages(repo, 5, "user", "text", cluster_type="observation")

        mgr = ClusteringManager(
            repo,
            [ClusteringJob(type="observation", category="text")],
        )
        results = await mgr.run()

        assert results[0].message_ids == [m.id for m in msgs]
