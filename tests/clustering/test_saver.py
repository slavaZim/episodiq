"""Tests for ClusterSaver."""

from uuid import uuid4


from episodiq.clustering.manager import CategoryResult
from episodiq.clustering.saver import ClusterSaver
from tests.in_memory_repos import InMemoryClusterRepository


class TestClusterSaver:

    async def test_assignments_match_clusters(self):
        """Each assignment points to the cluster created for its label."""
        repo = InMemoryClusterRepository()
        saver = ClusterSaver(repo)

        msg1, msg2, msg3 = uuid4(), uuid4(), uuid4()
        result = CategoryResult(
            type="action", category="text",
            message_ids=[msg1, msg2, msg3],
            labels=["a:text:0", "a:text:0", "a:text:1"],
        )

        assignments = await saver.save([result])

        c0 = repo.get_by_label("a:text:0")
        c1 = repo.get_by_label("a:text:1")
        by_msg = {a.message_id: a.cluster_id for a in assignments}
        assert by_msg == {msg1: c0.id, msg2: c0.id, msg3: c1.id}

    async def test_skips_noise_labels(self):
        """Labels ending with ':?' produce no cluster or assignment."""
        repo = InMemoryClusterRepository()
        saver = ClusterSaver(repo)

        msg1, msg2 = uuid4(), uuid4()
        result = CategoryResult(
            type="observation", category="bash",
            message_ids=[msg1, msg2],
            labels=["o:bash:0", "o:bash:?"],
        )

        assignments = await saver.save([result])

        c0 = repo.get_by_label("o:bash:0")
        assert len(assignments) == 1
        assert assignments[0].message_id == msg1
        assert assignments[0].cluster_id == c0.id
        assert repo.get_by_label("o:bash:?") is None

    async def test_deletes_old_clusters(self):
        """Old clusters for same type+category are replaced."""
        repo = InMemoryClusterRepository()
        await repo.create(type="action", category="text", label="a:text:old")

        saver = ClusterSaver(repo)
        result = CategoryResult(
            type="action", category="text",
            message_ids=[uuid4()],
            labels=["a:text:0"],
        )

        await saver.save([result])

        assert repo.get_by_label("a:text:old") is None
        assert repo.get_by_label("a:text:0") is not None

    async def test_preserves_other_categories(self):
        """Delete only affects matching type+category."""
        repo = InMemoryClusterRepository()
        await repo.create(type="observation", category="text", label="o:text:keep")

        saver = ClusterSaver(repo)
        result = CategoryResult(
            type="action", category="text",
            message_ids=[uuid4()],
            labels=["a:text:0"],
        )

        await saver.save([result])

        assert repo.get_by_label("o:text:keep") is not None

    async def test_empty_results(self):
        repo = InMemoryClusterRepository()
        saver = ClusterSaver(repo)
        assert await saver.save([]) == []
