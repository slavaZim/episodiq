"""Tests for ClusterAssigner with in-memory repository."""

from uuid import uuid4


from episodiq.clustering.assigner import ClusterAssigner
from tests.helpers import make_distant_vector, text_to_embedding
from tests.in_memory_repos import Cluster, InMemoryMessageRepository

DIM = 64


def _cluster(cluster_type: str = "action", category: str = "bash", label: str = "edit") -> Cluster:
    return Cluster(id=uuid4(), type=cluster_type, category=category, label=label)


async def _seed(repo, embedding, cluster, n=3):
    """Add n messages near `embedding` assigned to `cluster`."""
    for i in range(n):
        vec = make_distant_vector(embedding, min_distance=0.05)
        await repo.save(
            uuid4(), _msg(), embedding=vec, cluster_id=cluster.id,
            category=cluster.category, cluster_type=cluster.type,
        )


class _Msg:
    role = "assistant"
    content = []


def _msg():
    return _Msg()


class TestClusterAssigner:
    async def test_no_neighbors_returns_none(self):
        repo = InMemoryMessageRepository()
        assigner = ClusterAssigner(repo, k=5)
        result = await assigner.assign(uuid4(), text_to_embedding("x", DIM), "action", "bash")
        assert result is None

    async def test_unanimous_cluster(self):
        repo = InMemoryMessageRepository()
        ca = _cluster()
        repo.add_cluster(ca)
        base = text_to_embedding("edit_file", DIM)
        await _seed(repo, base, ca, n=5)

        msg = await repo.save(uuid4(), _msg(), embedding=base)
        assigner = ClusterAssigner(repo, k=5)
        result = await assigner.assign(msg.id, base, "action", "bash")
        assert result == ca.id
        assert repo._by_id[msg.id].cluster_id == ca.id

    async def test_majority_wins(self):
        repo = InMemoryMessageRepository()
        ca = _cluster(label="edit")
        cb = _cluster(label="run")
        repo.add_cluster(ca)
        repo.add_cluster(cb)

        base = text_to_embedding("query", DIM)
        await _seed(repo, base, ca, n=4)
        await _seed(repo, base, cb, n=1)

        msg = await repo.save(uuid4(), _msg(), embedding=base)
        assigner = ClusterAssigner(repo, k=7)
        result = await assigner.assign(msg.id, base, "action", "bash")
        assert result == ca.id

    async def test_below_confidence_returns_none(self):
        """Two clusters equally close → confidence ~0.5 → below 0.7 threshold."""
        repo = InMemoryMessageRepository()
        ca = _cluster(label="a")
        cb = _cluster(label="b")
        repo.add_cluster(ca)
        repo.add_cluster(cb)

        base = text_to_embedding("ambiguous", DIM)
        await _seed(repo, base, ca, n=3)
        await _seed(repo, base, cb, n=3)

        msg = await repo.save(uuid4(), _msg(), embedding=base)
        assigner = ClusterAssigner(repo, k=6, confidence_threshold=0.7)
        result = await assigner.assign(msg.id, base, "action", "bash")
        assert result is None
        assert msg.cluster_id is None

    async def test_excludes_self(self):
        repo = InMemoryMessageRepository()
        ca = _cluster()
        repo.add_cluster(ca)
        base = text_to_embedding("solo", DIM)
        msg = await repo.save(
            uuid4(), _msg(), embedding=base, cluster_id=ca.id,
            category="bash", cluster_type="action",
        )

        assigner = ClusterAssigner(repo, k=5)
        result = await assigner.assign(msg.id, base, "action", "bash")
        assert result is None

    async def test_category_filter(self):
        """Neighbors from different type+category are invisible."""
        repo = InMemoryMessageRepository()
        obs = _cluster(cluster_type="observation", category="text", label="obs")
        repo.add_cluster(obs)
        base = text_to_embedding("query", DIM)
        await _seed(repo, base, obs, n=5)

        msg = await repo.save(uuid4(), _msg(), embedding=base, category="bash", cluster_type="action")
        assigner = ClusterAssigner(repo, k=5)
        result = await assigner.assign(msg.id, base, "action", "bash")
        assert result is None

    async def test_distant_cluster_not_matched(self):
        """Neighbor is far away → high distance → low similarity weight."""
        repo = InMemoryMessageRepository()
        ca = _cluster(label="close")
        cb = _cluster(label="far")
        repo.add_cluster(ca)
        repo.add_cluster(cb)

        base = text_to_embedding("target", DIM)
        await _seed(repo, base, ca, n=2)
        # cb is far away
        far = make_distant_vector(base, min_distance=0.8)
        await repo.save(uuid4(), _msg(), embedding=far, cluster_id=cb.id, category="bash", cluster_type="action")

        msg = await repo.save(uuid4(), _msg(), embedding=base)
        assigner = ClusterAssigner(repo, k=5)
        result = await assigner.assign(msg.id, base, "action", "bash")
        assert result == ca.id

    async def test_k_limits_neighbors(self):
        """With k=2 only closest 2 vote, ignoring further cluster_b majority."""
        repo = InMemoryMessageRepository()
        ca = _cluster(label="close")
        cb = _cluster(label="many")
        repo.add_cluster(ca)
        repo.add_cluster(cb)

        base = text_to_embedding("ktest", DIM)
        # 2 very close to base in ca
        for _ in range(2):
            await repo.save(uuid4(), _msg(), embedding=make_distant_vector(base, 0.02), cluster_id=ca.id, category="bash", cluster_type="action")
        # 5 slightly further in cb
        shifted = make_distant_vector(base, 0.15)
        for _ in range(5):
            await repo.save(uuid4(), _msg(), embedding=make_distant_vector(shifted, 0.02), cluster_id=cb.id, category="bash", cluster_type="action")

        msg = await repo.save(uuid4(), _msg(), embedding=base)
        assigner = ClusterAssigner(repo, k=2)
        result = await assigner.assign(msg.id, base, "action", "bash")
        assert result == ca.id
