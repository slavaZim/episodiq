"""Tests for AnnotationSaver."""


from episodiq.clustering.annotator.annotator import Annotation
from episodiq.clustering.annotator.saver import AnnotationSaver
from tests.clustering.annotator.conftest import make_cluster
from tests.in_memory_repos import (
    InMemoryClusterRepository,
    InMemoryMessageRepository,
)


class TestAnnotationSaver:

    async def test_single_cluster_sets_annotation(self):
        """Single-cluster annotation updates cluster annotation field."""
        msg_repo = InMemoryMessageRepository()
        cluster_repo = InMemoryClusterRepository()


        cluster = await make_cluster(cluster_repo, msg_repo, "action", "text", "a:text:0")

        ann = Annotation(
            cluster_id=cluster.id, type="action", category="text",
            label="a:text:0", text="agent explored code",
        )

        saver = AnnotationSaver(cluster_repo, msg_repo)
        absorbed = await saver.save([ann])

        assert absorbed == 0
        assert cluster.annotation == "agent explored code"

    async def test_merged_clusters_reassign_messages(self):
        """Merged annotation reassigns messages and deletes absorbed cluster."""
        msg_repo = InMemoryMessageRepository()
        cluster_repo = InMemoryClusterRepository()


        c0 = await make_cluster(cluster_repo, msg_repo, "action", "text", "a:text:0")
        c1 = await make_cluster(cluster_repo, msg_repo, "action", "text", "a:text:1")

        ann = Annotation(
            cluster_id=c0.id, type="action", category="text",
            label="a:text:0", text="agent explored code",
            merged_ids={c0.id, c1.id},
        )

        saver = AnnotationSaver(cluster_repo, msg_repo)
        absorbed = await saver.save([ann])

        assert absorbed == 1
        # Representative (smallest label) keeps annotation
        assert c0.annotation == "agent explored code"
        # Absorbed cluster deleted
        found = await cluster_repo.find_by(id=c1.id)
        assert found == []
        # Messages reassigned to representative
        reassigned = await msg_repo.find_by(cluster_id=c0.id)
        assert len(reassigned) == 6  # 3 from c0 + 3 from c1

    async def test_skips_annotations_without_text(self):
        msg_repo = InMemoryMessageRepository()
        cluster_repo = InMemoryClusterRepository()


        cluster = await make_cluster(cluster_repo, msg_repo, "action", "text", "a:text:0")

        ann = Annotation(
            cluster_id=cluster.id, type="action", category="text",
            label="a:text:0", text=None,
        )

        saver = AnnotationSaver(cluster_repo, msg_repo)
        absorbed = await saver.save([ann])

        assert absorbed == 0
        assert cluster.annotation is None

    async def test_picks_smallest_label_as_representative(self):
        """Representative is the cluster with the smallest label."""
        msg_repo = InMemoryMessageRepository()
        cluster_repo = InMemoryClusterRepository()


        c1 = await make_cluster(cluster_repo, msg_repo, "action", "text", "a:text:1")
        c0 = await make_cluster(cluster_repo, msg_repo, "action", "text", "a:text:0")

        ann = Annotation(
            cluster_id=c1.id, type="action", category="text",
            label="a:text:1", text="merged",
            merged_ids={c0.id, c1.id},
        )

        saver = AnnotationSaver(cluster_repo, msg_repo)
        await saver.save([ann])

        # c0 is representative (smallest label), c1 is absorbed
        assert c0.annotation == "merged"
        found_c1 = await cluster_repo.find_by(id=c1.id)
        assert found_c1 == []

    async def test_empty_annotations(self):
        msg_repo = InMemoryMessageRepository()
        cluster_repo = InMemoryClusterRepository()


        saver = AnnotationSaver(cluster_repo, msg_repo)
        absorbed = await saver.save([])

        assert absorbed == 0
