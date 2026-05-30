"""Tests for MessageUpdater."""

from uuid import uuid4


from episodiq.clustering.updater import ClusterAssignment, MessageUpdater
from tests.in_memory_repos import InMemoryMessageRepository, Message


def _msg(tid, role="user", index=0) -> Message:
    return Message(id=uuid4(), trajectory_id=tid, role=role, content=[], index=index)


class TestMessageUpdater:

    async def test_updates_cluster_ids(self):
        repo = InMemoryMessageRepository()
        tid = uuid4()
        m1 = _msg(tid, index=0)
        m2 = _msg(tid, index=1)
        repo.add_message(m1)
        repo.add_message(m2)

        c1, c2 = uuid4(), uuid4()
        updater = MessageUpdater(repo)
        await updater.update([
            ClusterAssignment(m1.id, c1),
            ClusterAssignment(m2.id, c2),
        ])

        assert repo._by_id[m1.id].cluster_id == c1
        assert repo._by_id[m2.id].cluster_id == c2

    async def test_empty_assignments_noop(self):
        repo = InMemoryMessageRepository()
        updater = MessageUpdater(repo)
        await updater.update([])
