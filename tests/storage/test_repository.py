"""Integration tests for repository queries.

Requires PostgreSQL with pgvector:
    docker compose -f docker-compose.test.yml up -d
"""
from uuid import UUID, uuid4

import numpy as np
import pytest
from sqlalchemy import select, text

from episodiq.api_adapters.base import (
    CanonicalAssistantMessage,
    CanonicalUserMessage,
    CanonicalToolMessage,
    Usage,
)
from episodiq.config import get_config
from episodiq.storage.postgres.models import Cluster, Message, TrajectoryPath, Trajectory
from episodiq.storage.postgres.repository import (
    ClusterRepository,
    MessageRepository,
    TrajectoryPathRepository,
    TrajectoryRepository,
    TrajectoryWindowLSHRepository,
)

_cfg = get_config()


def random_embedding() -> list[float]:
    vec = np.random.randn(_cfg.message_dims).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


@pytest.mark.asyncio(loop_scope="session")
class TestTrajectoryRepository:

    async def test_find_or_create_new(self, db_session):
        """Creates new trajectory with status='pending'."""
        repo = TrajectoryRepository(db_session)
        traj = await repo.find_or_create(uuid4())
        await db_session.commit()

        assert traj.status == "pending"

    async def test_find_or_create_existing(self, db_session):
        """Returns existing trajectory without changing it."""
        traj_id = uuid4()
        repo = TrajectoryRepository(db_session)

        traj = await repo.find_or_create(traj_id)
        await db_session.commit()

        traj2 = await repo.find_or_create(traj_id)
        assert traj2.id == traj.id


@pytest.mark.asyncio(loop_scope="session")
class TestGetMaxIndex:

    async def test_empty_trajectory(self, db_session):
        """Returns None when trajectory has no messages."""
        repo = MessageRepository(db_session)
        result = await repo.get_max_index(uuid4())
        assert result is None

    async def test_returns_highest_index(self, db_session):
        """Returns the highest index among trajectory messages."""
        tid = uuid4()
        db_session.add_all([
            Message(trajectory_id=tid, role="user", content=[], index=0),
            Message(trajectory_id=tid, role="assistant", content=[], index=1),
            Message(trajectory_id=tid, role="user", content=[], index=2),
        ])
        await db_session.flush()

        repo = MessageRepository(db_session)
        assert await repo.get_max_index(tid) == 2


@pytest.mark.asyncio(loop_scope="session")
class TestSaveMessage:

    async def test_save_user_message(self, db_session):
        """Saves user message with embedding."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        await db_session.flush()

        emb = random_embedding()
        msg = CanonicalUserMessage.build("hello")
        repo = MessageRepository(db_session)
        saved = await repo.save(tid, msg, emb)

        assert saved.role == "user"
        assert saved.trajectory_id == tid
        assert np.allclose(saved.embedding, emb, atol=1e-5)

    async def test_save_tool_message(self, db_session):
        """Saves tool message."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        await db_session.flush()

        msg = CanonicalToolMessage.build("call_1", "get_weather", "sunny")
        repo = MessageRepository(db_session)
        saved = await repo.save(tid, msg)

        assert saved.role == "tool"
        assert saved.embedding is None

    async def test_save_assistant_creates_origin_response(self, db_session):
        """Assistant message also creates an OriginResponse row."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        await db_session.flush()

        msg = CanonicalAssistantMessage.build(
            adapter_id="openai",
            text="hi there",
            model="gpt-4",
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        repo = MessageRepository(db_session)
        saved = await repo.save(tid, msg)
        await db_session.flush()

        assert saved.role == "assistant"

        # Check OriginResponse was created
        row = (await db_session.execute(
            text("SELECT adapter_id, model, input_tokens, output_tokens FROM origin_responses WHERE message_id = :mid"),
            {"mid": saved.id},
        )).one()
        assert row.adapter_id == "openai"
        assert row.model == "gpt-4"
        assert row.input_tokens == 10
        assert row.output_tokens == 5


@pytest.mark.asyncio(loop_scope="session")
class TestGetMessagesForClustering:

    async def test_observation_text(self, db_session):
        """Returns user observations with category='text' and embedding.

        Adds an initial-task observation (index=0) so the test observation
        (index=2) survives the skip_initial_observation filter.
        """
        tid = uuid4()
        emb = random_embedding()
        db_session.add_all([
            Message(
                trajectory_id=tid, role="user",
                content=[{"type": "text", "text": "task"}],
                embedding=random_embedding(), category="text",
                cluster_type="observation", index=0,
            ),
            Message(
                trajectory_id=tid, role="assistant",
                content=[{"type": "text", "text": "hi"}],
                embedding=random_embedding(), category="text",
                cluster_type="action", index=1,
            ),
            Message(
                trajectory_id=tid, role="user",
                content=[{"type": "text", "text": "hello"}],
                embedding=emb, category="text",
                cluster_type="observation", index=2,
            ),
        ])
        await db_session.commit()

        repo = MessageRepository(db_session)
        rows = await repo.get_messages_for_clustering("observation", "text")

        assert any(np.allclose(r.embedding, emb, atol=1e-5) for r in rows)
        assert all(r.role in ("user", "tool") for r in rows)

    async def test_observation_tool(self, db_session):
        """Returns tool observations with matching category.

        Adds an initial-task observation (index=0) so the test tool
        observation (index=2) survives the skip_initial_observation filter.
        """
        tid = uuid4()
        emb = random_embedding()
        db_session.add_all([
            Message(
                trajectory_id=tid, role="user",
                content=[{"type": "text", "text": "task"}],
                embedding=random_embedding(), category="text",
                cluster_type="observation", index=0,
            ),
            Message(
                trajectory_id=tid, role="assistant",
                content=[{"type": "tool_call", "id": "c1", "tool_name": "bash", "input": {}}],
                embedding=random_embedding(), category="bash",
                cluster_type="action", index=1,
            ),
            Message(
                trajectory_id=tid, role="tool",
                content=[{"type": "tool_response", "id": "c1", "tool_name": "bash", "tool_response": "ok"}],
                embedding=emb, category="bash",
                cluster_type="observation", index=2,
            ),
        ])
        await db_session.commit()

        repo = MessageRepository(db_session)
        rows = await repo.get_messages_for_clustering("observation", "bash")

        assert any(np.allclose(r.embedding, emb, atol=1e-5) for r in rows)

    async def test_action_text(self, db_session):
        """Returns assistant messages with category='text'."""
        tid = uuid4()
        emb_text = random_embedding()
        emb_tool = random_embedding()
        db_session.add_all([
            Message(
                trajectory_id=tid, role="assistant",
                content=[{"type": "text", "text": "plain answer"}],
                embedding=emb_text, category="text", cluster_type="action",
            ),
            Message(
                trajectory_id=tid, role="assistant",
                content=[{"type": "tool_call", "id": "c1", "tool_name": "get_weather", "input": {}}],
                embedding=emb_tool, category="get_weather", cluster_type="action",
            ),
        ])
        await db_session.commit()

        repo = MessageRepository(db_session)
        rows = await repo.get_messages_for_clustering("action", "text")

        embeddings = [r.embedding for r in rows]
        assert any(np.allclose(e, emb_text, atol=1e-5) for e in embeddings)
        assert not any(np.allclose(e, emb_tool, atol=1e-5) for e in embeddings)

    async def test_action_tool(self, db_session):
        """Returns assistant messages with specific tool category."""
        tid = uuid4()
        emb_text = random_embedding()
        emb_tool = random_embedding()
        db_session.add_all([
            Message(
                trajectory_id=tid, role="assistant",
                content=[{"type": "text", "text": "plain answer"}],
                embedding=emb_text, category="text", cluster_type="action",
            ),
            Message(
                trajectory_id=tid, role="assistant",
                content=[{"type": "tool_call", "id": "c1", "tool_name": "get_weather", "input": {}}],
                embedding=emb_tool, category="get_weather", cluster_type="action",
            ),
        ])
        await db_session.commit()

        repo = MessageRepository(db_session)
        rows = await repo.get_messages_for_clustering("action", "get_weather")

        embeddings = [r.embedding for r in rows]
        assert any(np.allclose(e, emb_tool, atol=1e-5) for e in embeddings)
        assert not any(np.allclose(e, emb_text, atol=1e-5) for e in embeddings)

    async def test_skips_messages_without_embedding(self, db_session):
        """Messages without embedding are excluded."""
        tid = uuid4()
        db_session.add_all([
            Message(
                trajectory_id=tid, role="user",
                content=[{"type": "text", "text": "no embedding"}],
                embedding=None, category="text", cluster_type="observation",
            ),
            Message(
                trajectory_id=tid, role="user",
                content=[{"type": "text", "text": "has embedding"}],
                embedding=random_embedding(), category="text", cluster_type="observation",
            ),
        ])
        await db_session.commit()

        repo = MessageRepository(db_session)
        rows = await repo.get_messages_for_clustering("observation", "text")

        for r in rows:
            assert r.embedding is not None


@pytest.mark.asyncio(loop_scope="session")
class TestFindClusterNeighbors:

    async def test_finds_nearest_cluster(self, db_session):
        """Returns closest clustered message's cluster."""
        tid = uuid4()
        cluster = Cluster(type="observation", category="text", label="greetings")
        db_session.add(cluster)
        await db_session.flush()

        target_emb = random_embedding()
        # Close neighbor — same direction with small perturbation
        close_emb = np.array(target_emb) + np.random.randn(len(target_emb)) * 0.01
        close_emb = (close_emb / np.linalg.norm(close_emb)).tolist()

        msg = Message(
            trajectory_id=tid, role="user",
            content=[{"type": "text", "text": "hi"}],
            embedding=close_emb,
            cluster_id=cluster.id,
            cluster_type="observation", category="text",
        )
        db_session.add(msg)
        await db_session.flush()

        repo = MessageRepository(db_session)
        neighbors = await repo.find_neighbors(
            target_emb, "observation", "text", uuid4(), k=5,
        )

        cluster_ids = [n.cluster_id for n in neighbors]
        assert cluster.id in cluster_ids

    async def test_excludes_unclustered_messages(self, db_session):
        """Messages without cluster_id are excluded."""
        tid = uuid4()
        emb = random_embedding()
        db_session.add(Message(
            trajectory_id=tid, role="user",
            content=[{"type": "text", "text": "no cluster"}],
            embedding=emb,
            cluster_id=None,
            cluster_type="observation", category="text",
        ))
        await db_session.flush()

        repo = MessageRepository(db_session)
        neighbors = await repo.find_neighbors(
            emb, "observation", "text", uuid4(), k=5,
        )
        # The unclustered message should not appear
        assert all(n.distance > 0 for n in neighbors) or len(neighbors) == 0

    async def test_excludes_self(self, db_session):
        """Excludes message with the given ID."""
        tid = uuid4()
        cluster = Cluster(type="observation", category="text", label="test")
        db_session.add(cluster)
        await db_session.flush()

        emb = random_embedding()
        msg = Message(
            trajectory_id=tid, role="user",
            content=[{"type": "text", "text": "self"}],
            embedding=emb, cluster_id=cluster.id,
            cluster_type="observation", category="text",
        )
        db_session.add(msg)
        await db_session.flush()

        repo = MessageRepository(db_session)
        neighbors = await repo.find_neighbors(
            emb, "observation", "text", msg.id, k=5,
        )
        assert all(n.cluster_id != msg.id for n in neighbors)


@pytest.mark.asyncio(loop_scope="session")
class TestUpdateCluster:

    async def test_sets_cluster_id(self, db_session):
        """Updates message's cluster_id."""
        tid = uuid4()
        cluster = Cluster(type="action", category="text", label="answer")
        db_session.add(cluster)
        msg = Message(trajectory_id=tid, role="assistant", content=[])
        db_session.add(msg)
        await db_session.flush()

        repo = MessageRepository(db_session)
        await repo.update(msg.id, cluster_id=cluster.id)
        await db_session.flush()

        await db_session.refresh(msg)
        assert msg.cluster_id == cluster.id


@pytest.mark.asyncio(loop_scope="session")
class TestGetDistinctCategories:

    async def test_returns_action_categories(self, db_session):
        """Returns distinct categories for assistant messages."""
        tid = uuid4()
        db_session.add_all([
            Message(trajectory_id=tid, role="assistant", content=[], category="text", cluster_type="action"),
            Message(trajectory_id=tid, role="assistant", content=[], category="bash", cluster_type="action"),
            Message(trajectory_id=tid, role="assistant", content=[], category="bash", cluster_type="action"),
            Message(trajectory_id=tid, role="user", content=[], category="text", cluster_type="observation"),
        ])
        await db_session.commit()

        repo = MessageRepository(db_session)
        cats = await repo.get_categories("action")

        assert "text" in cats
        assert "bash" in cats

    async def test_returns_observation_categories(self, db_session):
        """Returns distinct categories for user/tool messages."""
        tid = uuid4()
        db_session.add_all([
            Message(trajectory_id=tid, role="user", content=[], category="text", cluster_type="observation"),
            Message(trajectory_id=tid, role="tool", content=[], category="bash", cluster_type="observation"),
            Message(trajectory_id=tid, role="assistant", content=[], category="editor", cluster_type="action"),
        ])
        await db_session.commit()

        repo = MessageRepository(db_session)
        cats = await repo.get_categories("observation")

        assert "text" in cats
        assert "bash" in cats
        assert "editor" not in cats

    async def test_excludes_null_categories(self, db_session):
        """Messages without category are excluded."""
        tid = uuid4()
        db_session.add(Message(trajectory_id=tid, role="assistant", content=[], category=None, cluster_type="action"))
        await db_session.commit()

        repo = MessageRepository(db_session)
        cats = await repo.get_categories("action")
        # Should not contain None
        assert None not in cats


@pytest.mark.asyncio(loop_scope="session")
class TestGetDistinctTrajectoryIds:

    async def test_returns_distinct_ids(self, db_session):
        """Returns unique trajectory IDs."""
        tid1, tid2 = uuid4(), uuid4()
        db_session.add_all([
            Message(trajectory_id=tid1, role="user", content=[]),
            Message(trajectory_id=tid1, role="assistant", content=[]),
            Message(trajectory_id=tid2, role="user", content=[]),
        ])
        await db_session.commit()

        repo = MessageRepository(db_session)
        ids = await repo.get_distinct_trajectory_ids()

        assert tid1 in ids
        assert tid2 in ids


@pytest.mark.asyncio(loop_scope="session")
class TestGetTrajectoryWithClusters:

    async def test_returns_ordered_with_clusters(self, db_session):
        """Returns messages sorted by index with cluster relationship loaded."""
        tid = uuid4()
        cluster = Cluster(type="observation", category="text", label="o:text:0")
        db_session.add(cluster)
        await db_session.flush()

        m0 = Message(trajectory_id=tid, role="user", content=[], index=0, cluster_id=cluster.id)
        m1 = Message(trajectory_id=tid, role="assistant", content=[], index=1)
        m2 = Message(trajectory_id=tid, role="user", content=[], index=2)
        db_session.add(m0)
        db_session.add(m1)
        db_session.add(m2)
        await db_session.commit()

        repo = MessageRepository(db_session)
        msgs = await repo.get_trajectory_with_clusters(tid)

        assert [m.index for m in msgs] == [0, 1, 2]
        assert msgs[0].cluster is not None
        assert msgs[0].cluster.label == "o:text:0"
        assert msgs[1].cluster is None

    async def test_empty_trajectory(self, db_session):
        repo = MessageRepository(db_session)
        msgs = await repo.get_trajectory_with_clusters(uuid4())
        assert msgs == []


@pytest.mark.asyncio(loop_scope="session")
class TestClusterRepository:

    async def test_create(self, db_session):
        """Creates cluster with generated ID."""
        repo = ClusterRepository(db_session)
        cluster = await repo.create(type="action", category="text", label="a:text:0")

        assert cluster.id is not None
        assert cluster.type == "action"
        assert cluster.category == "text"
        assert cluster.label == "a:text:0"

    async def test_delete_by_type_category(self, db_session):
        """Deletes only clusters matching type + category."""
        repo = ClusterRepository(db_session)
        keep = await repo.create(type="observation", category="text", label="o:text:0")
        to_delete = await repo.create(type="action", category="text", label="a:text:0")
        await db_session.flush()

        await repo.delete_by_type_category("action", "text")
        await db_session.flush()


        remaining = (await db_session.execute(
            select(Cluster.id).where(Cluster.id.in_([keep.id, to_delete.id]))
        )).scalars().all()

        assert keep.id in remaining
        assert to_delete.id not in remaining

    async def test_get_categories_alphabetically_sorted_per_type(self, db_session):
        """``get_categories`` returns DISTINCT categories ``ORDER BY
        category`` for one cluster type — the SQL invariant that
        TokenAssigner relies on for stable per-cat noise ids
        (encode_noise_token(idx) is keyed off the position in this list).
        Insert in non-alpha order to ensure the sort actually fires.
        """
        repo = ClusterRepository(db_session)
        # Mixed types; categories repeat within a type so DISTINCT must dedup.
        for type_, cat, label in [
            ("action", "zeta", "a:zeta:0"),
            ("action", "alpha", "a:alpha:0"),
            ("action", "alpha", "a:alpha:1"),  # dup category → single output
            ("action", "mu", "a:mu:0"),
            ("observation", "obs_z", "o:obs_z:0"),
            ("observation", "obs_a", "o:obs_a:0"),
        ]:
            await repo.create(type=type_, category=cat, label=label)
        await db_session.flush()

        action_cats = await repo.get_categories("action")
        obs_cats = await repo.get_categories("observation")

        assert action_cats == ["alpha", "mu", "zeta"]
        assert obs_cats == ["obs_a", "obs_z"]

    async def test_get_categories_returns_empty_for_unknown_type(self, db_session):
        repo = ClusterRepository(db_session)
        await repo.create(type="action", category="exec", label="a:exec:0")
        await db_session.flush()
        assert await repo.get_categories("nonexistent") == []


@pytest.mark.asyncio(loop_scope="session")
class TestTrajectoryPathDeleteAll:

    async def test_deletes_all_paths(self, db_session):
        """Removes all trajectory path rows."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        obs = Message(trajectory_id=tid, role="user", content=[], index=0)
        db_session.add(obs)
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        await repo.create(trajectory_id=tid, from_observation_id=obs.id)
        await db_session.flush()

        await repo.delete_all()
        await db_session.flush()


        count = (await db_session.execute(
            select(TrajectoryPath)
        )).scalars().all()
        assert len(count) == 0


@pytest.mark.asyncio(loop_scope="session")
class TestTrajectoryPathCreate:

    async def test_create_first_observation(self, db_session):
        """First observation in trajectory — pending path, no action."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        obs = Message(trajectory_id=tid, role="user", content=[], index=0)
        db_session.add(obs)
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        path = await repo.create(
            trajectory_id=tid, from_observation_id=obs.id,
            trace=["o:text:0"],
        )

        assert path.trajectory_id == tid
        assert path.from_observation_id == obs.id
        assert path.action_message_id is None
        assert path.to_observation_id is None
        assert path.trace == ["o:text:0"]
        assert path.trace_tokens is None
        assert path.parallel_group is None

    async def test_create_with_action_and_tokens(self, db_session):
        """Closed path with action, to_observation, trace_tokens."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        obs = Message(trajectory_id=tid, role="user", content=[], index=0)
        act = Message(trajectory_id=tid, role="assistant", content=[], index=1)
        obs2 = Message(trajectory_id=tid, role="user", content=[], index=2)
        db_session.add_all([obs, act, obs2])
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        path = await repo.create(
            trajectory_id=tid, from_observation_id=obs.id,
            trace=["o:text:0", "a:text:0", "o:text:1"],
            trace_tokens=[3, 7],
            action_message_id=act.id,
            to_observation_id=obs2.id,
        )

        assert path.action_message_id == act.id
        assert path.to_observation_id == obs2.id
        assert path.trace == ["o:text:0", "a:text:0", "o:text:1"]
        assert path.trace_tokens == [3, 7]
        assert path.parallel_group is None

    async def test_create_with_parallel_group(self, db_session):
        """Parallel tool-call batch — all N paths share the same parallel_group
        (= the assistant message's index)."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        obs = Message(trajectory_id=tid, role="user", content=[], index=0)
        act = Message(
            trajectory_id=tid, role="assistant", content=[
                {"type": "tool_call", "id": "c1", "tool_name": "x", "input": {}},
                {"type": "tool_call", "id": "c2", "tool_name": "y", "input": {}},
            ],
            index=1,
        )
        resp_a = Message(trajectory_id=tid, role="tool", content=[], index=2)
        resp_b = Message(trajectory_id=tid, role="tool", content=[], index=3)
        db_session.add_all([obs, act, resp_a, resp_b])
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        p1 = await repo.create(
            trajectory_id=tid, from_observation_id=obs.id,
            action_message_id=act.id, to_observation_id=resp_a.id,
            parallel_group=act.index,
        )
        p2 = await repo.create(
            trajectory_id=tid, from_observation_id=obs.id,
            action_message_id=act.id, to_observation_id=resp_b.id,
            parallel_group=act.index,
        )

        assert p1.parallel_group == act.index
        assert p2.parallel_group == act.index
        assert p1.action_message_id == p2.action_message_id == act.id
        assert p1.to_observation_id != p2.to_observation_id

    async def test_constraint_rejects_to_obs_without_action(self, db_session):
        """CHECK constraint rejects to_observation_id set while action_message_id is NULL."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        obs0 = Message(trajectory_id=tid, role="user", content=[], index=0)
        obs1 = Message(trajectory_id=tid, role="user", content=[], index=1)
        db_session.add_all([obs0, obs1])
        await db_session.flush()

        db_session.add(TrajectoryPath(
            trajectory_id=tid,
            from_observation_id=obs0.id,
            action_message_id=None,
            to_observation_id=obs1.id,
        ))
        with pytest.raises(Exception, match="ck_tp_resolved_or_pending"):
            await db_session.flush()
        await db_session.rollback()


@pytest.mark.asyncio(loop_scope="session")
class TestTrajectoryPathUpdate:

    async def test_updates_fields(self, db_session):
        """Generic update sets arbitrary fields on a path row."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        obs1 = Message(trajectory_id=tid, role="user", content=[], index=0)
        act = Message(trajectory_id=tid, role="assistant", content=[], index=1)
        obs2 = Message(trajectory_id=tid, role="user", content=[], index=2)
        db_session.add_all([obs1, act, obs2])
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        path = await repo.create(
            trajectory_id=tid, from_observation_id=obs1.id, trace=["o:text:0"],
        )
        await repo.update(path.id, action_message_id=act.id, to_observation_id=obs2.id)
        await db_session.flush()

        await db_session.refresh(path)
        assert path.action_message_id == act.id
        assert path.to_observation_id == obs2.id


@pytest.mark.asyncio(loop_scope="session")
class TestTrajectoryPathGetLast:

    async def test_returns_most_recent(self, db_session):
        """Returns the last created path for a trajectory."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        obs1 = Message(trajectory_id=tid, role="user", content=[], index=0)
        obs2 = Message(trajectory_id=tid, role="user", content=[], index=1)
        db_session.add_all([obs1, obs2])
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        await repo.create(
            trajectory_id=tid, from_observation_id=obs1.id, trace=["o:text:0"],
        )
        second = await repo.create(
            trajectory_id=tid, from_observation_id=obs2.id,
            trace=["o:text:0", "a:text:0", "o:text:1"],
        )

        last = await repo.get_last(tid)
        assert last.id == second.id

    async def test_returns_none_for_empty(self, db_session):
        """Returns None when trajectory has no paths."""
        repo = TrajectoryPathRepository(db_session)
        assert await repo.get_last(uuid4()) is None


@pytest.mark.asyncio(loop_scope="session")
class TestGetClusterInfo:

    async def test_returns_id_and_label(self, db_session):
        """Returns (cluster_id, label) for a clustered message."""
        cluster = Cluster(type="observation", category="text", label="booking_request")
        db_session.add(cluster)
        await db_session.flush()

        msg = Message(
            trajectory_id=uuid4(), role="user", content=[],
            cluster_id=cluster.id,
        )
        db_session.add(msg)
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        assert await repo.get_cluster_info(msg.id) == (cluster.id, "booking_request")

    async def test_returns_fallback_for_unclustered(self, db_session):
        """Returns (None, fallback_label) when message has no cluster but has category."""
        msg = Message(trajectory_id=uuid4(), role="user", content=[], category="text")
        db_session.add(msg)
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        assert await repo.get_cluster_info(msg.id) == (None, "o:text:?")


@pytest.mark.asyncio(loop_scope="session")
class TestTrajectoryStatusTrigger:

    async def test_cascade_on_trajectory_update(self, db_session):
        """Updating trajectory.status cascades to trajectory_paths.trajectory_status."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        obs = Message(trajectory_id=tid, role="user", content=[], index=0)
        db_session.add(obs)
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        path = await repo.create(
            trajectory_id=tid, from_observation_id=obs.id, trace=["o:text:0"],
        )
        await db_session.commit()

        assert path.trajectory_status == "pending"

        # Update trajectory status → trigger cascades
        traj_repo = TrajectoryRepository(db_session)
        await traj_repo.update(tid, status="success")
        await db_session.commit()

        await db_session.refresh(path)
        assert path.trajectory_status == "success"

    async def test_cascade_to_multiple_paths(self, db_session):
        """Trigger updates all paths belonging to the trajectory."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        obs1 = Message(trajectory_id=tid, role="user", content=[], index=0)
        act = Message(trajectory_id=tid, role="assistant", content=[], index=1)
        obs2 = Message(trajectory_id=tid, role="user", content=[], index=2)
        db_session.add_all([obs1, act, obs2])
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        p1 = await repo.create(
            trajectory_id=tid, from_observation_id=obs1.id, trace=["o:0"],
            action_message_id=act.id, to_observation_id=obs2.id,
        )
        p2 = await repo.create(
            trajectory_id=tid, from_observation_id=obs2.id,
            trace=["o:0", "a:0", "o:1"],
        )
        await db_session.commit()

        traj_repo = TrajectoryRepository(db_session)
        await traj_repo.update(tid, status="failure")
        await db_session.commit()

        await db_session.refresh(p1)
        await db_session.refresh(p2)
        assert p1.trajectory_status == "failure"
        assert p2.trajectory_status == "failure"


@pytest.mark.asyncio(loop_scope="session")
class TestSyncTrajectoryStatus:

    async def test_sync_updates_all_paths(self, db_session):
        """sync_trajectory_status bulk-updates from trajectories table."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid, status="success"))
        obs = Message(trajectory_id=tid, role="user", content=[], index=0)
        db_session.add(obs)
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        path = await repo.create(
            trajectory_id=tid, from_observation_id=obs.id, trace=["o:text:0"],
        )
        await db_session.flush()

        # Path defaults to "pending"
        assert path.trajectory_status == "pending"

        await repo.sync_trajectory_status()
        await db_session.commit()

        await db_session.refresh(path)
        assert path.trajectory_status == "success"


@pytest.mark.asyncio(loop_scope="session")
class TestTrajectoryWindowLSH:
    """Integration tests for the LSH band index — bulk_insert + lookup
    against a real Postgres table. Each test seeds rows so the SQL
    aggregation, step filter, exclude clause, and ``top_uniq`` cap are
    exercised against an actual query plan, not the in-memory stub.
    """

    async def _insert_rows(
        self, repo: TrajectoryWindowLSHRepository,
        rows: list[tuple], db_session,
    ) -> None:
        # FK to trajectories — seed parent rows first.
        seen: set[UUID] = set()
        for tid, *_ in rows:
            if tid not in seen:
                db_session.add(Trajectory(id=tid, status="success"))
                seen.add(tid)
        await db_session.flush()
        await repo.bulk_insert(rows)
        await db_session.flush()

    async def test_bulk_insert_then_lookup_returns_candidate(self, db_session):
        repo = TrajectoryWindowLSHRepository(db_session)
        tid = uuid4()
        # One window at center=5 with two band hits.
        rows = [(tid, 5, 0, 111), (tid, 5, 1, 222)]
        await self._insert_rows(repo, rows, db_session)

        results = await repo.lookup(
            [(0, 111), (1, 222)],
            step_min=0, step_max=10, top_uniq=5,
        )
        assert results == [(tid, pytest.approx(2.0))]

    async def test_lookup_filters_by_step_range(self, db_session):
        repo = TrajectoryWindowLSHRepository(db_session)
        tid = uuid4()
        rows = [
            (tid, 5, 0, 111),    # in range
            (tid, 99, 0, 111),   # out of range
        ]
        await self._insert_rows(repo, rows, db_session)

        results = await repo.lookup(
            [(0, 111)], step_min=0, step_max=10, top_uniq=5,
        )
        # Only the in-range window contributes — band_count = 1.
        assert results == [(tid, pytest.approx(1.0))]

    async def test_lookup_excludes_trajectory_id(self, db_session):
        repo = TrajectoryWindowLSHRepository(db_session)
        keep = uuid4()
        drop = uuid4()
        rows = [
            (keep, 5, 0, 111),
            (drop, 5, 0, 111),
        ]
        await self._insert_rows(repo, rows, db_session)

        results = await repo.lookup(
            [(0, 111)], step_min=0, step_max=10, top_uniq=5,
            exclude_trajectory_id=drop,
        )
        assert [tid for tid, _ in results] == [keep]

    async def test_aggregation_min_distance_picks_max_count(self, db_session):
        """``min_distance`` returns ``max(band_count)`` per candidate —
        a trajectory with one strong window outranks one whose windows
        are uniformly weaker but average higher."""
        repo = TrajectoryWindowLSHRepository(db_session)
        strong = uuid4()
        average = uuid4()
        rows = [
            # strong: one perfect window (3 band hits), one zero-hit window.
            (strong, 5, 0, 111), (strong, 5, 1, 222), (strong, 5, 2, 333),
            (strong, 6, 0, 999),  # zero match-band counted but not via these pairs
            # average: two windows, each with 2 band hits.
            (average, 5, 0, 111), (average, 5, 1, 222),
            (average, 6, 0, 111), (average, 6, 1, 222),
        ]
        await self._insert_rows(repo, rows, db_session)

        results = await repo.lookup(
            [(0, 111), (1, 222), (2, 333)],
            step_min=0, step_max=10, top_uniq=5,
            aggregation="min_distance",
        )
        scores = dict(results)
        # strong's best window has 3 hits; average's best has 2.
        assert scores[strong] == pytest.approx(3.0)
        assert scores[average] == pytest.approx(2.0)
        assert results[0][0] == strong

    async def test_aggregation_mean_averages_band_counts(self, db_session):
        repo = TrajectoryWindowLSHRepository(db_session)
        tid = uuid4()
        # Two windows: one with 2 hits, one with 1 hit. Mean = 1.5.
        rows = [
            (tid, 5, 0, 111), (tid, 5, 1, 222),  # 2 hits
            (tid, 6, 0, 111),                    # 1 hit
        ]
        await self._insert_rows(repo, rows, db_session)

        results = await repo.lookup(
            [(0, 111), (1, 222)], step_min=0, step_max=10, top_uniq=5,
            aggregation="mean",
        )
        assert results == [(tid, pytest.approx(1.5))]

    async def test_top_uniq_caps_results(self, db_session):
        repo = TrajectoryWindowLSHRepository(db_session)
        tids = [uuid4() for _ in range(5)]
        rows = [(t, 5, 0, 111) for t in tids]
        await self._insert_rows(repo, rows, db_session)

        results = await repo.lookup(
            [(0, 111)], step_min=0, step_max=10, top_uniq=3,
        )
        assert len(results) == 3

    async def test_empty_band_pairs_returns_empty(self, db_session):
        repo = TrajectoryWindowLSHRepository(db_session)
        results = await repo.lookup(
            [], step_min=0, step_max=10, top_uniq=5,
        )
        assert results == []

    async def test_delete_for_trajectories_removes_rows(self, db_session):
        repo = TrajectoryWindowLSHRepository(db_session)
        a, b = uuid4(), uuid4()
        rows = [(a, 5, 0, 111), (b, 5, 0, 111)]
        await self._insert_rows(repo, rows, db_session)
        await repo.delete_for_trajectories([a])
        await db_session.flush()

        results = await repo.lookup(
            [(0, 111)], step_min=0, step_max=10, top_uniq=5,
        )
        assert [tid for tid, _ in results] == [b]


@pytest.mark.asyncio(loop_scope="session")
class TestCollectActObs:
    """``TrajectoryPathRepository.collect_act_obs`` returns the
    distinct (a_cluster, o_cluster) act_obs pairs across completed
    paths, sorted by COUNT(*) DESC with NULLS LAST tie-break — the
    density prior the tokenizer relies on. Tests against a real
    Postgres so the SQL ordering and NULL-side handling are exercised
    as written.
    """

    async def _make_msg(
        self, db_session, *, tid: UUID, idx: int, role: str,
        cluster: Cluster | None = None,
    ) -> Message:
        m = Message(
            trajectory_id=tid, role=role, content=[], index=idx,
            cluster_id=cluster.id if cluster else None,
            cluster_type=cluster.type if cluster else None,
            category=cluster.category if cluster else None,
        )
        db_session.add(m)
        await db_session.flush()
        return m

    async def _make_cluster(
        self, db_session, *, type_: str, category: str, label: str,
    ) -> Cluster:
        c = Cluster(type=type_, category=category, label=label)
        db_session.add(c)
        await db_session.flush()
        return c

    async def _add_path(
        self, repo: TrajectoryPathRepository, *,
        tid: UUID, obs: Message, act: Message, to_obs: Message,
    ) -> None:
        await repo.create(
            trajectory_id=tid,
            from_observation_id=obs.id,
            action_message_id=act.id,
            to_observation_id=to_obs.id,
        )

    async def test_pairs_sorted_count_desc(self, db_session):
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        await db_session.flush()
        c_a = await self._make_cluster(
            db_session, type_="action", category="exec", label="A",
        )
        c_o1 = await self._make_cluster(
            db_session, type_="observation", category="text", label="O1",
        )
        c_o2 = await self._make_cluster(
            db_session, type_="observation", category="text", label="O2",
        )

        # Three paths with pair (A, O1) and one path with (A, O2).
        repo = TrajectoryPathRepository(db_session)
        obs0 = await self._make_msg(
            db_session, tid=tid, idx=0, role="user", cluster=c_o1,
        )
        for i in range(3):
            act = await self._make_msg(
                db_session, tid=tid, idx=1 + 2 * i, role="assistant",
                cluster=c_a,
            )
            to_obs = await self._make_msg(
                db_session, tid=tid, idx=2 + 2 * i, role="user",
                cluster=c_o1,
            )
            await self._add_path(
                repo, tid=tid, obs=obs0, act=act, to_obs=to_obs,
            )
        act_rare = await self._make_msg(
            db_session, tid=tid, idx=99, role="assistant", cluster=c_a,
        )
        to_rare = await self._make_msg(
            db_session, tid=tid, idx=100, role="user", cluster=c_o2,
        )
        await self._add_path(
            repo, tid=tid, obs=obs0, act=act_rare, to_obs=to_rare,
        )

        pairs = await repo.collect_act_obs()
        # The high-count pair must come first.
        assert pairs[0].a_cluster_id == c_a.id
        assert pairs[0].o_cluster_id == c_o1.id
        assert pairs[1].o_cluster_id == c_o2.id

    async def test_null_side_pairs_included_after_full_pairs(self, db_session):
        """One-side-noise pairs (one cluster_id is NULL) are included
        in the result so they can seed per-category noise centroids in
        the tokenizer — but the ``NULLS LAST`` tie-break sinks them
        below fully-clustered pairs at the same count."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        await db_session.flush()
        c_a = await self._make_cluster(
            db_session, type_="action", category="exec", label="A",
        )
        c_o = await self._make_cluster(
            db_session, type_="observation", category="text", label="O",
        )
        # Noise-side observation: category is set but cluster_id is NULL.
        obs_noise = Message(
            trajectory_id=tid, role="user", content=[], index=10,
            cluster_id=None, cluster_type="observation", category="text",
        )
        db_session.add(obs_noise)
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        obs0 = await self._make_msg(
            db_session, tid=tid, idx=0, role="user", cluster=c_o,
        )
        # One fully-clustered path (count=1) and one noise-side path
        # (count=1) — same count, NULLS LAST must put fully-clustered first.
        act_full = await self._make_msg(
            db_session, tid=tid, idx=1, role="assistant", cluster=c_a,
        )
        to_full = await self._make_msg(
            db_session, tid=tid, idx=2, role="user", cluster=c_o,
        )
        await self._add_path(
            repo, tid=tid, obs=obs0, act=act_full, to_obs=to_full,
        )

        act_noise = await self._make_msg(
            db_session, tid=tid, idx=3, role="assistant", cluster=c_a,
        )
        await self._add_path(
            repo, tid=tid, obs=obs0, act=act_noise, to_obs=obs_noise,
        )

        pairs = await repo.collect_act_obs()
        # Both pairs present.
        assert len(pairs) == 2
        # NULLS LAST: full pair first.
        assert pairs[0].o_cluster_id is not None
        assert pairs[1].o_cluster_id is None

    async def test_both_side_null_pairs_excluded(self, db_session):
        """When BOTH sides are noise (cluster_id NULL), the path
        contributes neither tokenization signal nor centroid mass —
        excluded by the WHERE clause."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        await db_session.flush()
        c_a = await self._make_cluster(
            db_session, type_="action", category="exec", label="A",
        )
        c_o = await self._make_cluster(
            db_session, type_="observation", category="text", label="O",
        )

        repo = TrajectoryPathRepository(db_session)
        obs0 = await self._make_msg(
            db_session, tid=tid, idx=0, role="user", cluster=c_o,
        )
        # One fully-clustered path.
        act_full = await self._make_msg(
            db_session, tid=tid, idx=1, role="assistant", cluster=c_a,
        )
        to_full = await self._make_msg(
            db_session, tid=tid, idx=2, role="user", cluster=c_o,
        )
        await self._add_path(
            repo, tid=tid, obs=obs0, act=act_full, to_obs=to_full,
        )
        # Both-noise path: action and observation both lack cluster_id
        # but still carry categories.
        act_noise = Message(
            trajectory_id=tid, role="assistant", content=[], index=3,
            cluster_id=None, cluster_type="action", category="exec",
        )
        to_noise = Message(
            trajectory_id=tid, role="user", content=[], index=4,
            cluster_id=None, cluster_type="observation", category="text",
        )
        db_session.add_all([act_noise, to_noise])
        await db_session.flush()
        await self._add_path(
            repo, tid=tid, obs=obs0, act=act_noise, to_obs=to_noise,
        )

        pairs = await repo.collect_act_obs()
        assert len(pairs) == 1
        assert pairs[0].a_cluster_id == c_a.id
        assert pairs[0].o_cluster_id == c_o.id
