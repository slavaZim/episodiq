"""Integration tests for repository queries.

Requires PostgreSQL with pgvector:
    docker compose -f docker-compose.test.yml up -d
"""
from uuid import uuid4

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
        """Returns user messages with category='text' and embedding."""
        tid = uuid4()
        emb = random_embedding()
        db_session.add_all([
            Message(
                trajectory_id=tid, role="user",
                content=[{"type": "text", "text": "hello"}],
                embedding=emb, category="text", cluster_type="observation",
            ),
            Message(
                trajectory_id=tid, role="assistant",
                content=[{"type": "text", "text": "hi"}],
                embedding=random_embedding(), category="text", cluster_type="action",
            ),
        ])
        await db_session.commit()

        repo = MessageRepository(db_session)
        rows = await repo.get_messages_for_clustering("observation", "text")

        assert any(np.allclose(r.embedding, emb, atol=1e-5) for r in rows)
        assert all(r.role in ("user", "tool") for r in rows)

    async def test_observation_tool(self, db_session):
        """Returns tool messages with matching category."""
        tid = uuid4()
        emb = random_embedding()
        db_session.add_all([
            Message(
                trajectory_id=tid, role="tool",
                content=[{"type": "tool_response", "id": "c1", "tool_name": "bash", "tool_response": "ok"}],
                embedding=emb, category="bash", cluster_type="observation",
            ),
            Message(
                trajectory_id=tid, role="user",
                content=[{"type": "text", "text": "hi"}],
                embedding=random_embedding(), category="text", cluster_type="observation",
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
        cats = await repo.get_distinct_categories("action")

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
        cats = await repo.get_distinct_categories("observation")

        assert "text" in cats
        assert "bash" in cats
        assert "editor" not in cats

    async def test_excludes_null_categories(self, db_session):
        """Messages without category are excluded."""
        tid = uuid4()
        db_session.add(Message(trajectory_id=tid, role="assistant", content=[], category=None, cluster_type="action"))
        await db_session.commit()

        repo = MessageRepository(db_session)
        cats = await repo.get_distinct_categories("action")
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
        await repo.create(tid, obs.id)
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
        """First observation in trajectory — no profile, no action."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        obs = Message(trajectory_id=tid, role="user", content=[], index=0)
        db_session.add(obs)
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        path = await repo.create(tid, obs.id, trace=["o:text:0"])

        assert path.trajectory_id == tid
        assert path.from_observation_id == obs.id
        assert path.action_message_id is None
        assert path.to_observation_id is None
        assert path.trace == ["o:text:0"]
        assert path.transition_profile is None
        assert path.profile_embed is None

    async def test_create_with_profile_and_action(self, db_session):
        """Full path: profile, embed, trace, action, to_observation."""
        tid = uuid4()
        db_session.add(Trajectory(id=tid))
        obs = Message(trajectory_id=tid, role="user", content=[], index=0)
        act = Message(trajectory_id=tid, role="assistant", content=[], index=1)
        obs2 = Message(trajectory_id=tid, role="user", content=[], index=2)
        db_session.add_all([obs, act, obs2])
        await db_session.flush()

        profile = {"o:text:0.a:text:0.o:text:1": 1.0}
        embed = [0.1] * 2000

        repo = TrajectoryPathRepository(db_session)
        path = await repo.create(
            tid, obs.id,
            transition_profile=profile,
            profile_embed=embed,
            trace=["o:text:0", "a:text:0", "o:text:1"],
            action_message_id=act.id,
            to_observation_id=obs2.id,
        )

        assert path.action_message_id == act.id
        assert path.to_observation_id == obs2.id
        assert path.transition_profile == profile
        assert path.trace == ["o:text:0", "a:text:0", "o:text:1"]
        assert len(path.profile_embed) == 2000

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
        path = await repo.create(tid, obs1.id, trace=["o:text:0"])
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
        await repo.create(tid, obs1.id, trace=["o:text:0"])
        second = await repo.create(tid, obs2.id, trace=["o:text:0", "a:text:0", "o:text:1"])

        last = await repo.get_last(tid)
        assert last.id == second.id

    async def test_returns_none_for_empty(self, db_session):
        """Returns None when trajectory has no paths."""
        repo = TrajectoryPathRepository(db_session)
        assert await repo.get_last(uuid4()) is None


@pytest.mark.asyncio(loop_scope="session")
class TestGetClusterLabel:

    async def test_returns_label(self, db_session):
        """Returns cluster label for a clustered message."""
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
        assert await repo.get_cluster_label(msg.id) == "booking_request"

    async def test_returns_fallback_for_unclustered(self, db_session):
        """Returns fallback label when message has no cluster but has category."""
        msg = Message(trajectory_id=uuid4(), role="user", content=[], category="text")
        db_session.add(msg)
        await db_session.flush()

        repo = TrajectoryPathRepository(db_session)
        assert await repo.get_cluster_label(msg.id) == "o:text:?"


@pytest.mark.asyncio(loop_scope="session")
class TestPrefetchSimilar:

    async def _make_path(self, db_session, *, tid, profile_embed, with_action=True, status="success"):
        """Helper: create trajectory + path with profile_embed."""
        db_session.add(Trajectory(id=tid, status=status))

        obs = Message(trajectory_id=tid, role="user", content=[], index=0)
        db_session.add(obs)
        await db_session.flush()

        action_id = None
        to_obs_id = None
        if with_action:
            act_cluster = Cluster(type="action", category="text", label="a:text:0")
            db_session.add(act_cluster)
            await db_session.flush()

            act = Message(
                trajectory_id=tid, role="assistant", content=[], index=1,
                cluster_id=act_cluster.id,
            )
            to_obs = Message(trajectory_id=tid, role="user", content=[], index=2)
            db_session.add_all([act, to_obs])
            await db_session.flush()
            action_id = act.id
            to_obs_id = to_obs.id

        repo = TrajectoryPathRepository(db_session)
        return await repo.create(
            tid, obs.id,
            profile_embed=profile_embed,
            transition_profile={"t": 1.0},
            trace=["o:text:0"],
            action_message_id=action_id,
            to_observation_id=to_obs_id,
            trajectory_status=status,
        )

    async def test_finds_similar_by_cosine(self, db_session):
        """Returns paths ordered by cosine distance."""
        query_embed = np.zeros(2000, dtype=np.float32)
        query_embed[0] = 1.0

        close_embed = np.zeros(2000, dtype=np.float32)
        close_embed[0] = 0.9
        close_embed[1] = 0.1

        far_embed = np.zeros(2000, dtype=np.float32)
        far_embed[999] = 1.0

        close_path = await self._make_path(db_session, tid=uuid4(), profile_embed=close_embed.tolist())
        far_path = await self._make_path(db_session, tid=uuid4(), profile_embed=far_embed.tolist())
        await db_session.commit()

        repo = TrajectoryPathRepository(db_session)
        results = await repo.prefetch_similar(
            profile_embed=query_embed.tolist(),
            exclude_trajectory_id=uuid4(),
            limit=10,
        )

        ids = [r.id for r in results]
        assert close_path.id in ids
        if far_path.id in ids:
            assert ids.index(close_path.id) < ids.index(far_path.id)

    async def test_excludes_own_trajectory(self, db_session):
        """Paths from excluded trajectory are not returned."""
        embed = np.zeros(2000, dtype=np.float32)
        embed[0] = 1.0

        tid = uuid4()
        await self._make_path(db_session, tid=tid, profile_embed=embed.tolist())
        await db_session.commit()

        repo = TrajectoryPathRepository(db_session)
        results = await repo.prefetch_similar(
            profile_embed=embed.tolist(),
            exclude_trajectory_id=tid,
            limit=10,
        )
        assert all(r.trajectory_id != tid for r in results)

    async def test_excludes_pending_paths(self, db_session):
        """Paths without to_observation_id are excluded."""
        embed = np.zeros(2000, dtype=np.float32)
        embed[0] = 1.0

        pending = await self._make_path(db_session, tid=uuid4(), profile_embed=embed.tolist(), with_action=False)
        await db_session.commit()

        repo = TrajectoryPathRepository(db_session)
        results = await repo.prefetch_similar(
            profile_embed=embed.tolist(),
            exclude_trajectory_id=uuid4(),
            limit=10,
        )
        assert all(r.id != pending.id for r in results)

    async def test_eager_loads_relationships(self, db_session):
        """Returned paths have trajectory and action_message.cluster loaded."""
        embed = np.zeros(2000, dtype=np.float32)
        embed[0] = 1.0

        path = await self._make_path(db_session, tid=uuid4(), profile_embed=embed.tolist())
        await db_session.commit()

        repo = TrajectoryPathRepository(db_session)
        results = await repo.prefetch_similar(
            profile_embed=embed.tolist(),
            exclude_trajectory_id=uuid4(),
            limit=10,
        )

        match = next((r for r in results if r.id == path.id), None)
        assert match is not None
        assert match.trajectory is not None
        assert match.action_message is not None
        assert match.action_message.cluster is not None

    async def test_excludes_pending_trajectory_status(self, db_session):
        """Paths from pending trajectories are not returned."""
        embed = np.zeros(2000, dtype=np.float32)
        embed[0] = 1.0

        pending = await self._make_path(
            db_session, tid=uuid4(), profile_embed=embed.tolist(), status="pending",
        )
        await db_session.commit()

        repo = TrajectoryPathRepository(db_session)
        results = await repo.prefetch_similar(
            profile_embed=embed.tolist(),
            exclude_trajectory_id=uuid4(),
            limit=10,
        )
        assert all(r.id != pending.id for r in results)


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
        path = await repo.create(tid, obs.id, trace=["o:text:0"])
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
            tid, obs1.id, trace=["o:0"],
            action_message_id=act.id, to_observation_id=obs2.id,
        )
        p2 = await repo.create(tid, obs2.id, trace=["o:0", "a:0", "o:1"])
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
        path = await repo.create(tid, obs.id, trace=["o:text:0"])
        await db_session.flush()

        # Path defaults to "pending"
        assert path.trajectory_status == "pending"

        await repo.sync_trajectory_status()
        await db_session.commit()

        await db_session.refresh(path)
        assert path.trajectory_status == "success"


