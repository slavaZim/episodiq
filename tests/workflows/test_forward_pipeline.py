"""Integration test: full LoggingPipeline with sync + deferred steps."""

import asyncio

import respx
from uuid import uuid4

import pytest

from episodiq.storage.postgres.models import Cluster
from episodiq.storage.postgres.repository import MessageRepository, TrajectoryPathRepository, TrajectoryRepository
from episodiq.workflows.base import Workflow
from episodiq.workflows.steps import (
    TrajectoryStep,
    SaveInputStep,
    ForwardStep,
    SaveOutputStep,
    ProcessInputStep,
    ProcessOutputStep,
    BuildPathStep,
    Step,
    StepResult,
)
from episodiq.workflows.trajectory_manager import TrajectoryManager
from episodiq.server.app import create_app
from tests.conftest import EchoAdapter, ApiAdapterConfig
from tests.helpers import MockEmbedder, run_scenario, text_to_embedding


class GateDeferredStep(Step):
    """Deferred step that blocks until gate event is set."""

    step_id = "gate"
    deferred = True
    gate: asyncio.Event  # set per-test to avoid event loop binding issues

    async def exec(self) -> StepResult:
        await self.__class__.gate.wait()
        return StepResult(passable=True)


def _make_pipeline_cls(with_gate: bool):
    """Build pipeline with optional gate step before deferred steps."""

    class TestPipeline(Workflow):
        def __init__(self, **kwargs):
            deferred = [ProcessInputStep, ProcessOutputStep, BuildPathStep]
            if with_gate:
                deferred = [GateDeferredStep] + deferred
            super().__init__(
                steps=[
                    TrajectoryStep,
                    SaveInputStep,
                    ForwardStep,
                    SaveOutputStep,
                    *deferred,
                ],
                fallback_step=ForwardStep,
                **kwargs,
            )

    return TestPipeline


@pytest.fixture
def echo_adapter():
    config = ApiAdapterConfig(id="echo", upstream_base_url="https://test.local")
    return EchoAdapter(config)


class TestLoggingPipelineIntegration:
    """Send request → get response → verify sync DB records → verify deferred completion."""

    @respx.mock
    async def test_response_before_deferred_completes(self, session_factory, echo_adapter):
        """Response is returned while deferred steps are still blocked."""
        GateDeferredStep.gate = asyncio.Event()

        embedder = MockEmbedder(embedding_fn=text_to_embedding, dims=1024)
        manager = TrajectoryManager(postprocess_timeout=10.0)

        Pipeline = _make_pipeline_cls(with_gate=True)
        pipeline = Pipeline(
            api_adapter=echo_adapter,
            session_factory=session_factory,
            embedder=embedder,
            trajectory_manager=manager,
        )

        app = create_app([pipeline])
        await echo_adapter.startup()

        trajectory_id = uuid4()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        # Response comes back while gate is closed → deferred steps blocked
        response = await run_scenario(app, trajectory_id, messages)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hi there"

        # --- Sync records exist (messages saved without embeddings) ---
        async with session_factory() as session:
            traj_repo = TrajectoryRepository(session)
            trajs = await traj_repo.find_by(id=trajectory_id)
            assert len(trajs) == 1
            assert trajs[0].status == "pending"

            msg_repo = MessageRepository(session)
            msgs = await msg_repo.find_by(trajectory_id=trajectory_id)

        assert len(msgs) == 2
        # Deferred steps haven't run yet — no embeddings
        for msg in msgs:
            assert msg.embedding is None, f"Message {msg.role} should have no embedding yet"

        # --- Release gate → deferred steps proceed ---
        GateDeferredStep.gate.set()
        await manager.shutdown()

        # --- Deferred steps completed — embeddings present ---
        async with session_factory() as session:
            msg_repo = MessageRepository(session)
            msgs = await msg_repo.find_by(trajectory_id=trajectory_id)

        for msg in msgs:
            assert msg.embedding is not None, f"Message {msg.role} index={msg.index} has no embedding"

        # Trajectory still pending (no errors)
        async with session_factory() as session:
            traj_repo = TrajectoryRepository(session)
            trajs = await traj_repo.find_by(id=trajectory_id)
            assert trajs[0].status == "pending"

        await echo_adapter.shutdown()

    @respx.mock
    async def test_deferred_fifo_across_requests(self, session_factory, echo_adapter):
        """Two requests for same trajectory → deferred steps execute in FIFO order."""
        GateDeferredStep.gate = asyncio.Event()

        embedder = MockEmbedder(embedding_fn=text_to_embedding, dims=1024)
        manager = TrajectoryManager(postprocess_timeout=10.0)

        Pipeline = _make_pipeline_cls(with_gate=True)
        pipeline = Pipeline(
            api_adapter=echo_adapter,
            session_factory=session_factory,
            embedder=embedder,
            trajectory_manager=manager,
        )

        app = create_app([pipeline])
        await echo_adapter.startup()

        # BuildPathStep skips when no clusters exist — seed one
        async with session_factory() as session:
            session.add(Cluster(type="observation", category="text", label="o:text:0"))
            await session.commit()

        trajectory_id = uuid4()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        # Two requests queued while gate is closed
        response = await run_scenario(app, trajectory_id, messages)
        assert response.status_code == 200

        # Release gate → deferred jobs execute in FIFO order
        GateDeferredStep.gate.set()
        await manager.shutdown()

        # Verify trajectory paths created in correct order
        async with session_factory() as session:
            path_repo = TrajectoryPathRepository(session)
            paths = await path_repo.find_by(trajectory_id=trajectory_id)

            msg_repo = MessageRepository(session)
            msgs = await msg_repo.find_by(trajectory_id=trajectory_id)

        # 4 messages: user, assistant, user, assistant
        assert len(msgs) == 4
        user_msgs = sorted(
            [m for m in msgs if m.role == "user"],
            key=lambda m: m.index,
        )
        assert len(user_msgs) == 2

        # 2 paths: one per user observation
        assert len(paths) == 2

        # Paths ordered by creation — first path from "hello", second from "what is 2+2?"
        p0, p1 = sorted(paths, key=lambda p: p.created_at)

        hello_msg = next(m for m in user_msgs if "hello" in str(m.content))
        question_msg = next(m for m in user_msgs if "2+2" in str(m.content))

        assert p0.from_observation_id == hello_msg.id
        assert p1.from_observation_id == question_msg.id

        # Second path closes first
        assert p0.to_observation_id == question_msg.id

        await echo_adapter.shutdown()

    @respx.mock
    async def test_deferred_error_marks_internal_error(self, session_factory, echo_adapter):
        """When deferred step fails, trajectory is marked internal_error."""
        embedder = MockEmbedder(embedding_fn=text_to_embedding, dims=1024)
        manager = TrajectoryManager(postprocess_timeout=10.0)

        Pipeline = _make_pipeline_cls(with_gate=False)
        pipeline = Pipeline(
            api_adapter=echo_adapter,
            session_factory=session_factory,
            embedder=embedder,
            trajectory_manager=manager,
        )

        app = create_app([pipeline])
        await echo_adapter.startup()

        trajectory_id = uuid4()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        response = await run_scenario(app, trajectory_id, messages)
        assert response.status_code == 200

        # Break embedder — deferred steps will fail
        async def exploding_embed(text):
            raise RuntimeError("Embedder exploded")

        embedder.embed_text = exploding_embed

        await manager.shutdown()

        async with session_factory() as session:
            traj_repo = TrajectoryRepository(session)
            trajs = await traj_repo.find_by(id=trajectory_id)
            assert trajs[0].status == "internal_error"

        await echo_adapter.shutdown()
