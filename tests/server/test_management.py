"""Integration tests for the episodiq management API."""

from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from episodiq.server.app import create_app
from episodiq.storage.postgres.models import Trajectory


def _make_app(session_factory):
    return create_app(workflows=[], session_factory=session_factory)


@pytest_asyncio.fixture(loop_scope="session")
async def client(session_factory):
    app = _make_app(session_factory)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


@pytest.mark.asyncio(loop_scope="session")
class TestUpdateTrajectory:
    async def test_success(self, client, session_factory):
        tid = uuid4()
        async with session_factory() as session:
            session.add(Trajectory(id=tid))
            await session.commit()

        resp = await client.patch(
            f"/episodiq/trajectories/{tid}",
            json={"status": "success"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"id": str(tid), "status": "success"}

        async with session_factory() as session:
            traj = await session.get(Trajectory, tid)
            assert traj.status == "success"

    async def test_failure(self, client, session_factory):
        tid = uuid4()
        async with session_factory() as session:
            session.add(Trajectory(id=tid))
            await session.commit()

        resp = await client.patch(
            f"/episodiq/trajectories/{tid}",
            json={"status": "failure"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "failure"

    async def test_not_found(self, client):
        resp = await client.patch(
            f"/episodiq/trajectories/{uuid4()}",
            json={"status": "success"},
        )
        assert resp.status_code == 404

    async def test_already_completed(self, client, session_factory):
        tid = uuid4()
        async with session_factory() as session:
            session.add(Trajectory(id=tid, status="success"))
            await session.commit()

        resp = await client.patch(
            f"/episodiq/trajectories/{tid}",
            json={"status": "failure"},
        )
        assert resp.status_code == 409

    async def test_invalid_status(self, client, session_factory):
        tid = uuid4()
        async with session_factory() as session:
            session.add(Trajectory(id=tid))
            await session.commit()

        resp = await client.patch(
            f"/episodiq/trajectories/{tid}",
            json={"status": "active"},
        )
        assert resp.status_code == 422
