from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from episodiq.inference import EmbedderClient
from episodiq.server.management import router as management_router
from episodiq.workflows.base import Workflow
from episodiq.workflows.trajectory_manager import TrajectoryManager


class DuplicateRouteError(Exception):
    pass


def create_app(
    workflows: list[Workflow],
    embedder: EmbedderClient | None = None,
    engine: AsyncEngine | None = None,
    trajectory_manager: TrajectoryManager | None = None,
    session_factory: async_sessionmaker | None = None,
) -> FastAPI:
    seen_routes: dict[str, str] = {}

    for workflow in workflows:
        adapter = workflow.api_adapter
        for route in adapter.routes:
            full_path = f"{adapter.mount_path}{route.path}"
            for method in route.methods:
                route_key = f"{method} {full_path}"
                if route_key in seen_routes:
                    raise DuplicateRouteError(
                        f"Route '{route_key}' already registered by adapter "
                        f"'{seen_routes[route_key]}', cannot register for '{adapter.id}'"
                    )
                seen_routes[route_key] = adapter.id

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if embedder:
            await embedder.startup()
        for workflow in workflows:
            await workflow.api_adapter.startup()
        yield
        if trajectory_manager:
            await trajectory_manager.shutdown()
        for workflow in workflows:
            await workflow.api_adapter.shutdown()
        if embedder:
            await embedder.shutdown()
        if engine:
            await engine.dispose()

    app = FastAPI(title="Episodiq Proxy", lifespan=lifespan)
    app.state.session_factory = session_factory

    for workflow in workflows:
        app.include_router(workflow.build_router())

    app.include_router(management_router)

    return app
