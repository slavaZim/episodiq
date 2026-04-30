from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from episodiq.api_adapters import (
    AnthropicConfig,
    AnthropicMessagesAdapter,
    OpenAIConfig,
    OpenAICompletionsAdapter,
)
from episodiq.config import get_config
from episodiq.inference import Embedder, EmbedderClient
from episodiq.server.app import create_app
from episodiq.workflows import LoggingPipeline, TrajectoryManager


def build_app() -> FastAPI:
    """Composition root — creates the full dependency graph and returns a FastAPI app."""
    config = get_config()

    engine = create_async_engine(config.get_database_url())
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    embedder = Embedder(EmbedderClient(config.embedder), dims=config.message_dims)
    trajectory_manager = TrajectoryManager(config.postprocess_timeout)

    adapters = [
        OpenAICompletionsAdapter(OpenAIConfig()),
        AnthropicMessagesAdapter(AnthropicConfig()),
    ]

    workflows = [
        LoggingPipeline(
            api_adapter=adapter,
            session_factory=session_factory,
            embedder=embedder,
            trajectory_manager=trajectory_manager,
        )
        for adapter in adapters
    ]

    return create_app(
        workflows,
        embedder=embedder,
        engine=engine,
        trajectory_manager=trajectory_manager,
        session_factory=session_factory,
    )
