import gzip
import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest
import pytest_asyncio
from dotenv import load_dotenv

TESTS_DIR = Path(__file__).parent
ROOT_DIR = TESTS_DIR.parent

# Load .env.test BEFORE any episodiq imports (models.py calls get_config() at import time)
load_dotenv(ROOT_DIR / ".env.test")

from episodiq.api_adapters.base import ApiAdapterConfig, ApiAdapter, Route  # noqa: E402
from episodiq.api_adapters.mixins import OpenAIChatMixin  # noqa: E402
from episodiq.config import get_config  # noqa: E402
from episodiq.storage.postgres.models import Base  # noqa: E402
from alembic.config import Config  # noqa: E402
from alembic import command  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession  # noqa: E402
from sqlalchemy import event, text  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402


def mock_session_factory():
    """Create a mock session factory that tracks calls."""
    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    @asynccontextmanager
    async def factory():
        yield session

    factory.session = session
    return factory


@pytest.fixture(scope="session")
def db_engine():
    alembic_cfg = Config(str(ROOT_DIR / "alembic.ini"))
    command.upgrade(alembic_cfg, "head")

    # Use NullPool to allow engine usage across different event loops
    engine = create_async_engine(get_config().get_database_url(), poolclass=NullPool)

    yield engine


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def db_session(db_engine):
    """Session with transaction rollback for test isolation."""
    async with db_engine.connect() as conn:
        await conn.begin()
        await conn.begin_nested()

        session = AsyncSession(conn, expire_on_commit=False)

        @event.listens_for(session.sync_session, "after_transaction_end")
        def end_savepoint(sync_session, transaction):
            if conn.closed:
                return
            if not conn.in_nested_transaction():
                conn.sync_connection.begin_nested()

        yield session

        await session.close()


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def session_factory(db_engine):
    """Session factory for creating new sessions."""
    factory = async_sessionmaker(db_engine, expire_on_commit=False)

    yield factory

    async with db_engine.begin() as conn:
        table_names = ', '.join([table.name for table in Base.metadata.sorted_tables])
        await conn.execute(text(f"TRUNCATE TABLE {table_names} CASCADE"))


class EchoAdapter(OpenAIChatMixin, ApiAdapter):
    @property
    def mount_path(self) -> str:
        return "/test/v1"

    @property
    def routes(self) -> list[Route]:
        return [Route("/chat", ["POST"], "chat")]


@pytest.fixture
def echo_adapter():
    config = ApiAdapterConfig(id="echo", upstream_base_url="https://test.local")
    return EchoAdapter(config)


API_KEY_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
]


def _scrub_response(response):
    return response


def _scrub_request(request):
    return request


def _make_scrubber():
    sensitive_values = [os.getenv(var) for var in API_KEY_ENV_VARS if os.getenv(var)]

    def scrub_string(s: str) -> str:
        for val in sensitive_values:
            s = s.replace(val, "FILTERED")
        return s

    return scrub_string


@pytest.fixture(scope="module")
def vcr_config():
    scrub = _make_scrubber()

    def before_record_request(request):
        if request.body:
            request.body = scrub(request.body.decode()).encode()
        return request

    def before_record_response(response):
        if response.get("body", {}).get("string"):
            body = response["body"]["string"]
            if isinstance(body, bytes):
                # Try to decompress gzip if it's gzipped
                was_gzipped = False
                try:
                    body = gzip.decompress(body).decode()
                    was_gzipped = True
                except (gzip.BadGzipFile, OSError):
                    # Not gzipped, try regular decode
                    body = body.decode()
            response["body"]["string"] = scrub(body).encode()

            # Remove content-encoding header if we decompressed
            if was_gzipped and "headers" in response:
                response["headers"].pop("Content-Encoding", None)
                response["headers"].pop("content-encoding", None)
        return response

    return {
        "filter_headers": [
            ("authorization", "FILTERED"),
            ("x-api-key", "FILTERED"),
        ],
        "record_mode": "once",
        "before_record_request": before_record_request,
        "before_record_response": before_record_response,
    }


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    test_file = Path(request.fspath)
    relative = test_file.relative_to(TESTS_DIR)
    cassette_subdir = test_file.stem.removeprefix("test_")
    return str(TESTS_DIR / "cassettes" / relative.parent / cassette_subdir)
