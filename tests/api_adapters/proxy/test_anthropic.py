"""Anthropic proxy tests: unit tests + VCR pipeline tests with real DB."""

import os

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from episodiq.api_adapters import ApiAdapterConfig, AnthropicConfig, AnthropicMessagesAdapter
from episodiq.api_adapters.base import CanonicalMessage, CanonicalToolMessage
from episodiq.server.app import create_app
from episodiq.storage.postgres.models import Message
from episodiq.workflows import LoggingPipeline
from tests.helpers import MockEmbedder

TOOL_DEFS = [
    {
        "name": "hello_world",
        "description": "Says hello",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "fizzbuzz",
        "description": "Does fizzbuzz",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# --- Unit tests (no VCR, no DB) ---

def test_client_auth_header_passthrough():
    """Client x-api-key passes through to upstream."""
    config = ApiAdapterConfig(id="anthropic", upstream_base_url="https://api.anthropic.com")
    adapter = AnthropicMessagesAdapter(config)
    headers = adapter.build_request_headers({
        "x-api-key": "sk-ant-test-123",
        "anthropic-version": "2023-06-01",
    })
    assert headers["x-api-key"] == "sk-ant-test-123"
    assert headers["anthropic-version"] == "2023-06-01"


def test_tool_response_roundtrip():
    """Tool response round-trips through extract → canonical → adapter format."""
    adapter = AnthropicMessagesAdapter(
        ApiAdapterConfig(id="test", upstream_base_url="https://test.local"),
    )
    body = {
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_abc", "name": "hello_world", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "Hello!"},
            ]},
        ],
    }
    msgs = adapter.extract_request_messages(body)
    tool_msg = [m for m in msgs if isinstance(m, CanonicalToolMessage)][0]
    roundtripped = adapter.to_adapter_format(tool_msg)

    assert roundtripped["role"] == "user"
    assert roundtripped["content"][0]["type"] == "tool_result"
    assert roundtripped["content"][0]["tool_use_id"] == "toolu_abc"
    assert roundtripped["content"][0]["content"] == "Hello!"

    # tool_name must be resolved from assistant's tool_use
    assert tool_msg.content[0]["tool_name"] == "hello_world"


# --- VCR pipeline tests (real upstream + real DB) ---

@pytest.fixture
def adapter():
    return AnthropicMessagesAdapter(AnthropicConfig())


@pytest.fixture
def app(adapter, session_factory):
    pipeline = LoggingPipeline(
        api_adapter=adapter,
        session_factory=session_factory,
        embedder=MockEmbedder(),
    )
    return create_app([pipeline])


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_all_message_types_persisted_and_roundtrip(app, adapter, session_factory):
    """Full conversation: system + user → tool_use → tool_result → final answer.

    Verifies every message type is persisted and round-trips through
    DB → canonical → adapter format.
    """
    trajectory_id = "22222222-2222-2222-2222-222222222222"
    auth_headers = {
        "x-api-key": os.getenv("ANTHROPIC_API_KEY", "test"),
        "anthropic-version": "2023-06-01",
    }

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1) system + user with forced tool use
            r1 = await client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 100,
                    "system": "You are helpful",
                    "messages": [
                        {"role": "user", "content": "Call the hello_world tool"},
                    ],
                    "tools": [TOOL_DEFS[0]],
                    "tool_choice": {"type": "tool", "name": "hello_world"},
                },
                headers={**auth_headers, "X-Trajectory-ID": trajectory_id},
            )
            assert r1.status_code == 200, r1.text
            data1 = r1.json()
            upstream_tool_use = data1["content"]
            tool_use_block = next(b for b in upstream_tool_use if b["type"] == "tool_use")

            # 2) send tool result back, get final answer
            r2 = await client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 100,
                    "system": "You are helpful",
                    "messages": [
                        {"role": "user", "content": "Call the hello_world tool"},
                        {"role": "assistant", "content": upstream_tool_use},
                        {"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": tool_use_block["id"], "content": "Hello from tool!"},
                        ]},
                    ],
                    "tools": [TOOL_DEFS[0]],
                },
                headers={**auth_headers, "X-Trajectory-ID": trajectory_id},
            )
            assert r2.status_code == 200, r2.text
            data2 = r2.json()
            upstream_final = data2["content"]

    # Load all persisted messages ordered by index
    async with session_factory() as session:
        result = await session.execute(
            select(Message).order_by(Message.index)
        )
        db_msgs = list(result.scalars())

    roles = [m.role for m in db_msgs]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]

    by_role = {}
    for m in db_msgs:
        by_role.setdefault(m.role, []).append(m)

    # System round-trip
    sys_rt = adapter.to_adapter_format(CanonicalMessage.from_db(by_role["system"][0]))
    assert sys_rt == {"role": "system", "content": "You are helpful"}

    # User round-trip
    user_rt = adapter.to_adapter_format(CanonicalMessage.from_db(by_role["user"][0]))
    assert user_rt == {"role": "user", "content": "Call the hello_world tool"}

    # Assistant tool_use round-trip
    asst1_rt = adapter.to_adapter_format(CanonicalMessage.from_db(by_role["assistant"][0]))
    assert asst1_rt["role"] == "assistant"
    rt_tool_use = [b for b in asst1_rt["content"] if b["type"] == "tool_use"]
    assert len(rt_tool_use) >= 1
    assert rt_tool_use[0]["id"] == tool_use_block["id"]
    assert rt_tool_use[0]["name"] == "hello_world"

    # Tool result round-trip
    tool_rt = adapter.to_adapter_format(CanonicalMessage.from_db(by_role["tool"][0]))
    assert tool_rt["role"] == "user"
    assert tool_rt["content"][0]["type"] == "tool_result"
    assert tool_rt["content"][0]["tool_use_id"] == tool_use_block["id"]
    assert tool_rt["content"][0]["content"] == "Hello from tool!"

    # Final assistant round-trip
    asst2_rt = adapter.to_adapter_format(CanonicalMessage.from_db(by_role["assistant"][1]))
    assert asst2_rt["role"] == "assistant"
    final_text = " ".join(b["text"] for b in asst2_rt["content"] if b.get("type") == "text")
    upstream_text = " ".join(b["text"] for b in upstream_final if b.get("type") == "text")
    assert final_text == upstream_text


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_multi_tool_call_roundtrip(app, adapter, session_factory):
    """Two parallel tool calls: both persisted and round-trip correctly."""
    trajectory_id = "22222222-2222-2222-2222-222222222223"
    auth_headers = {
        "x-api-key": os.getenv("ANTHROPIC_API_KEY", "test"),
        "anthropic-version": "2023-06-01",
    }

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1) force 2 tool calls
            r1 = await client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 100,
                    "messages": [
                        {"role": "user", "content": "Call both hello_world and fizzbuzz tools"},
                    ],
                    "tools": TOOL_DEFS,
                    "tool_choice": {"type": "any"},
                },
                headers={**auth_headers, "X-Trajectory-ID": trajectory_id},
            )
            assert r1.status_code == 200, r1.text
            data1 = r1.json()
            upstream_content = data1["content"]
            tool_use_blocks = [b for b in upstream_content if b["type"] == "tool_use"]
            assert len(tool_use_blocks) == 2, f"Expected 2 tool_use blocks, got {len(tool_use_blocks)}"

            # 2) send both tool results back
            r2 = await client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 100,
                    "messages": [
                        {"role": "user", "content": "Call both hello_world and fizzbuzz tools"},
                        {"role": "assistant", "content": upstream_content},
                        {"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": tool_use_blocks[0]["id"], "content": "world"},
                            {"type": "tool_result", "tool_use_id": tool_use_blocks[1]["id"], "content": "buzzz"},
                        ]},
                    ],
                    "tools": TOOL_DEFS,
                },
                headers={**auth_headers, "X-Trajectory-ID": trajectory_id},
            )
            assert r2.status_code == 200, r2.text

    # Load persisted messages
    async with session_factory() as session:
        result = await session.execute(
            select(Message).order_by(Message.index)
        )
        db_msgs = list(result.scalars())

    roles = [m.role for m in db_msgs]
    assert roles == ["user", "assistant", "tool", "tool", "assistant"]

    # Turn 2: assistant with 2 tool_use blocks
    asst_msg = db_msgs[1]
    tc_blocks = [b for b in asst_msg.content if b["type"] == "tool_call"]
    assert len(tc_blocks) == 2
    tc_names = {b["tool_name"] for b in tc_blocks}
    assert tc_names == {"hello_world", "fizzbuzz"}

    # Round-trip assistant tool calls
    asst_rt = adapter.to_adapter_format(CanonicalMessage.from_db(asst_msg))
    rt_tool_use = [b for b in asst_rt["content"] if b["type"] == "tool_use"]
    for orig, rt in zip(tool_use_blocks, rt_tool_use):
        assert rt["id"] == orig["id"]
        assert rt["name"] == orig["name"]

    # Turn 3: both tool results
    tool_msgs = [m for m in db_msgs if m.role == "tool"]
    assert len(tool_msgs) == 2

    tool_rt_0 = adapter.to_adapter_format(CanonicalMessage.from_db(tool_msgs[0]))
    assert tool_rt_0["content"][0]["tool_use_id"] == tool_use_blocks[0]["id"]
    assert tool_rt_0["content"][0]["content"] == "world"

    tool_rt_1 = adapter.to_adapter_format(CanonicalMessage.from_db(tool_msgs[1]))
    assert tool_rt_1["content"][0]["tool_use_id"] == tool_use_blocks[1]["id"]
    assert tool_rt_1["content"][0]["content"] == "buzzz"
