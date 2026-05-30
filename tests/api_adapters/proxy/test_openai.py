"""OpenAI proxy tests: unit tests + VCR pipeline tests with real DB."""

import os

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from episodiq.api_adapters import ApiAdapterConfig, OpenAIConfig, OpenAICompletionsAdapter
from episodiq.api_adapters.base import CanonicalMessage, CanonicalToolMessage
from episodiq.server.app import create_app
from episodiq.storage.postgres.models import Message
from episodiq.workflows import LoggingPipeline
from tests.helpers import MockEmbedder

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "hello_world",
            "description": "Says hello",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fizzbuzz",
            "description": "Does fizzbuzz",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# --- Unit tests (no VCR, no DB) ---

def test_client_auth_header_passthrough():
    """Client Authorization header passes through to upstream."""
    config = ApiAdapterConfig(id="openai", upstream_base_url="https://api.openai.com/v1")
    adapter = OpenAICompletionsAdapter(config)
    headers = adapter.build_request_headers({"authorization": "Bearer client-key-123"})
    assert headers["authorization"] == "Bearer client-key-123"


def test_tool_response_roundtrip():
    """Tool response round-trips through extract → canonical → adapter format."""
    adapter = OpenAICompletionsAdapter(
        ApiAdapterConfig(id="test", upstream_base_url="https://test.local"),
    )
    body = {
        "messages": [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "hello_world", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call_abc", "content": "Hello!"},
        ],
    }
    msgs = adapter.extract_request_messages(body)
    tool_msg = [m for m in msgs if isinstance(m, CanonicalToolMessage)][0]
    roundtripped = adapter.to_adapter_format(tool_msg)

    assert roundtripped["role"] == "tool"
    assert roundtripped["tool_call_id"] == "call_abc"
    assert roundtripped["content"] == "Hello!"

    # tool_name must be resolved from assistant's tool_calls
    assert tool_msg.content[0]["tool_name"] == "hello_world"


# --- VCR pipeline tests (real upstream + real DB) ---

@pytest.fixture
def adapter():
    return OpenAICompletionsAdapter(OpenAIConfig())


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
    """Full conversation: system + user → tool call → tool response → final answer.

    Verifies every message type is persisted and round-trips through
    DB → canonical → adapter format.
    """
    trajectory_id = "11111111-1111-1111-1111-111111111111"
    auth = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'test')}"}

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1) system + user with forced tool call
            r1 = await client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Call the hello_world tool"},
                    ],
                    "tools": [TOOL_DEFS[0]],
                    "tool_choice": {"type": "function", "function": {"name": "hello_world"}},
                },
                headers={**auth, "X-Trajectory-ID": trajectory_id},
            )
            assert r1.status_code == 200, r1.text
            data1 = r1.json()
            upstream_tool_call = data1["choices"][0]["message"]
            tool_calls_raw = upstream_tool_call["tool_calls"]

            # 2) send tool response back, get final answer
            r2 = await client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Call the hello_world tool"},
                        {"role": "assistant", "content": upstream_tool_call.get("content"), "tool_calls": tool_calls_raw},
                        {"role": "tool", "tool_call_id": tool_calls_raw[0]["id"], "content": "Hello from tool!"},
                    ],
                    "tools": [TOOL_DEFS[0]],
                },
                headers={**auth, "X-Trajectory-ID": trajectory_id},
            )
            assert r2.status_code == 200, r2.text
            data2 = r2.json()
            upstream_final = data2["choices"][0]["message"]

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

    # Assistant tool call round-trip
    asst1_rt = adapter.to_adapter_format(CanonicalMessage.from_db(by_role["assistant"][0]))
    for orig, rt in zip(upstream_tool_call["tool_calls"], asst1_rt["tool_calls"]):
        assert rt["id"] == orig["id"]
        assert rt["function"]["name"] == orig["function"]["name"]
        assert rt["function"]["arguments"] == orig["function"]["arguments"]

    # Tool response round-trip
    tool_rt = adapter.to_adapter_format(CanonicalMessage.from_db(by_role["tool"][0]))
    assert tool_rt["role"] == "tool"
    assert tool_rt["tool_call_id"] == tool_calls_raw[0]["id"]
    assert tool_rt["content"] == "Hello from tool!"

    # Final assistant round-trip
    asst2_rt = adapter.to_adapter_format(CanonicalMessage.from_db(by_role["assistant"][1]))
    assert asst2_rt["role"] == upstream_final["role"]
    assert asst2_rt["content"] == (upstream_final.get("content") or "")


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_multi_tool_call_roundtrip(app, adapter, session_factory):
    """Two parallel tool calls: both persisted and round-trip correctly."""
    trajectory_id = "11111111-1111-1111-1111-111111111112"
    auth = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'test')}"}

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1) force 2 tool calls
            r1 = await client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": "Call both hello_world and fizzbuzz tools"},
                    ],
                    "tools": TOOL_DEFS,
                    "tool_choice": "required",
                    "parallel_tool_calls": True,
                },
                headers={**auth, "X-Trajectory-ID": trajectory_id},
            )
            assert r1.status_code == 200, r1.text
            data1 = r1.json()
            upstream_asst = data1["choices"][0]["message"]
            tool_calls_raw = upstream_asst["tool_calls"]
            assert len(tool_calls_raw) == 2, f"Expected 2 tool calls, got {len(tool_calls_raw)}"

            # 2) send both tool responses back
            r2 = await client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": "Call both hello_world and fizzbuzz tools"},
                        {"role": "assistant", "content": upstream_asst.get("content"), "tool_calls": tool_calls_raw},
                        {"role": "tool", "tool_call_id": tool_calls_raw[0]["id"], "content": "world"},
                        {"role": "tool", "tool_call_id": tool_calls_raw[1]["id"], "content": "buzzz"},
                    ],
                    "tools": TOOL_DEFS,
                },
                headers={**auth, "X-Trajectory-ID": trajectory_id},
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

    # Turn 2: assistant with 2 tool calls
    asst_msg = db_msgs[1]
    tc_blocks = [b for b in asst_msg.content if b["type"] == "tool_call"]
    assert len(tc_blocks) == 2
    tc_names = {b["tool_name"] for b in tc_blocks}
    assert tc_names == {"hello_world", "fizzbuzz"}

    # Round-trip assistant tool calls
    asst_rt = adapter.to_adapter_format(CanonicalMessage.from_db(asst_msg))
    for orig, rt in zip(tool_calls_raw, asst_rt["tool_calls"]):
        assert rt["id"] == orig["id"]
        assert rt["function"]["name"] == orig["function"]["name"]
        assert rt["function"]["arguments"] == orig["function"]["arguments"]

    # Turn 3: both tool responses
    tool_msgs = [m for m in db_msgs if m.role == "tool"]
    assert len(tool_msgs) == 2

    tool_rt_0 = adapter.to_adapter_format(CanonicalMessage.from_db(tool_msgs[0]))
    assert tool_rt_0["tool_call_id"] == tool_calls_raw[0]["id"]
    assert tool_rt_0["content"] == "world"

    tool_rt_1 = adapter.to_adapter_format(CanonicalMessage.from_db(tool_msgs[1]))
    assert tool_rt_1["tool_call_id"] == tool_calls_raw[1]["id"]
    assert tool_rt_1["content"] == "buzzz"
