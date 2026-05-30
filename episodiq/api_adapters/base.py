from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from episodiq.storage.postgres.models import Message

import httpx
from fastapi import Request
from fastapi.responses import Response

from episodiq.api_adapters.trajectory_handler import (
    DefaultTrajectoryHandler,
    TrajectoryHandler,
)
from episodiq.utils import json_to_text


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class CanonicalToolCall:
    id: str
    name: str
    arguments: str | dict


_REQUIRED_KEYS = {
    "text": ("text",),
    "tool_call": ("id", "tool_name", "input"),
    "tool_response": ("id", "tool_name", "tool_response"),
}


@dataclass
class CanonicalMessage(ABC):
    role: Role
    content: list[dict]

    def __post_init__(self):
        for block in self.content:
            block_type = block.get("type")
            required = _REQUIRED_KEYS.get(block_type)
            if required is None:
                raise ValueError(f"Unknown block type: {block_type!r}")
            for key in required:
                if key not in block:
                    raise ValueError(f"{block_type} block missing '{key}': {block}")

    @property
    def cluster_type(self) -> str | None:
        """'action' or 'observation'."""
        if self.role in (Role.USER, Role.TOOL):
            return "observation"
        if self.role == Role.ASSISTANT:
            return "action"
        return None

    @property
    def category(self) -> str | None:
        """Tool name or 'text'."""
        for block in self.content:
            if block.get("type") in ("tool_call", "tool_response"):
                return block["tool_name"]
        if any(b.get("type") == "text" for b in self.content):
            return "text"
        return None

    @property
    def text(self) -> str:
        """Extract text from the single text block."""
        block = next((b for b in self.content if b.get("type") == "text"), None)
        return block["text"] if block else ""

    def to_embedder_format(self) -> str:
        """Serialize content blocks to flat text for embedding.

        Uses json_to_text (depth-first flattening) for tool arguments
        and responses so that arbitrarily nested JSON becomes readable
        key-value lines suitable for embedding models.

        For assistant messages with tool_calls, text blocks are skipped
        (they're usually uninformative filler like "Let me check that").
        """
        has_tool_calls = any(b.get("type") == "tool_call" for b in self.content)
        parts = []
        for block in self.content:
            match block.get("type"):
                case "text":
                    if not has_tool_calls:
                        parts.append(block["text"])
                case "tool_call":
                    args_text = json_to_text(block["input"])
                    parts.append(f"tool_call: {block['tool_name']}\n{args_text}")
                case "tool_response":
                    resp_text = json_to_text(block["tool_response"])
                    parts.append(f"tool_response: {block['tool_name']}\n{resp_text}")
        return "\n".join(parts)

    @classmethod
    def from_db(cls, msg: Message) -> "CanonicalMessage":
        """Reconstruct canonical message from ORM Message."""
        match msg.role:
            case "user":
                return CanonicalUserMessage(content=msg.content)
            case "assistant":
                return CanonicalAssistantMessage(content=msg.content, adapter_id="")
            case "tool":
                return CanonicalToolMessage(content=msg.content)
            case "system":
                return CanonicalSystemMessage(content=msg.content)
            case _:
                return CanonicalUserMessage(content=msg.content)


@dataclass
class CanonicalSystemMessage(CanonicalMessage):
    role: Role = field(default=Role.SYSTEM, init=False)

    @classmethod
    def build(cls, text: str) -> "CanonicalSystemMessage":
        return cls(content=[{"type": "text", "text": text}])


@dataclass
class CanonicalUserMessage(CanonicalMessage):
    role: Role = field(default=Role.USER, init=False)

    @classmethod
    def build(cls, text: str) -> "CanonicalUserMessage":
        return cls(content=[{"type": "text", "text": text}])


@dataclass
class CanonicalToolMessage(CanonicalMessage):
    role: Role = field(default=Role.TOOL, init=False)

    @property
    def tool_call_ids(self) -> list[str]:
        return [b["id"] for b in self.content if b.get("type") == "tool_response"]

    @classmethod
    def build(cls, tool_call_id: str, tool_name: str, response: str | dict) -> "CanonicalToolMessage":
        return cls(content=[{"type": "tool_response", "id": tool_call_id, "tool_name": tool_name, "tool_response": response}])


@dataclass
class CanonicalAssistantMessage(CanonicalMessage):
    adapter_id: str
    role: Role = field(default=Role.ASSISTANT, init=False)
    external_id: str | None = None
    model: str | None = None
    usage: Usage | None = None

    @property
    def tool_calls(self) -> list[CanonicalToolCall] | None:
        calls = [
            CanonicalToolCall(id=b["id"], name=b["tool_name"], arguments=b["input"])
            for b in self.content if b.get("type") == "tool_call"
        ]
        return calls or None

    @classmethod
    def build(
        cls,
        *,
        adapter_id: str,
        text: str = "",
        tool_calls: list[CanonicalToolCall] | None = None,
        external_id: str | None = None,
        model: str | None = None,
        usage: Usage | None = None,
    ) -> "CanonicalAssistantMessage":
        content: list[dict] = []
        if text:
            content.append({"type": "text", "text": text})
        if tool_calls:
            for tc in tool_calls:
                content.append({
                    "type": "tool_call", "id": tc.id,
                    "tool_name": tc.name, "input": tc.arguments,
                })
        return cls(
            content=content, adapter_id=adapter_id,
            external_id=external_id, model=model, usage=usage,
        )




HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class Route(NamedTuple):
    path: str
    methods: list[str]
    operation_id: str


@dataclass
class ApiAdapterConfig:
    id: str
    upstream_base_url: str
    timeout: float = 120.0
    extra_headers: dict[str, str] = field(default_factory=dict)


class ApiAdapter(ABC):
    """Utilities for working with a provider. Doesn't know about workflow."""

    def __init__(
        self,
        config: ApiAdapterConfig,
        trajectory_handler: TrajectoryHandler | None = None,
    ):
        self.config = config
        self.trajectory_handler = trajectory_handler or DefaultTrajectoryHandler()
        self._client: httpx.AsyncClient | None = None

    @property
    def id(self) -> str:
        return self.config.id

    @property
    @abstractmethod
    def mount_path(self) -> str:
        ...

    @property
    @abstractmethod
    def routes(self) -> list[Route]:
        ...

    @abstractmethod
    def extract_request_messages(self, body: dict) -> list[CanonicalSystemMessage | CanonicalUserMessage | CanonicalAssistantMessage | CanonicalToolMessage]:
        """Extract all messages from request body (system, user, assistant, tool)."""
        ...

    @abstractmethod
    def extract_response_message(self, body: dict) -> CanonicalAssistantMessage:
        ...

    @abstractmethod
    def to_adapter_format(self, message: CanonicalMessage) -> dict:
        """Convert canonical message to provider-specific dict format."""
        ...

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.config.upstream_base_url,
            timeout=self.config.timeout,
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


    def build_request_headers(self, original_headers: dict[str, str]) -> dict[str, str]:
        headers = {
            k: v
            for k, v in original_headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS | {"host", "content-length"}
        }
        headers.update(self.config.extra_headers)
        return headers

    def build_response_headers(self, upstream_headers: httpx.Headers) -> dict[str, str]:
        # Strip hop-by-hop + content-encoding (httpx already decompresses)
        skip = HOP_BY_HOP_HEADERS | {"content-encoding"}
        headers = {
            k: v
            for k, v in upstream_headers.items()
            if k.lower() not in skip
        }
        return headers

    # why ?     
    def transform_request(self, body: dict[str, Any]) -> dict[str, Any]:
        return body

    def transform_response(self, body: bytes) -> bytes:
        return body

    async def forward(
        self,
        request: Request,
        body: dict[str, Any],
    ) -> Response:
        if self._client is None:
            raise RuntimeError("Adapter not started. Call startup() first.")

        upstream_path = request.url.path.removeprefix(self.mount_path)

        upstream_response = await self._client.request(
            method=request.method,
            url=upstream_path,
            json=self.transform_request(body),
            headers=self.build_request_headers(dict(request.headers)),
        )

        return Response(
            content=self.transform_response(upstream_response.content),
            status_code=upstream_response.status_code,
            headers=self.build_response_headers(upstream_response.headers),
        )
