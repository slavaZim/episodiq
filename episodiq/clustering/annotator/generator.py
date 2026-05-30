"""Generator: LLM text generation via adapter transform path."""

import logging
import os
from abc import ABC, abstractmethod

import tenacity
from httpx import HTTPStatusError, NetworkError, TimeoutException

from episodiq.api_adapters.base import (
    ApiAdapter,
    CanonicalMessage,
    CanonicalSystemMessage,
    CanonicalUserMessage,
    Usage,
)

logger = logging.getLogger(__name__)


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutException, NetworkError)):
        return True
    if isinstance(exc, HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 529)
    return False


_retry = tenacity.retry(
    retry=tenacity.retry_if_exception(_retryable),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
    stop=tenacity.stop_after_attempt(5),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
)


class Generator(ABC):
    """Base class for LLM generation: to_adapter_format → _build_body → _send → extract_response_message."""

    def __init__(self, adapter: ApiAdapter, model: str):
        self._adapter = adapter
        self._model = model
        self._total_usage = Usage(input_tokens=0, output_tokens=0)

    @property
    def total_usage(self) -> Usage:
        return self._total_usage

    def _headers(self) -> dict[str, str]:
        """Adapter extra_headers merged with provider API key from env."""
        return dict(self._adapter.config.extra_headers)

    @abstractmethod
    def _build_body(self, messages: list[dict], max_tokens: int) -> dict:
        """Build provider-specific request body from formatted messages."""
        ...

    @abstractmethod
    async def _send(self, body: dict) -> dict:
        """Send completion request, return raw response body."""
        ...

    async def generate(
        self,
        messages: list[CanonicalMessage],
        *,
        max_tokens: int = 256,
    ) -> str:
        """Convert canonical messages → adapter format → send → extract response. Returns text."""
        formatted = [self._adapter.to_adapter_format(m) for m in messages]
        body = self._build_body(formatted, max_tokens)
        response_body = await self._send(body)
        canonical = self._adapter.extract_response_message(response_body)
        if canonical.usage:
            self._total_usage.input_tokens += canonical.usage.input_tokens
            self._total_usage.output_tokens += canonical.usage.output_tokens
        return canonical.text.strip()


class OpenAICompletionsGenerator(Generator):
    """Generator for OpenAI-compatible adapters."""

    id: str = "openai"

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        if key := os.getenv("EPISODIQ_OPENAI_API_KEY"):
            headers.setdefault("Authorization", f"Bearer {key}")
        return headers

    def _build_body(self, messages: list[dict], max_tokens: int) -> dict:
        return {"model": self._model, "messages": messages, "max_tokens": max_tokens}

    @_retry
    async def _send(self, body: dict) -> dict:
        resp = await self._adapter._client.post(
            "/chat/completions",
            json=body,
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()


class AnthropicMessagesGenerator(Generator):
    """Generator for Anthropic adapters."""

    id: str = "anthropic"

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        if key := os.getenv("EPISODIQ_ANTHROPIC_API_KEY"):
            headers.setdefault("x-api-key", key)
        headers.setdefault("anthropic-version", "2023-06-01")
        return headers

    def _build_body(self, messages: list[dict], max_tokens: int) -> dict:
        system = None
        api_messages = []
        for m in messages:
            if m.get("role") == "system":
                system = m["content"]
            else:
                api_messages.append(m)
        body: dict = {"model": self._model, "messages": api_messages, "max_tokens": max_tokens}
        if system:
            body["system"] = system
        return body

    @_retry
    async def _send(self, body: dict) -> dict:
        resp = await self._adapter._client.post(
            "/messages",
            json=body,
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()


def system_message(text: str) -> CanonicalSystemMessage:
    """Build a CanonicalSystemMessage from text."""
    return CanonicalSystemMessage.build(text)


def user_message(text: str) -> CanonicalUserMessage:
    """Build a CanonicalUserMessage from text."""
    return CanonicalUserMessage.build(text)
