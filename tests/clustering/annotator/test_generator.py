"""Tests for Generator class hierarchy."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from episodiq.api_adapters.base import (
    CanonicalAssistantMessage,
    CanonicalSystemMessage,
    CanonicalUserMessage,
    Usage,
)
from episodiq.clustering.annotator.generator import (
    AnthropicMessagesGenerator,
    OpenAICompletionsGenerator,
    _retryable,
    system_message,
    user_message,
)


def _mock_adapter(formatted_messages=None, response_text="generated text"):
    """Create a mock adapter with to_adapter_format and extract_response_message."""
    adapter = MagicMock()
    adapter.to_adapter_format.side_effect = lambda m: {
        "role": "system" if isinstance(m, CanonicalSystemMessage) else "user",
        "content": m.content[0]["text"] if m.content else "",
    }
    adapter.extract_response_message.return_value = CanonicalAssistantMessage(
        content=[{"type": "text", "text": response_text}],
        adapter_id="test",
        external_id="ext-1",
        model="test-model",
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    adapter.config = MagicMock()
    adapter.config.extra_headers = {}
    return adapter


class TestOpenAICompletionsGenerator:

    def test_build_body(self):
        adapter = _mock_adapter()
        gen = OpenAICompletionsGenerator(adapter, "gpt-4")
        body = gen._build_body(
            [{"role": "system", "content": "hi"}, {"role": "user", "content": "hello"}],
            max_tokens=50,
        )
        assert body == {
            "model": "gpt-4",
            "messages": [{"role": "system", "content": "hi"}, {"role": "user", "content": "hello"}],
            "max_tokens": 50,
        }

    async def test_generate_calls_adapter(self):
        adapter = _mock_adapter(response_text="  result  ")
        resp_mock = MagicMock()
        resp_mock.raise_for_status = MagicMock()
        resp_mock.json.return_value = {"choices": [{"message": {"content": "result"}}]}
        adapter._client = MagicMock()
        adapter._client.post = AsyncMock(return_value=resp_mock)

        gen = OpenAICompletionsGenerator(adapter, "gpt-4")
        result = await gen.generate(
            [system_message("prompt"), user_message("input")],
            max_tokens=50,
        )

        assert result == "result"
        assert adapter.to_adapter_format.call_count == 2
        adapter._client.post.assert_called_once()
        assert gen.total_usage.input_tokens == 10
        assert gen.total_usage.output_tokens == 5


class TestAnthropicMessagesGenerator:

    def test_build_body_separates_system(self):
        adapter = _mock_adapter()
        gen = AnthropicMessagesGenerator(adapter, "claude-3")
        body = gen._build_body(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
            max_tokens=100,
        )
        assert body["system"] == "sys"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert body["model"] == "claude-3"
        assert body["max_tokens"] == 100

    def test_build_body_no_system(self):
        adapter = _mock_adapter()
        gen = AnthropicMessagesGenerator(adapter, "claude-3")
        body = gen._build_body(
            [{"role": "user", "content": "hello"}],
            max_tokens=100,
        )
        assert "system" not in body

    async def test_generate_tracks_usage(self):
        adapter = _mock_adapter(response_text="output")
        resp_mock = MagicMock()
        resp_mock.raise_for_status = MagicMock()
        resp_mock.json.return_value = {"content": [{"type": "text", "text": "output"}]}
        adapter._client = MagicMock()
        adapter._client.post = AsyncMock(return_value=resp_mock)

        gen = AnthropicMessagesGenerator(adapter, "claude-3")
        await gen.generate([system_message("p"), user_message("i")])
        await gen.generate([user_message("another")])

        assert gen.total_usage.input_tokens == 20
        assert gen.total_usage.output_tokens == 10


class TestRetryable:

    def test_timeout_is_retryable(self):
        assert _retryable(httpx.ReadTimeout("")) is True
        assert _retryable(httpx.ConnectTimeout("")) is True

    def test_network_error_is_retryable(self):
        assert _retryable(httpx.ReadError("")) is True
        assert _retryable(httpx.ConnectError("")) is True

    def test_retryable_status_codes(self):
        for code in (429, 500, 502, 503, 529):
            resp = httpx.Response(code, request=httpx.Request("POST", "http://x"))
            assert _retryable(httpx.HTTPStatusError("", request=resp.request, response=resp)) is True

    def test_400_is_not_retryable(self):
        resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
        assert _retryable(httpx.HTTPStatusError("", request=resp.request, response=resp)) is False

    def test_random_error_not_retryable(self):
        assert _retryable(ValueError("oops")) is False


def _ok_response():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    return resp


class TestRetry:

    @pytest.fixture(autouse=True)
    def _no_wait(self, monkeypatch):
        """Disable retry wait so tests run instantly."""
        import tenacity
        import episodiq.clustering.annotator.generator as gen_mod
        no_wait_retry = tenacity.retry(
            retry=tenacity.retry_if_exception(_retryable),
            wait=tenacity.wait_none(),
            stop=tenacity.stop_after_attempt(5),
        )
        monkeypatch.setattr(gen_mod.OpenAICompletionsGenerator._send, "retry", no_wait_retry)

    async def test_retries_on_timeout_then_succeeds(self):
        adapter = _mock_adapter(response_text="ok")
        adapter._client = MagicMock()
        adapter._client.post = AsyncMock(side_effect=[httpx.ReadTimeout(""), _ok_response()])

        gen = OpenAICompletionsGenerator(adapter, "gpt-4")
        result = await gen.generate([system_message("p"), user_message("i")])

        assert result == "ok"
        assert adapter._client.post.call_count == 2

    async def test_retries_on_read_error_then_succeeds(self):
        adapter = _mock_adapter(response_text="ok")
        adapter._client = MagicMock()
        adapter._client.post = AsyncMock(side_effect=[httpx.ReadError(""), _ok_response()])

        gen = OpenAICompletionsGenerator(adapter, "gpt-4")
        result = await gen.generate([system_message("p"), user_message("i")])

        assert result == "ok"
        assert adapter._client.post.call_count == 2

    async def test_retries_on_429_then_succeeds(self):
        adapter = _mock_adapter(response_text="ok")
        err_resp = httpx.Response(429, request=httpx.Request("POST", "http://x"))
        adapter._client = MagicMock()
        adapter._client.post = AsyncMock(
            side_effect=[httpx.HTTPStatusError("", request=err_resp.request, response=err_resp), _ok_response()],
        )

        gen = OpenAICompletionsGenerator(adapter, "gpt-4")
        result = await gen.generate([system_message("p"), user_message("i")])

        assert result == "ok"
        assert adapter._client.post.call_count == 2

    async def test_retries_on_500_then_succeeds(self):
        adapter = _mock_adapter(response_text="ok")
        err_resp = httpx.Response(500, request=httpx.Request("POST", "http://x"))
        adapter._client = MagicMock()
        adapter._client.post = AsyncMock(
            side_effect=[httpx.HTTPStatusError("", request=err_resp.request, response=err_resp), _ok_response()],
        )

        gen = OpenAICompletionsGenerator(adapter, "gpt-4")
        result = await gen.generate([system_message("p"), user_message("i")])

        assert result == "ok"
        assert adapter._client.post.call_count == 2

    async def test_no_retry_on_400(self):
        adapter = _mock_adapter()
        err_resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
        adapter._client = MagicMock()
        adapter._client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError("bad", request=err_resp.request, response=err_resp),
        )

        gen = OpenAICompletionsGenerator(adapter, "gpt-4")
        with pytest.raises(httpx.HTTPStatusError):
            await gen.generate([system_message("p"), user_message("i")])

        assert adapter._client.post.call_count == 1


class TestHelpers:

    def test_system_message(self):
        msg = system_message("hello")
        assert isinstance(msg, CanonicalSystemMessage)
        assert msg.content[0]["text"] == "hello"

    def test_user_message(self):
        msg = user_message("world")
        assert isinstance(msg, CanonicalUserMessage)
        assert msg.content[0]["text"] == "world"
