"""Tests for CanonicalMessage validation and to_embedder_format."""

import pytest

from episodiq.api_adapters.base import (
    CanonicalAssistantMessage,
    CanonicalSystemMessage,
    CanonicalToolCall,
    CanonicalToolMessage,
    CanonicalUserMessage,
)


class TestContentValidation:

    def test_valid_text_block(self):
        msg = CanonicalUserMessage.build("hello")
        assert msg.text == "hello"

    def test_valid_tool_response_block(self):
        msg = CanonicalToolMessage.build(tool_call_id="c1", tool_name="func", response="result")
        assert msg.tool_call_ids == ["c1"]

    def test_valid_tool_call_block(self):
        msg = CanonicalAssistantMessage.build(
            adapter_id="test",
            text="calling tool",
            tool_calls=[CanonicalToolCall(id="c1", name="func", arguments={"x": 1})],
        )
        assert msg.tool_calls[0].name == "func"

    def test_unknown_block_type_raises(self):
        with pytest.raises(ValueError, match="Unknown block type"):
            CanonicalUserMessage(content=[{"type": "unknown", "data": "x"}])

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="Unknown block type"):
            CanonicalUserMessage(content=[{"text": "no type field"}])

    def test_text_block_missing_text_raises(self):
        with pytest.raises(ValueError, match="missing 'text'"):
            CanonicalUserMessage(content=[{"type": "text"}])

    def test_tool_call_missing_id_raises(self):
        with pytest.raises(ValueError, match="missing 'id'"):
            CanonicalAssistantMessage(
                content=[{"type": "tool_call", "tool_name": "f", "input": {}}],
                adapter_id="test",
            )

    def test_tool_response_missing_tool_name_raises(self):
        with pytest.raises(ValueError, match="missing 'tool_name'"):
            CanonicalToolMessage(content=[{"type": "tool_response", "id": "c1", "tool_response": "r"}])

    def test_tool_response_missing_tool_response_raises(self):
        with pytest.raises(ValueError, match="missing 'tool_response'"):
            CanonicalToolMessage(content=[{"type": "tool_response", "id": "c1", "tool_name": "f"}])


class TestToEmbedderFormat:

    def test_user_text(self):
        msg = CanonicalUserMessage.build("hello world")
        assert msg.to_embedder_format() == "hello world"

    def test_system_text(self):
        msg = CanonicalSystemMessage.build("be helpful")
        assert msg.to_embedder_format() == "be helpful"

    def test_tool_response_string(self):
        msg = CanonicalToolMessage.build(tool_call_id="c1", tool_name="func", response="result text")
        assert msg.to_embedder_format() == "tool_response: func\nresult text"

    def test_tool_response_dict_sorted(self):
        msg = CanonicalToolMessage.build(tool_call_id="c1", tool_name="func", response={"z": 1, "a": 2})
        assert msg.to_embedder_format() == "tool_response: func\na 2\nz 1"

    def test_tool_response_nested(self):
        msg = CanonicalToolMessage.build(
            tool_call_id="c1", tool_name="search",
            response={"results": [{"title": "A"}, {"title": "B"}], "count": 2},
        )
        result = msg.to_embedder_format()
        assert "tool_response: search" in result
        assert "results title A" in result
        assert "results title B" in result
        assert "count 2" in result

    def test_assistant_with_tool_calls_skips_text(self):
        msg = CanonicalAssistantMessage.build(
            adapter_id="test",
            text="let me call",
            tool_calls=[CanonicalToolCall(id="c1", name="search", arguments={"q": "test"})],
        )
        result = msg.to_embedder_format()
        assert "let me call" not in result
        assert "tool_call: search\nq test" in result

    def test_assistant_text_only(self):
        msg = CanonicalAssistantMessage.build(adapter_id="test", text="hello")
        assert msg.to_embedder_format() == "hello"

    def test_empty_content(self):
        msg = CanonicalAssistantMessage.build(adapter_id="test")
        assert msg.to_embedder_format() == ""
