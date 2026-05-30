import json

from episodiq.api_adapters.base import (
    CanonicalMessage,
    CanonicalSystemMessage,
    CanonicalUserMessage,
    CanonicalToolCall,
    CanonicalToolMessage,
    CanonicalAssistantMessage,
    Role,
    Usage,
)


class OpenAIChatMixin:
    @staticmethod
    def _normalize_content(content: str | list | None) -> str:
        """Normalize OpenAI content to plain string.

        OpenAI API accepts content as string or list of content parts:
        [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        # list of content parts — extract text blocks
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts)

    def extract_request_messages(self, body: dict) -> list[CanonicalSystemMessage | CanonicalUserMessage | CanonicalAssistantMessage | CanonicalToolMessage]:
        """Extract all messages from OpenAI-format request."""
        messages = body.get("messages", [])
        result = []
        for msg in messages:
            content = self._normalize_content(msg.get("content"))
            match msg.get("role"):
                case Role.SYSTEM.value:
                    result.append(CanonicalSystemMessage.build(content))
                case Role.USER.value:
                    result.append(CanonicalUserMessage.build(content))
                case Role.ASSISTANT.value:
                    tool_calls = None
                    if msg.get("tool_calls"):
                        tool_calls = [
                            CanonicalToolCall(
                                id=tc["id"],
                                name=tc["function"]["name"],
                                arguments=tc["function"].get("arguments", {}),
                            )
                            for tc in msg["tool_calls"]
                        ]
                    result.append(CanonicalAssistantMessage.build(
                        adapter_id=self.id,
                        text=content,
                        tool_calls=tool_calls,
                    ))
                case Role.TOOL.value:
                    tool_call_id = msg["tool_call_id"]
                    name = ""
                    for prev in reversed(result):
                        if isinstance(prev, CanonicalAssistantMessage) and prev.tool_calls:
                            for tc in prev.tool_calls:
                                if tc.id == tool_call_id:
                                    name = tc.name
                                    break
                            if name:
                                break
                    result.append(CanonicalToolMessage.build(
                        tool_call_id=tool_call_id,
                        tool_name=name,
                        response=content,
                    ))
        return result

    def to_adapter_format(self, message: CanonicalMessage) -> dict:
        """Convert canonical message to OpenAI chat format."""
        msg: dict = {"role": message.role.value}
        if isinstance(message, CanonicalToolMessage):
            # OpenAI tool messages: one message per tool response
            block = message.content[0]
            msg["tool_call_id"] = block["id"]
            msg["content"] = block["tool_response"]
        elif isinstance(message, CanonicalAssistantMessage):
            msg["content"] = message.text
            if message.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                    },
                    }
                    for tc in message.tool_calls
                ]
        else:
            msg["content"] = message.text
        return msg

    def extract_response_message(self, body: dict) -> CanonicalAssistantMessage:
        choice = body["choices"][0]
        msg = choice["message"]
        usage_data = body.get("usage")
        usage = None
        if usage_data:
            usage = Usage(
                input_tokens=usage_data["prompt_tokens"],
                output_tokens=usage_data["completion_tokens"],
            )
        tool_calls = None
        if msg.get("tool_calls"):
            tool_calls = []
            for tc in msg["tool_calls"]:
                raw_args = tc["function"]["arguments"]
                if isinstance(raw_args, str):
                    arguments = json.loads(raw_args) if raw_args else {}
                else:
                    arguments = raw_args if isinstance(raw_args, dict) else {}
                tool_calls.append(CanonicalToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=arguments,
                ))
        return CanonicalAssistantMessage.build(
            adapter_id=self.id,
            text=msg.get("content", ""),
            tool_calls=tool_calls,
            external_id=body.get("id"),
            model=body.get("model"),
            usage=usage,
        )
