import os
from dataclasses import dataclass, field

from episodiq.api_adapters.base import (
    ApiAdapterConfig,
    ApiAdapter,
    CanonicalAssistantMessage,
    CanonicalMessage,
    CanonicalSystemMessage,
    CanonicalToolCall,
    CanonicalToolMessage,
    CanonicalUserMessage,
    Role,
    Route,
    Usage,
)


@dataclass
class AnthropicConfig(ApiAdapterConfig):
    id: str = "anthropic"
    upstream_base_url: str = field(
        default_factory=lambda: os.getenv(
            "EPISODIQ_ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"
        )
    )


class AnthropicMessagesAdapter(ApiAdapter):
    @property
    def mount_path(self) -> str:
        return "/anthropic/v1"

    @property
    def routes(self) -> list[Route]:
        return [
            Route("/messages", ["POST"], "messages"),
        ]

    def extract_request_messages(self, body: dict) -> list[CanonicalSystemMessage | CanonicalUserMessage | CanonicalAssistantMessage | CanonicalToolMessage]:
        result = []

        # System is top-level, not in messages array
        system = body.get("system")
        if system:
            text = system if isinstance(system, str) else system[0].get("text", "")
            result.append(CanonicalSystemMessage.build(text))

        for msg in body.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")

            if role == Role.USER.value:
                # Could be string or content blocks with tool_result
                if isinstance(content, str):
                    result.append(CanonicalUserMessage.build(content))
                elif isinstance(content, list):
                    tool_results = [b for b in content if b.get("type") == "tool_result"]
                    if tool_results:
                        for tr in tool_results:
                            tool_call_id = tr["tool_use_id"]
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
                                response=tr.get("content", ""),
                            ))
                    else:
                        text = next((b["text"] for b in content if b.get("type") == "text"), "")
                        result.append(CanonicalUserMessage.build(text))

            elif role == Role.ASSISTANT.value:
                if isinstance(content, str):
                    result.append(CanonicalAssistantMessage.build(
                        adapter_id=self.id, text=content,
                    ))
                elif isinstance(content, list):
                    text = " ".join(b["text"] for b in content if b.get("type") == "text")
                    tool_calls = [
                        CanonicalToolCall(id=b["id"], name=b["name"], arguments=b["input"])
                        for b in content if b.get("type") == "tool_use"
                    ]
                    result.append(CanonicalAssistantMessage.build(
                        adapter_id=self.id,
                        text=text,
                        tool_calls=tool_calls or None,
                    ))

        return result

    def extract_response_message(self, body: dict) -> CanonicalAssistantMessage:
        content_blocks = body.get("content", [])
        text = " ".join(b["text"] for b in content_blocks if b.get("type") == "text")

        tool_calls = None
        tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
        if tool_use_blocks:
            tool_calls = [
                CanonicalToolCall(id=b["id"], name=b["name"], arguments=b["input"])
                for b in tool_use_blocks
            ]

        usage = None
        usage_data = body.get("usage")
        if usage_data:
            usage = Usage(
                input_tokens=usage_data["input_tokens"],
                output_tokens=usage_data["output_tokens"],
            )

        return CanonicalAssistantMessage.build(
            adapter_id=self.id,
            text=text,
            tool_calls=tool_calls,
            external_id=body.get("id"),
            model=body.get("model"),
            usage=usage,
        )

    def to_adapter_format(self, message: CanonicalMessage) -> dict:
        if isinstance(message, CanonicalToolMessage):
            block = message.content[0]
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": block["tool_response"],
                }],
            }

        if isinstance(message, CanonicalAssistantMessage):
            content = []
            if message.text:
                content.append({"type": "text", "text": message.text})
            if message.tool_calls:
                for tc in message.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments if isinstance(tc.arguments, dict) else {},
                    })
            return {"role": "assistant", "content": content}

        if isinstance(message, CanonicalSystemMessage):
            return {"role": "system", "content": message.text}

        # CanonicalUserMessage
        return {"role": "user", "content": message.text}
