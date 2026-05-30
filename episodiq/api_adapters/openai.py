import os
from dataclasses import dataclass, field

from episodiq.api_adapters.base import ApiAdapterConfig, ApiAdapter, Route
from episodiq.api_adapters.mixins import OpenAIChatMixin


@dataclass
class OpenAIConfig(ApiAdapterConfig):
    id: str = "openai"
    upstream_base_url: str = field(
        default_factory=lambda: os.getenv("EPISODIQ_OPENAI_BASE_URL", "https://api.openai.com/v1")
    )


class OpenAICompletionsAdapter(OpenAIChatMixin, ApiAdapter):
    @property
    def mount_path(self) -> str:
        return "/openai/v1"

    @property
    def routes(self) -> list[Route]:
        return [
            Route("/chat/completions", ["POST"], "chat_completions"),
        ]


