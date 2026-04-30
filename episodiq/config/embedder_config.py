import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbedderConfig:
    """Configuration for embedder service."""

    url: str
    chunk_size: int
    batch_size: int
    model: str | None = None
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "EmbedderConfig":
        return cls(
            url=os.getenv("EPISODIQ_EMBEDDER_URL", ""),
            chunk_size=int(os.getenv("EPISODIQ_EMBEDDER_CHUNK_SIZE", "8191")),
            batch_size=int(os.getenv("EPISODIQ_EMBEDDER_BATCH_SIZE", "64")),
            model=os.getenv("EPISODIQ_EMBEDDER_MODEL"),
            api_key=os.getenv("EPISODIQ_EMBEDDER_API_KEY"),
        )
