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
    max_connections: int = 20
    max_retries: int = 10
    retry_backoff: float = 2.0
    backoff_mode: str = "exponential"

    @classmethod
    def from_env(cls) -> "EmbedderConfig":
        return cls(
            url=os.getenv("EPISODIQ_EMBEDDER_URL", ""),
            chunk_size=int(os.getenv("EPISODIQ_EMBEDDER_CHUNK_SIZE", "8191")),
            batch_size=int(os.getenv("EPISODIQ_EMBEDDER_BATCH_SIZE", "64")),
            model=os.getenv("EPISODIQ_EMBEDDER_MODEL"),
            api_key=os.getenv("EPISODIQ_EMBEDDER_API_KEY"),
            max_connections=int(os.getenv("EPISODIQ_EMBEDDER_MAX_CONNECTIONS", "20")),
            max_retries=int(os.getenv("EPISODIQ_EMBEDDER_MAX_RETRIES", "10")),
            retry_backoff=float(os.getenv("EPISODIQ_EMBEDDER_RETRY_BACKOFF", "2.0")),
            backoff_mode=os.getenv("EPISODIQ_EMBEDDER_BACKOFF_MODE", "exponential"),
        )
