import asyncio
import logging

import httpx
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from episodiq.config.embedder_config import EmbedderConfig
from episodiq.utils import l2_normalize

logger = logging.getLogger(__name__)

MAX_RETRIES = 10
RETRY_BACKOFF = 2.0


class EmbedderClient:
    """Low-level HTTP transport for embedding API."""

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.config.url,
            headers=headers,
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def embed(self, texts: list[str], dims: int) -> list[list[float]]:
        payload: dict = {
            "input": texts,
            "dimensions": dims,
        }
        if self.config.model:
            payload["model"] = self.config.model

        for attempt in range(MAX_RETRIES):
            try:
                response = await self._client.post(
                    "/v1/embeddings",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = RETRY_BACKOFF * (2 ** attempt)
                logger.warning("embed failed (attempt %d/%d), retrying in %.1fs", attempt + 1, MAX_RETRIES, delay, exc_info=True)
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")


class Embedder:
    """High-level: chunk, batch, average, L2 normalize."""

    def __init__(self, client: EmbedderClient, dims: int):
        self._client = client
        self._dims = dims

    @property
    def config(self) -> EmbedderConfig:
        return self._client.config

    async def startup(self) -> None:
        await self._client.startup()

    async def shutdown(self) -> None:
        await self._client.shutdown()

    async def embed_text(self, text: str) -> list[float]:
        """Chunk text, embed in batches, average, and L2 normalize."""
        cfg = self._client.config
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=int(cfg.chunk_size * 0.1),
        )
        chunks = splitter.split_text(text)

        all_embeddings = []
        for i in range(0, len(chunks), cfg.batch_size):
            batch = chunks[i : i + cfg.batch_size]
            batch_embeddings = await self._client.embed(batch, self._dims)
            all_embeddings.extend(batch_embeddings)

        avg = np.mean(all_embeddings, axis=0).tolist()
        return l2_normalize(avg)
