"""MapReduceSummarizer: chunk-based summarization using LLM map-reduce."""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from episodiq.clustering.annotator.constants import MAP_PROMPT, REDUCE_PROMPT
from episodiq.clustering.annotator.generator import Generator, system_message, user_message

logger = logging.getLogger(__name__)


class MapReduceSummarizer:
    """Summarize long text via map-reduce: split → summarize chunks → combine."""

    def __init__(
        self,
        generator: Generator,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        max_tokens_map: int = 200,
        max_tokens_reduce: int = 300,
    ):
        self._generator = generator
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._max_tokens_map = max_tokens_map
        self._max_tokens_reduce = max_tokens_reduce

    async def summarize(self, text: str) -> str:
        """Summarize text. Short texts get single LLM call; long texts get map + reduce."""
        chunks = self._splitter.split_text(text)
        if len(chunks) <= 1:
            return text

        summaries = [await self._map(chunk) for chunk in chunks]
        return await self._reduce(summaries)

    async def _map(self, chunk: str) -> str:
        return await self._generator.generate(
            [system_message(MAP_PROMPT), user_message(f"<logged_message>\n{chunk}\n</logged_message>")],
            max_tokens=self._max_tokens_map,
        )

    async def _reduce(self, summaries: list[str]) -> str:
        combined = "\n\n---\n\n".join(
            f"{i+1}. {s}" for i, s in enumerate(summaries)
        )
        return await self._generator.generate(
            [system_message(REDUCE_PROMPT), user_message(combined)],
            max_tokens=self._max_tokens_reduce,
        )
