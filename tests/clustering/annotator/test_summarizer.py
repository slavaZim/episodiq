"""Tests for MapReduceSummarizer."""

from tests.clustering.annotator.conftest import MockGenerator
from episodiq.clustering.annotator.summarizer import MapReduceSummarizer


class TestMapReduceSummarizer:

    async def test_short_text_returns_as_is(self):
        gen = MockGenerator()
        summarizer = MapReduceSummarizer(gen, chunk_size=800)
        result = await summarizer.summarize("Short text here.")
        assert result == "Short text here."
        assert len(gen.calls) == 0

    async def test_long_text_map_reduce(self):
        gen = MockGenerator(default="chunk summary")
        summarizer = MapReduceSummarizer(gen, chunk_size=50, chunk_overlap=10)
        long_text = "word " * 200
        result = await summarizer.summarize(long_text)
        assert result == "chunk summary"
        assert len(gen.calls) >= 3  # at least 2 map + 1 reduce

    async def test_empty_text(self):
        gen = MockGenerator()
        summarizer = MapReduceSummarizer(gen)
        result = await summarizer.summarize("")
        assert result == ""
        assert len(gen.calls) == 0

