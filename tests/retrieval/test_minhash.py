"""Tests for MinHash signature + Jaccard estimation."""

import pytest

from episodiq.retrieval.minhash import (
    MinHasher,
    jaccard_estimate,
    max_pool_jaccard_per_key,
    ngrams,
)


class TestNgrams:
    def test_empty_sequence(self):
        assert ngrams([], 3) == set()

    def test_shorter_than_n(self):
        assert ngrams([1, 2], 3) == set()

    def test_basic_trigrams(self):
        # 4 tokens → 2 trigrams (sliding window step 1)
        out = ngrams([1, 2, 3, 4], 3)
        assert len(out) == 2

    def test_distinct_ngrams_distinct_hashes(self):
        a = ngrams([1, 2, 3, 4], 3)
        b = ngrams([5, 6, 7, 8], 3)
        assert a.isdisjoint(b)

    def test_noise_kept_as_regular_token(self):
        """Noise tokens (-1) take part in n-grams alongside ordinary tokens."""
        out_with_noise = ngrams([1, -1, 2, 3], 3)
        out_without = ngrams([1, 2, 3], 3)
        # Different sequences → different n-grams
        assert out_with_noise != out_without
        assert len(out_with_noise) == 2  # (1,-1,2) and (-1,2,3)


class TestMinHasher:
    def test_signature_size_matches_k(self):
        mh = MinHasher(k=64, seed=1)
        sig = mh.signature({1, 2, 3, 4, 5})
        assert len(sig) == 64

    def test_empty_set_returns_none(self):
        mh = MinHasher(k=32)
        assert mh.signature(set()) is None

    def test_deterministic_for_same_seed(self):
        a = MinHasher(k=16, seed=42).signature({10, 20, 30})
        b = MinHasher(k=16, seed=42).signature({10, 20, 30})
        assert a == b

    def test_different_seed_different_signature(self):
        a = MinHasher(k=16, seed=1).signature({10, 20, 30})
        b = MinHasher(k=16, seed=2).signature({10, 20, 30})
        assert a != b

    def test_signature_from_tokens(self):
        mh = MinHasher(k=32, seed=1)
        # Same n-gram extraction → same signature
        a = mh.signature_from_tokens([1, 2, 3, 4, 5], n=3)
        b = mh.signature(ngrams([1, 2, 3, 4, 5], 3))
        assert a == b


class TestJaccardEstimate:
    def test_identical_signatures_equal_one(self):
        assert jaccard_estimate([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0

    def test_no_overlap_equals_zero(self):
        assert jaccard_estimate([1, 2, 3, 4], [5, 6, 7, 8]) == 0.0

    def test_partial_overlap(self):
        # 2 of 4 positions match → 0.5
        assert jaccard_estimate([1, 2, 3, 4], [1, 2, 9, 9]) == pytest.approx(0.5)

    def test_empty_returns_zero(self):
        assert jaccard_estimate([], [1, 2, 3]) == 0.0

    def test_mismatched_lengths_returns_zero(self):
        assert jaccard_estimate([1, 2], [1, 2, 3]) == 0.0


class TestMaxPoolJaccardPerKey:
    def test_keeps_max_per_key(self):
        query = [1, 2, 3, 4]
        # key 'a': two paths, the second is a closer match
        corpus = [
            ("a", [9, 9, 9, 9]),  # 0% match
            ("a", [1, 2, 9, 9]),  # 50% match
            ("b", [1, 2, 3, 9]),  # 75% match
        ]
        result = max_pool_jaccard_per_key(query, corpus)
        assert result == {"a": 0.5, "b": 0.75}

    def test_skips_mismatched_signature_length(self):
        query = [1, 2, 3, 4]
        corpus = [
            ("a", [1, 2]),         # wrong length, skipped
            ("a", [1, 2, 3, 4]),   # full match
        ]
        result = max_pool_jaccard_per_key(query, corpus)
        assert result == {"a": 1.0}

    def test_empty_query_returns_empty_dict(self):
        assert max_pool_jaccard_per_key([], [("a", [1])]) == {}

    def test_empty_corpus_returns_empty_dict(self):
        assert max_pool_jaccard_per_key([1, 2], []) == {}
