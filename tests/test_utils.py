from collections import Counter

import numpy as np
import pytest

from episodiq.utils import (
    binomial_margin,
    bootstrap_auc_ci,
    categorical_entropy,
    json_to_text,
    l2_normalize,
    levenshtein,
)


class TestL2Normalize:
    def test_unit_vector(self):
        # [3, 4] -> norm = 5 -> [0.6, 0.8]
        result = l2_normalize([3.0, 4.0])
        assert result == [0.6, 0.8]

    def test_zero_vector(self):
        result = l2_normalize([0.0, 0.0])
        assert result == [0.0, 0.0]


class TestJsonToText:
    def test_flat_dict(self):
        assert json_to_text({"b": 2, "a": 1}) == "a 1\nb 2"

    def test_nested_dict(self):
        result = json_to_text({"x": {"y": "z"}})
        assert result == "x y z"

    def test_list_values(self):
        result = json_to_text({"tags": ["foo", "bar"]})
        assert result == "tags foo\ntags bar"

    def test_string_passthrough(self):
        assert json_to_text("hello world") == "hello world"

    def test_number(self):
        assert json_to_text(42) == "42"

    def test_empty_dict(self):
        assert json_to_text({}) == ""

    def test_deeply_nested(self):
        data = {"a": {"b": {"c": "deep"}}}
        assert json_to_text(data) == "a b c deep"

    def test_collapse_length(self):
        data = {"a": [1, 2], "b": {"x": "y"}}
        result = json_to_text(data, collapse_length=20)
        assert 'a [1, 2]' in result
        assert 'b {"x": "y"}' in result

    def test_levels_back(self):
        data = {"a": {"b": {"c": "val"}}}
        result = json_to_text(data, levels_back=1)
        # Only 1 ancestor kept: "c val" instead of "a b c val"
        assert result == "c val"

    def test_dict_keys_sorted(self):
        """Keys are sorted for deterministic output."""
        result1 = json_to_text({"z": 1, "a": 2})
        result2 = json_to_text({"a": 2, "z": 1})
        assert result1 == result2 == "a 2\nz 1"



class TestCategoricalEntropy:
    def test_single_category(self):
        assert categorical_entropy(Counter({"a": 10})) == 0.0

    def test_uniform_two(self):
        assert categorical_entropy(Counter({"a": 5, "b": 5})) == pytest.approx(1.0)

    def test_uniform_four(self):
        assert categorical_entropy(Counter({"a": 1, "b": 1, "c": 1, "d": 1})) == pytest.approx(2.0)

    def test_empty(self):
        assert categorical_entropy(Counter()) == 0.0

    def test_skewed(self):
        # 9:1 split → low entropy
        h = categorical_entropy(Counter({"a": 9, "b": 1}))
        assert 0 < h < 0.6


class TestLevenshtein:
    def test_identical(self):
        assert levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_completely_different(self):
        assert levenshtein(["a"], ["b"]) == 0.0

    def test_empty_both(self):
        assert levenshtein([], []) == 0.0

    def test_one_empty(self):
        assert levenshtein(["a", "b"], []) == 0.0
        assert levenshtein([], ["a", "b"]) == 0.0

    def test_one_edit(self):
        # 1 substitution in 3 tokens → sim = 1 - 1/3 ≈ 0.667
        sim = levenshtein(["a", "b", "c"], ["a", "x", "c"])
        assert sim == pytest.approx(2 / 3)

    def test_different_lengths(self):
        # ["a","b","c"] vs ["a","b"] → 1 deletion, max_len=3 → sim = 2/3
        sim = levenshtein(["a", "b", "c"], ["a", "b"])
        assert sim == pytest.approx(2 / 3)


class TestBinomialMargin:
    def test_known_value(self):
        # p=0.5, n=100, z=1.96 → 1.96 * sqrt(0.25/100) = 1.96 * 0.05 = 0.098
        assert binomial_margin(0.5, 100) == pytest.approx(0.098, abs=1e-4)

    def test_zero_n(self):
        assert binomial_margin(0.5, 0) == 0.0

    def test_extreme_p(self):
        assert binomial_margin(0.0, 1000) == 0.0
        assert binomial_margin(1.0, 1000) == 0.0

    def test_large_n_small_margin(self):
        m = binomial_margin(0.5, 10000)
        assert m < 0.01


class TestBootstrapAucCi:
    def test_perfect_auc_tight_ci(self):
        """Perfect separation → CI near 1.0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        lo, hi = bootstrap_auc_ci(y_true, y_score)
        assert lo > 0.8
        assert hi <= 1.0

    def test_random_auc_wide_ci(self):
        """Random scores → CI spans around 0.5."""
        rng = np.random.RandomState(123)
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = rng.rand(100)
        lo, hi = bootstrap_auc_ci(y_true, y_score)
        assert lo < 0.5 < hi

    def test_single_class_returns_zeros(self):
        """All same class → no valid bootstrap samples → (0, 0)."""
        y_true = np.array([1, 1, 1, 1])
        y_score = np.array([0.5, 0.6, 0.7, 0.8])
        lo, hi = bootstrap_auc_ci(y_true, y_score)
        assert lo == 0.0
        assert hi == 0.0


