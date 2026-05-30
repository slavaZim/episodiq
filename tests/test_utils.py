from collections import Counter

import numpy as np
import pytest

from episodiq.utils import (
    bootstrap_auc_ci,
    categorical_entropy,
    json_to_text,
    l2_normalize,
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
