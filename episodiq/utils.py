"""Reusable utilities for Episodiq."""

import json
import math
from collections import Counter
from collections.abc import Generator
from typing import Any

import numpy as np


def l2_normalize(vector: list[float] | np.ndarray) -> list[float] | np.ndarray:
    """L2 normalize a vector. Returns same type as input."""
    arr = np.asarray(vector)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    if isinstance(vector, np.ndarray):
        return arr
    else:
        return arr.tolist()


def _depth_first_yield(
    data: Any,
    levels_back: int,
    collapse_length: int | None,
    path: list[str],
) -> Generator[str, None, None]:
    """Depth-first traversal of JSON, yielding flattened text lines.

    Based on LlamaIndex JSONReader algorithm. Combines ancestor keys
    with leaf values using spaces. Sorts dict keys for determinism.
    """
    if isinstance(data, (dict, list)):
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        if collapse_length is not None and len(json_str) <= collapse_length:
            new_path = path[-levels_back:] if levels_back else path[:]
            new_path.append(json_str)
            yield " ".join(new_path)
            return
        if isinstance(data, dict):
            for key in sorted(data.keys()):
                new_path = path[:]
                new_path.append(key)
                yield from _depth_first_yield(
                    data[key], levels_back, collapse_length, new_path,
                )
        else:
            for value in data:
                yield from _depth_first_yield(
                    value, levels_back, collapse_length, path,
                )
    else:
        new_path = path[-levels_back:] if levels_back else path[:]
        new_path.append(str(data))
        yield " ".join(new_path)


def json_to_text(
    data: Any,
    levels_back: int = 0,
    collapse_length: int | None = None,
) -> str:
    """Convert JSON-compatible value to flat text for embedding.

    Each leaf becomes a line: ``ancestor_key ... leaf_value``.
    Dict keys are sorted for deterministic output.
    """
    if isinstance(data, str):
        return data
    lines = list(_depth_first_yield(data, levels_back, collapse_length, []))
    return "\n".join(lines)


def categorical_entropy(counts: Counter) -> float:
    """Shannon entropy in bits over a distribution of counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for ROC AUC.

    Returns (lower, upper) bounds at (1-alpha) confidence.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(set(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not aucs:
        return 0.0, 0.0
    return (
        float(np.percentile(aucs, 100 * alpha / 2)),
        float(np.percentile(aucs, 100 * (1 - alpha / 2))),
    )
