"""Reusable utilities for Episodiq."""

import json
import math
import zlib
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

    Args:
        data: JSON-compatible value (dict, list, scalar).
        levels_back: How many ancestor keys to keep (0 = all).
        collapse_length: If set, small sub-trees (≤ this many chars
            when serialised) are emitted as a single collapsed line.
        path: Current ancestor key path (pass [] at top level).
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


def transition_profile_to_vector(
    profile: dict[str, float], dim: int = 2000,
) -> list[float]:
    """Feature-hash a transition profile into a fixed-size raw count vector.

    Each key (transition triplet string like "o:5.a:3.o:12") is mapped to a bin
    via crc32 % dim. Counts accumulate (collisions add). Not normalized —
    euclidean distance on raw counts preserves magnitude signal.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for triplet_str, count in profile.items():
        idx = zlib.crc32(triplet_str.encode()) % dim
        vec[idx] += count
    return vec.tolist()


def trunc_suffix(segs: list[str], n: int | None) -> list[str]:
    """Return last n elements of segs, or all if n is None or len <= n."""
    if n is None or len(segs) <= n:
        return segs
    return segs[-n:]


def levenshtein(a: list[str], b: list[str]) -> float:
    """Normalized token-level Levenshtein similarity: 1 - dist/max(len_a, len_b).

    Returns 0.0 if either sequence is empty.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0
    prev = list(range(nb + 1))
    for i in range(1, na + 1):
        curr = [i] + [0] * nb
        for j in range(1, nb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return 1.0 - prev[nb] / max(na, nb)


def sparse_cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity on sparse dicts (e.g. transition profiles)."""
    dot = sum(a[k] * b[k] for k in a if k in b)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


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


def binomial_margin(p: float, n: int, z: float = 1.96) -> float:
    """Wald margin of error for binomial proportion at confidence z."""
    if n <= 0:
        return 0.0
    return z * math.sqrt(p * (1 - p) / n)


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
    return float(np.percentile(aucs, 100 * alpha / 2)), float(np.percentile(aucs, 100 * (1 - alpha / 2)))


def wilson_bounds(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion.

    Returns (lower, upper) bounds.
    """
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    center = p + z * z / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (center - spread) / denom, (center + spread) / denom

