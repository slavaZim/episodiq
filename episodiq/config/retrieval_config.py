"""Cascade retrieval pipeline configuration."""

import os
from dataclasses import dataclass

# Single source for ``W = 2w`` — the token window size that must agree
# between the LSH index (WindowMinHashConfig.window) and the agg-shift
# rerank cell (AggShiftConfig.window). Read by both .from_env().
RETRIEVAL_WINDOW_ENV = "EPISODIQ_RETRIEVAL_WINDOW"
DEFAULT_RETRIEVAL_WINDOW = 10


def retrieval_window_from_env() -> int:
    return int(os.getenv(RETRIEVAL_WINDOW_ENV, str(DEFAULT_RETRIEVAL_WINDOW)))


def validate_retrieval_window(window: int) -> None:
    if window % 2 != 0:
        raise ValueError(
            f"retrieval window must be even (W = 2w; LSH and rerank "
            f"share this W and use w = W/2 as the neighborhood radius); "
            f"got {window}",
        )


@dataclass(frozen=True)
class WindowMinHashConfig:
    """MinHash signature parameters for per-window q-gram retrieval.

    LSH banding splits the K-position signature into ``num_bands`` slices of
    ``rows_per_band`` rows each (so ``signature_size == num_bands *
    rows_per_band``). Two windows are "LSH candidates" if any band slice
    hashes to the same value — coarse Jaccard threshold around
    ``(1/num_bands)^(1/rows_per_band)``.
    """
    window: int = 10                # token window size (W)
    q_gram: int = 2                 # q-gram size inside each window
    signature_size: int = 64        # MinHash signature length (= bands * rows)
    num_bands: int = 32
    rows_per_band: int = 2
    seed: int = 42                  # default rng seed for hash family

    def __post_init__(self) -> None:
        if self.num_bands * self.rows_per_band != self.signature_size:
            raise ValueError(
                f"signature_size ({self.signature_size}) must equal "
                f"num_bands * rows_per_band ({self.num_bands} * "
                f"{self.rows_per_band} = {self.num_bands * self.rows_per_band})",
            )
        validate_retrieval_window(self.window)

    @property
    def half_window(self) -> int:
        """``w = W / 2`` — half of the window. Used as the neighborhood
        radius around each window's center step in retrieval."""
        return self.window // 2

    @classmethod
    def from_env(cls) -> "WindowMinHashConfig":
        sig_size = int(os.getenv("EPISODIQ_WMH_SIG_SIZE", "64"))
        num_bands = int(os.getenv("EPISODIQ_WMH_NUM_BANDS", "32"))
        return cls(
            window=retrieval_window_from_env(),
            q_gram=int(os.getenv("EPISODIQ_WMH_QGRAM", "2")),
            signature_size=sig_size,
            num_bands=num_bands,
            rows_per_band=sig_size // num_bands,
            seed=int(os.getenv("EPISODIQ_WMH_SEED", "1464619080")),
        )


@dataclass(frozen=True)
class RetrievalConfig:
    """End-to-end cascade pipeline knobs.

    ``prefetch_n_uniq``: wide LSH pool per query anchor.
    ``jaccard_n_uniq``: post-cross-anchor survivor count fed into
    min-shift rerank. Cross-anchor aggregation operates on the entire
    prefetch pool — no per-anchor cap is applied.
    ``top_k``: final candidates after min-shift.
    """
    aggregation: str = "mean"
    prefetch_n_uniq: int = 200
    jaccard_n_uniq: int = 40
    top_k: int = 25

    def __post_init__(self) -> None:
        if self.aggregation not in ("min_distance", "mean"):
            raise ValueError(
                f"aggregation must be 'min_distance' or 'mean'; "
                f"got {self.aggregation!r}",
            )

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        return cls(
            aggregation=os.getenv("EPISODIQ_CASCADE_AGGREGATION", "mean"),
            prefetch_n_uniq=int(
                os.getenv("EPISODIQ_CASCADE_PREFETCH_N_UNIQ", "200"),
            ),
            jaccard_n_uniq=int(
                os.getenv("EPISODIQ_CASCADE_JACCARD_N_UNIQ", "40"),
            ),
            top_k=int(os.getenv("EPISODIQ_CASCADE_TOP_K", "25")),
        )
