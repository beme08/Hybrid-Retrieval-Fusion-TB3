"""Search configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for one retrieval run.

    Attributes
    ----------
    top_k:
        Requested final result count.
    candidate_depth:
        Requested per-modality candidate count.
    rrf_k:
        Fusion tuning constant.
    bm25_k1, bm25_b:
        Standard Okapi BM25 term-frequency saturation and length-normalization
        parameters.
    """

    top_k: int = 10
    candidate_depth: int = 50
    rrf_k: int = 60
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.candidate_depth <= 0:
            raise ValueError("candidate_depth must be positive")
        if self.candidate_depth < self.top_k:
            raise ValueError("candidate_depth must be >= top_k")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
