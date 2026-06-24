"""Search configuration.

A single immutable ``SearchConfig`` carries every tunable used by the rankers
and the fusion layer. Defaults encode the normative task contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for one retrieval run.

    Attributes
    ----------
    top_k:
        Size of the final fused ranking after truncation.
    candidate_depth:
        Number of results each modality returns *before* fusion. The fusion
        layer fuses the union of the two candidate lists and only then
        truncates the fused ranking to ``top_k``. ``candidate_depth`` is
        therefore always >= ``top_k`` in practice.
    rrf_k:
        Reciprocal-rank-fusion constant. With two modalities and 1-based ranks
        the maximum possible fused score is ``2 / (rrf_k + 1)``.
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
