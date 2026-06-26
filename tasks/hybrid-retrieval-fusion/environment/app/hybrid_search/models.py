"""Typed domain models shared across the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Document:
    """A corpus document."""

    doc_id: str
    text: str


@dataclass(frozen=True)
class Query:
    """A search query with a precomputed dense embedding.

    The dense ranker scores documents by cosine similarity between
    ``embedding`` and the shipped corpus vectors, so the embedding is carried
    on the query object rather than recomputed from ``text``.
    """

    query_id: str
    text: str
    embedding: Sequence[float]


@dataclass
class RankingEntry:
    """A single ranked document within one ranking.

    ``modality``, ``source_rank``, and ``raw_score`` are retained for internal
    bookkeeping and diagnostics.
    """

    doc_id: str
    score: float
    modality: str
    source_rank: int
    raw_score: float
