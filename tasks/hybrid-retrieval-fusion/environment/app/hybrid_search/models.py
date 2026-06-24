"""Typed domain models shared across the retrieval pipeline.

These dataclasses are the only objects that move between the rankers, the
fusion layer, and the serializer. Keeping them typed (rather than passing bare
dicts or tuples) is what lets the fusion layer stay decoupled from the rankers:
fusion consumes precomputed ``RankingEntry`` lists and never touches a ranker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Document:
    """A corpus document.

    ``doc_id`` is an **opaque string**. It is never parsed, split, normalized,
    or compared by substring anywhere in the pipeline; it is only ever used for
    exact equality and as a deterministic tie-break key.
    """

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

    Several of these fields are *internal* and are deliberately stripped before
    serialization (see :mod:`hybrid_search.serialization`):

    - ``doc_id``    — opaque document identifier (emitted).
    - ``score``     — the value used to sort this ranking. For a modality
      ranking this equals the modality's raw score; for a fused ranking it is
      the RRF fused score (emitted).
    - ``modality``  — ``"bm25"``, ``"dense"``, or ``"fused"`` (internal).
    - ``source_rank`` — 1-based rank of this entry within its own ranking
      (internal; the public schema re-derives rank from list position).
    - ``raw_score`` — the original modality score that produced this entry
      (internal; retained for diagnostics and never emitted).
    """

    doc_id: str
    score: float
    modality: str
    source_rank: int
    raw_score: float
